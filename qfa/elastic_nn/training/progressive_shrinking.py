#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

import json
import torch.nn as nn
from tqdm import tqdm
import random
import os
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
import horovod.torch as hvd

from qfa.utils import accuracy, AverageMeter, download_url
from qfa.imagenet_codebase.utils import DistributedMetric, DistributedTensor, list_mean, cross_entropy_loss_with_soft_target, \
    subset_mean, int2list
from qfa.imagenet_codebase.data_providers.base_provider import LoaderConfig
from qfa.imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager


def validate(run_manager, epoch=0, is_test=True, image_size_list=None,
             width_mult_list=None, ks_list=None, expand_ratio_list=None, depth_list=None, bits_list=None, additional_setting=None):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    if image_size_list is None:
        image_size_list = int2list(run_manager.run_config.data_provider.image_size, 1)
    if width_mult_list is None:
        width_mult_list = [i for i in range(len(dynamic_net.width_mult_list))]
    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list
    if bits_list is None:
        bits_list = dynamic_net.bits_list

    subnet_settings = []
    for b in bits_list:
        for k in ks_list:
            for agg in [min, max]:
                w = agg(width_mult_list)
                d = agg(depth_list)
                e = agg(expand_ratio_list)
                img_size = agg(image_size_list)
                subnet_settings.append([{
                    'image_size': img_size,
                    'wid': w,
                    'b': b,
                    'd': d,
                    'e': e,
                    'ks': k,
                }, 'R%s-W%s-D%s-E%s-K%s-B%d' % (img_size, w, d, e, k, b)])

    if additional_setting is not None:
        subnet_settings += additional_setting

    losses_of_subnets = []
    
    valid_log = ''
    metric_dict = defaultdict(list)
    for setting, name in subnet_settings:
        run_manager.write_log('-' * 30 + ' Validate %s ' % name + '-' * 30, 'train', should_print=False)
        run_manager.run_config.data_provider.assign_active_img_size(setting.pop('image_size'))
        dynamic_net.set_active_subnet(**setting)
        run_manager.write_log(dynamic_net.module_str, 'train', should_print=False)

        run_manager.reset_running_statistics(dynamic_net)
        loss, metrics = run_manager.validate(epoch=epoch, is_test=is_test, run_str=name, net=dynamic_net)
        losses_of_subnets.append(loss)
        for k, v in metrics.items():
            metric_dict[k].append(v)
        valid_log += '%s (%.3f), ' % ('target_metric', metrics['target_metric'])

    return list_mean(losses_of_subnets), {k: list_mean(v) for k, v in metric_dict.items()}, valid_log


def train_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.net

    # switch to train mode
    dynamic_net.train()
    run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    LoaderConfig.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric('train_loss')
    metric_meters = None
    accumulators = None

    with tqdm(total=nBatch,
              desc='Train Epoch #{}'.format(epoch + 1),
              disable=not run_manager.is_root) as t:
        end = time.time()
        for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
            data_time.update(time.time() - end)
            if epoch < warmup_epochs:
                new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                    run_manager.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                )
            else:
                new_lr = run_manager.run_config.adjust_learning_rate(
                    run_manager.optimizer, epoch - warmup_epochs, i, nBatch
                )

            labels = labels.cuda()
            images = {k: v.cuda() for k, v in images.items()}
            target = labels

            # soft target
            # If key 'teacher' is present, teacher operate at fixed resolution
            if args.kd_ratio > 0 and 'teacher' in images:
                args.teacher_model.eval()
                with torch.no_grad():
                    soft_logits = args.teacher_model(images['teacher']).detach()
                    soft_label = F.softmax(soft_logits, dim=1)

            # clear gradients
            run_manager.optimizer.zero_grad()

            # Prepare to collect stats
            loss_of_subnets = []
            metric_dict = defaultdict(list)
            subnet_str = ''

            # If using sandwich rule
            if args.sandwich:
                for batch_idx in ['min', 'max']:
                    # soft target
                    # If key 'teacher' is not present, teacher operate at same resolution at student
                    if args.kd_ratio > 0 and 'teacher' not in images:
                        args.teacher_model.eval()
                        with torch.no_grad():
                            soft_logits = args.teacher_model(images[batch_idx]).detach()
                            soft_label = F.softmax(soft_logits, dim=1)

                    subnet_settings = dynamic_net.set_sandwich_subnet(batch_idx)
                    subnet_str += '%s: ' % batch_idx
                    subnet_str += 'R_%d, ' % images[batch_idx].size(2)
                    subnet_str += ','.join(['%s_%s' % (
                        key, '%.1f' % subset_mean(val, list(range(len(val)))) if isinstance(val, list) else val
                    ) for key, val in subnet_settings.items()])

                    output = run_manager.net(images[batch_idx])
                    if args.kd_ratio == 0:
                        loss = run_manager.train_criterion(output, labels)
                        loss_type = 'ce'
                    else:
                        kd_loss = run_manager.kd_criterion(args, output, soft_logits, soft_label)
                        if 'only' in args.kd_type:
                            loss = kd_loss
                        else:
                            loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
                            loss = loss * (2 / (args.kd_ratio + 1))
                        loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

                    # measure accuracy and record loss
                    metrics = run_manager.train_metrics(output, target)
                    for k, v in metrics.items():
                        if 'accum' not in k:
                            subnet_str += ',%s_%.1f' % (k, v) + ' || '
                    loss_of_subnets.append(loss)
                    for k, v in metrics.items():
                        metric_dict[k].append(v)

                    loss.backward()

            # compute output
            for batch_idx in range(args.dynamic_batch_size):
                # soft target
                # If key 'teacher' is not present, teacher operate at same resolution at student
                if args.kd_ratio > 0 and 'teacher' not in images:
                    args.teacher_model.eval()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images[batch_idx]).detach()
                        soft_label = F.softmax(soft_logits, dim=1)

                # set random seed before sampling
                if args.independent_distributed_sampling:
                    subnet_seed = os.getpid() + time.time()
                else:
                    subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, batch_idx, 0))
                random.seed(subnet_seed)
                res = images[batch_idx].size(2)
                subnet_settings = dynamic_net.sample_active_subnet(res, subnet_seed)
                subnet_str += '%s: ' % str(batch_idx)
                subnet_str += 'R_%d, ' % images[batch_idx].size(2)
                subnet_str += ','.join(['%s_%s' % (
                    key, '%.1f' % subset_mean(val, list(range(len(val)))) if isinstance(val, list) else val
                ) for key, val in subnet_settings.items()])

                output = run_manager.net(images[batch_idx])
                if args.kd_ratio == 0:
                    loss = run_manager.train_criterion(output, labels)
                    loss_type = 'ce'
                else:
                    kd_loss = run_manager.kd_criterion(args, output, soft_logits, soft_label)
                    if 'only' in args.kd_type:
                        loss = kd_loss
                    else:
                        loss = run_manager.train_criterion(output, labels)
                        loss += args.kd_ratio * kd_loss
                        loss = loss * (2 / (args.kd_ratio + 1))
                    loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

                # measure accuracy and record loss
                metrics = run_manager.train_metrics(output, target)
                for k, v in metrics.items():
                    if 'accum' not in k:
                        subnet_str += ',%s_%.1f' % (k, v) + ' || '
                loss_of_subnets.append(loss)
                for k, v in metrics.items():
                    metric_dict[k].append(v)
                loss.backward()
                
            torch.nn.utils.clip_grad_norm_(run_manager.net.parameters(), 500)

            run_manager.optimizer.step()

            losses.update(list_mean(loss_of_subnets), images[0].size(0))
            if metric_meters is None:
                metric_meters = {k: DistributedMetric(k) for k in metric_dict.keys() if 'accum' not in k}
            accum_syncer = {k: DistributedMetric(k) for k in metric_dict.keys() if 'accum' in k}
            metric_meter_dict = {name: list_mean(value) for name, value in metric_dict.items()}
            for name, value in metric_meter_dict.items():
                if 'accum' in name:
                    accum_syncer[name].update(value, images[0].size(0))
                else:
                    metric_meters[name].update(value, images[0].size(0))
            accum_dict = {name: meter.sum.numpy() for name, meter in accum_syncer.items()}
            accumulators = run_manager.run_config.accumulate(accum_dict, accumulators)
            partial_results = run_manager.run_config.digest_accumulator({}, accumulators)
            postfix = {
                'loss': losses.avg.item(),
                'lr': new_lr,
                'loss_type': loss_type,
                'seed': str(subnet_seed),
                'str': subnet_str,
                'data_time': data_time.avg,
            }
            for name, meter in metric_meters.items():
                postfix[name] = meter.avg.item()
            for name, value in partial_results.items():
                postfix[name] = value
            t.set_postfix(postfix)
            t.update(1)
            end = time.time()
    results = {k: v.avg.item() for k, v in metric_meters.items()}
    results = run_manager.run_config.digest_accumulator(results, accumulators)
    return losses.avg.item(), results


def train(run_manager, args, validate_func=None):
    if validate_func is None:
        validate_func = validate

    for epoch in range(run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs):
        train_loss, train_metric_dict = train_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr)

        if (epoch + 1) % args.validation_frequency == 0:
            # validate under train mode
            val_loss, val_metric_dict, _val_log = validate_func(run_manager, epoch=epoch, is_test=True)
            # best_acc
            target_metric = val_metric_dict['target_metric']
            is_best = target_metric > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, target_metric)
            if run_manager.is_root:
                val_log = 'Valid [{0}/{1}] loss={2:.3f}, target metric={3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - args.warmup_epochs, run_manager.run_config.n_epochs, val_loss, target_metric,
                           run_manager.best_acc)
                val_log += ', Train loss {loss:.3f}\t'.format(loss=train_loss)
                for k, v in val_metric_dict.items():
                    val_log += ', Train {name} {value:.3f}\t'.format(name=k, value=v)
                val_log += _val_log
                run_manager.write_log(val_log, 'valid', should_print=False)

            # Saving at every worker to make checkpoints available for continue training
            run_manager.save_model({
                'epoch': epoch,
                'best_acc': run_manager.best_acc,
                # 'optimizer': run_manager.optimizer.state_dict(),
                'state_dict': run_manager.net.state_dict(),
            }, is_best=is_best, model_name='checkpoint%s.pth.tar' % str(epoch).zfill(4))


def load_models(run_manager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location='cpu')['state_dict']
    dynamic_net.load_weights_from_net(init)
    dynamic_net.cuda()
    run_manager.write_log('Loaded init from %s' % model_path, 'valid')


def supporting_elastic_quantize(train_func, run_manager, args, validate_func_dict):
    from qfa.elastic_nn.utils import set_activation_statistics

    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    # load pretrained models
    validate_func_dict['bits_list'] = sorted(dynamic_net.bits_list)
    
    bits_list = dynamic_net.bits_list.copy()
    
    run_manager.write_log(
        '-' * 30 + 'Supporting Bitwidth: %d -> %d' %
        (bits_list[0], bits_list[-1]) + '-' * 30, 'valid'
    )

    # add expand list constraints
    validate_func_dict['bits_list'] = sorted(bits_list)
    val_loss, val_metric_dict, _val_log = validate(run_manager, **validate_func_dict)
    val_log = 'Val loss {loss:.3f}\t'.format(loss=val_loss)
    for k, v in val_metric_dict.items():
        val_log += ', Train {name} {value:.3f}\t'.format(name=k, value=v)
    val_log += _val_log
    run_manager.write_log(val_log, 'valid')

    # train
    train_func(
        run_manager, args,
        lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
    )

    # next stage & reset
    run_manager.start_epoch = 0
    run_manager.best_acc = 0.0
    if isinstance(run_manager, DistributedRunManager):
        run_manager.broadcast()

    # save and validate
    run_manager.save_model(model_name='quantize.pth.tar', is_last=True)
    validate_func_dict['bits_list'] = sorted(dynamic_net.bits_list)
    val_loss, val_metric_dict, _val_log = validate(run_manager, **validate_func_dict)
    val_log = 'Val loss {loss:.3f}\t'.format(loss=val_loss)
    for k, v in val_metric_dict.items():
        val_log += ', Train {name} {value:.3f}\t'.format(name=k, value=v)
    val_log += _val_log
    run_manager.write_log(val_log, 'valid')
