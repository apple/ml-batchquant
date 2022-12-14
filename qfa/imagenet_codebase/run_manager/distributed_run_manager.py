#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

import os
import json
import time
import math
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import horovod.torch as hvd

# from imagenet_codebase.utils import *
from qfa.imagenet_codebase.utils import get_net_info, cross_entropy_with_label_smoothing, \
    cross_entropy_loss_with_soft_target, DistributedMetric, DistributedTensor, list_mean
from qfa.imagenet_codebase.run_manager import RunConfig
from qfa.imagenet_codebase.data_providers.base_provider import LoaderConfig

from qfa.utils import accuracy, AverageMeter


class DistributedRunManager:

    def __init__(self, path, net, run_config: RunConfig, hvd_compression, backward_steps=1, is_root=False, init=True):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.is_root = is_root

        self.best_acc = 0.0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        self.net.cuda()
        cudnn.benchmark = True
        if init and self.is_root:
            self.net.init_model(self.run_config.model_init)
        if self.is_root:
            # print net info
            net_info = get_net_info(self.net, self.run_config.data_provider.data_shape)
            with open('%s/net_info.txt' % self.path, 'w') as fout:
                fout.write(json.dumps(net_info, indent=4) + '\n')
                fout.write(self.net.module_str)

        # criterion
        self.train_criterion = self.run_config.build_train_criterion()
        self.kd_criterion = self.run_config.build_kd_criterion()
        self.test_criterion = self.run_config.build_test_criterion()

        # metrics
        self.train_metrics = self.run_config.build_train_metrics()
        self.test_metrics = self.run_config.build_test_metrics()
        
        self.reset_activation_statistics()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            net_params = [
                self.net.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.net.get_parameters(keys, mode='include'),  # parameters without weight decay
            ]
        else:
            net_params = self.net.weight_parameters()
        self.optimizer = self.run_config.build_optimizer(net_params)
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer, named_parameters=self.net.named_parameters(), compression=hvd_compression,
            backward_passes_per_step=backward_steps,
        )

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get('_save_path', None) is None:
            save_path = os.path.join(self.path, 'checkpoint', str(hvd.rank()))
            os.makedirs(save_path, exist_ok=True)
            self.__dict__['_save_path'] = save_path
        return self.__dict__['_save_path']

    @property
    def logs_path(self):
        if self.__dict__.get('_logs_path', None) is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__['_logs_path'] = logs_path
        return self.__dict__['_logs_path']

    def write_log(self, log_str, prefix='valid', should_print=True):
        if self.is_root:
            """ prefix: valid, train, test """
            if prefix in ['valid', 'test']:
                with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                    fout.write(log_str + '\n')
                    fout.flush()
            if prefix in ['valid', 'test', 'train']:
                with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                    if prefix in ['valid', 'test']:
                        fout.write('=' * 10)
                    fout.write(log_str + '\n')
                    fout.flush()
            else:
                with open(os.path.join(self.logs_path, '%s.txt' % prefix), 'a') as fout:
                    fout.write(log_str + '\n')
                    fout.flush()
            if should_print:
                print(log_str)

    """ save & load model & save_config & broadcast """

    def save_config(self):
        if self.is_root:
            run_save_path = os.path.join(self.path, 'run.config')
            if not os.path.isfile(run_save_path):
                json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
                print('Run configs dump to %s' % run_save_path)

            net_save_path = os.path.join(self.path, 'net.config')
            if not os.path.isfile(net_save_path):
                json.dump(self.net.config, open(net_save_path, 'w'), indent=4)
                print('Network configs dump to %s' % net_save_path)

    def save_model(self, checkpoint=None, is_best=False, model_name=None, is_last=False):
        if self.is_root or is_last:
            if checkpoint is None:
                checkpoint = {'state_dict': self.net.state_dict()}

            if model_name is None:
                model_name = 'checkpoint.pth.tar'

            latest_fname = os.path.join(self.save_path, 'latest.txt')
            model_path = os.path.join(self.save_path, model_name)
            with open(latest_fname, 'w') as _fout:
                _fout.write(model_path + '\n')
            torch.save(checkpoint, model_path)

        # Saving at every worker to make checkpoints available for continue training
        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        if self.is_root:
            latest_fname = os.path.join(self.save_path, 'latest.txt')
            if model_fname is None and os.path.exists(latest_fname):
                with open(latest_fname, 'r') as fin:
                    model_fname = fin.readline()
                    if model_fname[-1] == '\n':
                        model_fname = model_fname[:-1]
            try:
                if model_fname is None or not os.path.exists(model_fname):
                    model_fname = '%s/checkpoint.pth.tar' % self.save_path
                    with open(latest_fname, 'w') as fout:
                        fout.write(model_fname + '\n')
                print("=> loading checkpoint '{}'".format(model_fname))

                if torch.cuda.is_available():
                    checkpoint = torch.load(model_fname)
                else:
                    checkpoint = torch.load(model_fname, map_location='cpu')

                self.net.load_state_dict(checkpoint['state_dict'])

                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch'] + 1
                if 'best_acc' in checkpoint:
                    self.best_acc = checkpoint['best_acc']
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])

                self.write_log("=> loaded checkpoint '{}'".format(model_fname), 'valid')
            except Exception:
                self.write_log('fail to load checkpoint from %s' % self.save_path, 'valid')

    def broadcast(self):
        self.start_epoch = hvd.broadcast(torch.LongTensor(1).fill_(self.start_epoch)[0], 0, name='start_epoch').item()
        self.best_acc = hvd.broadcast(torch.Tensor(1).fill_(self.best_acc)[0], 0, name='best_acc').item()
        hvd.broadcast_parameters(self.net.state_dict(), 0)
        hvd.broadcast_optimizer_state(self.optimizer, 0)

    """ train & validate """

    def validate(self, epoch=0, is_test=True, run_str='', net=None, data_loader=None, no_logs=False):
        if net is None:
            net = self.net
        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.valid_loader
        

        net.eval()

        losses = DistributedMetric('val_loss')
        metric_meters = None
        accumulators = None

        with torch.no_grad():
            with tqdm(total=len(data_loader),
                      desc='Validate Epoch #{} {}'.format(epoch + 1, run_str),
                      disable=no_logs or not self.is_root) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.cuda(), labels.cuda()
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    metrics = self.test_metrics(output, labels)
                    if metric_meters is None:
                        metric_meters = {k: DistributedMetric(k) for k in metrics.keys() if 'accum' not in k}
                    accum_syncer = {k: DistributedMetric(k) for k in metrics.keys() if 'accum' in k}
                    losses.update(loss, images.size(0))
                    for name, value in metrics.items():
                        if 'accum' in name:
                            accum_syncer[name].update(value, images.size(0))
                        else:
                            metric_meters[name].update(value, images.size(0))
                    postfix = {
                        'loss': losses.avg.item(),
                        'img_size': images.size(2),
                    }
                    for name, meter in metric_meters.items():
                        postfix[name] = meter.avg.item()    
                    accum_dict = {name: meter.sum.numpy() for name, meter in accum_syncer.items()}
                    accumulators = self.run_config.accumulate(accum_dict, accumulators)
                    partial_results = self.run_config.digest_accumulator({}, accumulators)
                    for name, value in partial_results.items():
                        postfix[name] = value
                    t.set_postfix(postfix)
                    t.update(1)
        results = {k: v.avg.item() for k, v in metric_meters.items()}
        results = self.run_config.digest_accumulator(results, accumulators)
        return losses.avg.item(), results

    def validate_all_resolution(self, epoch=0, is_test=True, net=None):
        if net is None:
            net = self.net
        metric_dict = defaultdict(list)
        if isinstance(self.run_config.data_provider.image_size, list):
            img_size_list, loss_list, top1_list, top5_list = [], [], [], []
            for img_size in self.run_config.data_provider.image_size:
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(net=net)
                loss, metrics = self.validate(epoch, is_test, net=net)
                loss_list.append(loss)
                for k, v in metrics.items():
                    metric_dict[k].append(v)
            return img_size_list, loss_list, metric_dict
        else:
            loss, metrics = self.validate(epoch, is_test, net=net)
            for k, v in metrics.items():
                metric_dict[k].append(v)
            return [self.run_config.data_provider.active_img_size], [loss], metric_dict

    def train_one_epoch(self, args, epoch, warmup_epochs=5, warmup_lr=0):
        self.net.train()
        self.run_config.train_loader.sampler.set_epoch(epoch)
        LoaderConfig.EPOCH = epoch

        nBatch = len(self.run_config.train_loader)

        losses = DistributedMetric('train_loss')
        metric_meters = None
        accumulators = None
        data_time = AverageMeter()

        with tqdm(total=nBatch,
                  desc='Train Epoch #{}'.format(epoch + 1),
                  disable=not self.is_root) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch - warmup_epochs, i, nBatch)

                images, labels = images.cuda(), labels.cuda()
                target = labels

                # soft target
                if args.teacher_model is not None:
                    args.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)

                # compute output
                output = self.net(images)
                loss = self.train_criterion(output, labels)

                if args.teacher_model is None:
                    loss_type = 'ce'
                else:
                    kd_loss = self.kd_criterion(args, soft_logits, soft_label)
                    loss = args.kd_ratio * kd_loss + loss
                    loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                metrics = self.train_metrics(output, target)
                accumulators = self.run_config.accumulate(metrics, accumulators)
                if metric_meters is None:
                    metric_meters = {k: DistributedMetric(k) for k in metrics.keys() if 'accum' not in k}
                losses.update(loss, images.size(0))
                for name, value in metrics.items():
                    if 'accum' not in k:
                        metric_meters[name].update(value, images.size(0))
                postfix = {
                    'loss': losses.avg,
                    'img_size': images.size(2),
                    'lr': new_lr,
                    'loss_type': loss_type,
                    'data_time': data_time.avg,
                }
                for name, meter in metric_meters.items():
                    postfix[name] = meter.avg.item()
                t.set_postfix(postfix)
                t.update(1)
                end = time.time()
        results = {k: v.avg.item() for k, v in metric_meters.items()}
        results = self.run_config.digest_accumulator(results, accumulators)
        return losses.avg.item(), results

    def train(self, args, warmup_epochs=5, warmup_lr=0):
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epochs):
            train_loss, train_metric_dict = self.train_one_epoch(args, epoch, warmup_epochs, warmup_lr)
            img_size, val_loss, val_metric_dict = self.validate_all_resolution(epoch, is_test=False)

            target_metric = val_metric_dict['target_metric']
            is_best = list_mean(target_metric) > self.best_acc
            self.best_acc = max(self.best_acc, list_mean(target_metric))
            if self.is_root:
                val_log = '[{0}/{1}]\tloss {2:.3f}\ttarget metric {3:.3f} ({4:.3f})\tloss {train_loss:.3f}\t'. \
                    format(epoch + 1 - warmup_epochs, self.run_config.n_epochs, list_mean(val_loss),
                           self.best_acc, train_loss=train_loss)
                for name, metric in val_metric_dict.items():
                    val_log += '\t{name} {metric:.3f}'.format(name=name, metric=list_mean(metric))
                val_log += '\t'
                for i_s, v_a in zip(img_size, val_top1):
                    val_log += '(%d, %.3f), ' % (i_s, v_a)
                self.write_log(val_log, prefix='valid', should_print=False)

                self.save_model({
                    'epoch': epoch,
                    'best_acc': self.best_acc,
                    'optimizer': self.optimizer.state_dict(),
                    'state_dict': self.net.state_dict(),
                }, is_best=is_best)

    def reset_running_statistics(self, net=None):
        from qfa.elastic_nn.utils import set_running_statistics
        if net is None:
            net = self.net
        num_gpu = hvd.size()
        batch_size = self.run_config.train_batch_size
        n_images = batch_size * num_gpu * 30
        sub_train_loader = self.run_config.random_sub_train_loader(n_images, batch_size,
                                                                   num_replicas=num_gpu, rank=hvd.rank())
        set_running_statistics(net, sub_train_loader, distributed=True)

    def reset_activation_statistics(self, net=None):
        from qfa.elastic_nn.utils import set_activation_statistics
        if net is None:
            net = self.net
        num_gpu = hvd.size()
        batch_size = self.run_config.train_batch_size
        n_images = batch_size * num_gpu * 30
        sub_train_loader = self.run_config.random_sub_train_loader(n_images, batch_size,
                                                                   num_replicas=num_gpu, rank=hvd.rank())
        set_activation_statistics(net, sub_train_loader, distributed=True)
