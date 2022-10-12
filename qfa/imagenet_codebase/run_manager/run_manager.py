#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

import os
import time
import json
import math
from collections import defaultdict
from tqdm import tqdm

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision

from ..utils import get_net_info, cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
from qfa.utils import AverageMeter, accuracy


class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 mixup_alpha,
                 model_init, validation_frequency, print_frequency):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.mixup_alpha = mixup_alpha

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        elif self.lr_schedule_type == 'power':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = self.init_lr * (1. - T_cur / T_total) ** 1.0
        elif self.lr_schedule_type is None:
            lr = self.init_lr
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self.calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None, cache=True):
        return self.data_provider.build_sub_train_loader(n_images, batch_size, num_worker, num_replicas, rank, cache)

    """ criterion """

    def build_train_criterion(self):
        raise NotImplementedError

    def build_kd_criterion(self):
        raise NotImplementedError

    def build_test_criterion(self):
        raise NotImplementedError

    """ metrics """

    def build_train_metrics(self):
        raise NotImplementedError

    def build_test_metrics(self):
        raise NotImplementedError
        
    def accumulate(self, metrics, accumulator):
        return accumulator
        
    def digest_accumulator(self, metrics, accumulator):
        return metrics

    """ optimizer """

    def build_optimizer(self, net_params):
        if self.no_decay_keys is not None:
            assert isinstance(net_params, list) and len(net_params) == 2
            net_params = [
                {'params': net_params[0], 'weight_decay': self.weight_decay},
                {'params': net_params[1], 'weight_decay': 0},
            ]
        else:
            net_params = [{'params': net_params, 'weight_decay': self.weight_decay}]

        opt_param = {} if self.opt_param is None else self.opt_param
        if self.opt_type == 'sgd':
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov)
        elif self.opt_type == 'adam':
            optimizer = torch.optim.Adam(net_params, self.init_lr)
        else:
            raise NotImplementedError
        return optimizer


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, init=True, measure_latency=None, no_gpu=False, mix_prec=None):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.mix_prec = mix_prec

        self.best_acc = 0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device('cuda:0')
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        # initialize model (default)
        if init:
            self.network.init_model(run_config.model_init)

        # criterion
        self.train_criterion = self.run_config.build_train_criterion()
        self.kd_criterion = self.run_config.build_kd_criterion()
        self.test_criterion = self.run_config.build_test_criterion()

        # metrics
        self.train_metrics = self.run_config.build_train_metrics()
        self.test_metrics = self.run_config.build_test_metrics()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            net_params = [
                self.network.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.network.get_parameters(keys, mode='include'),  # parameters without weight decay
            ]
        else:
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = self.network.parameters()
        self.optimizer = self.run_config.build_optimizer(net_params)

        if mix_prec is not None:
            from apex import amp
            self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level=mix_prec)

        self.net = torch.nn.DataParallel(self.net)

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get('_save_path', None) is None:
            save_path = os.path.join(self.path, 'checkpoint')
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

    @property
    def network(self):
        if isinstance(self.net, nn.DataParallel):
            return self.net.module
        else:
            return self.net

    @network.setter
    def network(self, new_val):
        if isinstance(self.net, nn.DataParallel):
            self.net.module = new_val
        else:
            self.net = new_val

    def write_log(self, log_str, prefix='valid', should_print=True):
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

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.network.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        if self.mix_prec is not None:
            from apex import amp
            checkpoint['amp'] = amp.state_dict()

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
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

            self.network.load_state_dict(checkpoint['state_dict'])

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.mix_prec is not None and 'amp' in checkpoint:
                from apex import amp
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self):
        """ dump run_config and net_config to the model_folder """
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.network.config, open(net_save_path, 'w'), indent=4)
        print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def validate(self, epoch=0, is_test=True, run_str='', net=None, data_loader=None, no_logs=False):
        if net is None:
            net = self.net
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                print("Using val loader")
                data_loader = self.run_config.valid_loader

        net.eval()

        losses = AverageMeter()
        metric_meters = None
        accumulators = None

        with torch.no_grad():
            with tqdm(total=len(data_loader),
                      desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    metrics = self.test_metrics(output, labels)
                    if metric_meters is None:
                        metric_meters = {k: AverageMeter() for k in metrics.keys() if 'accum' not in k}
                    accum_syncer = {k: AverageMeter() for k in metrics.keys() if 'accum' in k}
                    losses.update(loss.item(), images.size(0))
                    for name, value in metrics.items():
                        if 'accum' in name:
                            accum_syncer[name].update(value, images.size(0))
                        else:
                            metric_meters[name].update(value.item(), images.size(0))
                    postfix = {
                        'loss': losses.avg,
                        'img_size': images.size(2),
                    }
                    for name, meter in metric_meters.items():
                        if 'accum' not in name:
                            postfix[name] = meter.avg
                    accum_dict = {name: meter.sum for name, meter in accum_syncer.items()}
                    accumulators = self.run_config.accumulate(accum_dict, accumulators)
                    partial_results = self.run_config.digest_accumulator({}, accumulators)
                    for name, value in partial_results.items():
                        postfix[name] = value
                    t.set_postfix(postfix)
                    t.update(1)
            results = {k: v.avg for k, v in metric_meters.items()}
            results = self.run_config.digest_accumulator(results, accumulators)
        return losses.avg, results

    def validate_all_resolution(self, epoch=0, is_test=True, net=None):
        if net is None:
            net = self.network
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

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        metric_meters = None
        data_time = AverageMeter()

        with tqdm(total=nBatch,
                  desc='Train Epoch #{}'.format(epoch + 1)) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch - warmup_epochs, i, nBatch)

                images, labels = images.to(self.device), labels.to(self.device)
                target = labels

                # soft target
                soft_label = None
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

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                if self.mix_prec is not None:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                metrics = self.train_metrics(output, target)
                if metric_meters is None:
                    metric_meters = {k: AverageMeter() for k in metrics.keys()}
                losses.update(loss.item(), images.size(0))
                for name, value in metrics.items():
                    metric_meters[name].update(value[0].item(), images.size(0))
                postfix = {
                    'loss': losses.avg,
                    'img_size': images.size(2),
                    'lr': new_lr,
                    'loss_type': loss_type,
                    'data_time': data_time.avg,
                }
                for name, meter in metric_meters.items():
                    postfix[name] = meter.avg
                t.set_postfix(postfix)
                t.update(1)
                end = time.time()
        return losses.avg, {k: v.avg for k, v in metric_meters.items()}

    def train(self, args, warmup_epoch=0, warmup_lr=0):
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
            train_loss, train_metric_dict = self.train_one_epoch(args, epoch, warmup_epoch, warmup_lr)

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                img_size, val_loss, val_metric_dict = self.validate_all_resolution(epoch=epoch, is_test=False)

                target_metric = val_metric_dict['target_metric']
                is_best = np.mean(target_metric) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(target_metric))
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttarget metric {3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - warmup_epoch, self.run_config.n_epochs,
                           np.mean(val_loss), np.mean(target_metric), self.best_acc)
                for name, metric in val_metric_dict.items():
                    val_log += '\t{name} {metric:.3f}'.format(name=name, metric=np.mean(metric))
                val_log += '\t'
                for i_s, v_a in zip(img_size, val_acc):
                    val_log += '(%d, %.3f), ' % (i_s, v_a)
                self.write_log(val_log, prefix='valid', should_print=False)
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.network.state_dict(),
            }, is_best=is_best)

    def reset_running_statistics(self, net=None):
        from qfa.elastic_nn.utils import set_running_statistics
        if net is None:
            net = self.network
        batch_size = self.run_config.train_batch_size
        n_images = batch_size * 30
        sub_train_loader = self.run_config.random_sub_train_loader(n_images, batch_size)
        set_running_statistics(net, sub_train_loader)

    def reset_activation_statistics(self, net=None):
        from qfa.elastic_nn.utils import set_activation_statistics
        if net is None:
            net = self.net
        batch_size = self.run_config.train_batch_size
        n_images = batch_size * 30
        sub_train_loader = self.run_config.random_sub_train_loader(n_images, batch_size)
        set_activation_statistics(net, sub_train_loader)
