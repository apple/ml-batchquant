#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_providers.imagenet import ImagenetDataProvider

from .run_manager import RunConfig
from .run_manager import RunManager

from ..utils import cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
from qfa.utils import accuracy


class ImagenetRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224, **kwargs):
        super(ImagenetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size
            )
        return self.__dict__['_data_provider']

    def build_train_criterion(self):
        if isinstance(self.mixup_alpha, float):
            return cross_entropy_loss_with_soft_target
        elif self.label_smoothing > 0:
            return lambda pred, target: \
                cross_entropy_with_label_smoothing(pred, target, self.label_smoothing)
        else:
            return nn.CrossEntropyLoss()
        
    def build_kd_criterion(self):
        def criterion(args, output, soft_logits, soft_label):
            if args.kd_type == 'ce':
                kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
            else:
                kd_loss = F.mse_loss(output, soft_logits)
            return kd_loss
        return criterion

    def build_test_criterion(self):
        return nn.CrossEntropyLoss()
    
    def build_train_metrics(self):
        def metrics(output, target):
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            return {
                'target_metric': acc1,
                'acc1': acc1,
                'acc5': acc5
            }
        return metrics
    
    def build_test_metrics(self):
        def metrics(output, target):
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            return {
                'target_metric': acc1,
                'acc1': acc1,
                'acc5': acc5
            }
        return metrics


class DistributedImageNetRunConfig(ImagenetRunConfig):

    def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=64, test_batch_size=64, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=8, resize_scale=0.08, distort_color='tf', image_size=224,
                 **kwargs):
        super(DistributedImageNetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency, n_worker, resize_scale, distort_color, image_size,
            **kwargs
        )

        self._num_replicas = kwargs['num_replicas']
        self._rank = kwargs['rank']

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
                num_replicas=self._num_replicas, rank=self._rank,
            )
        return self.__dict__['_data_provider']
