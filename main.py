#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

import argparse
import numpy as np
import time
import os
import random
from shutil import copyfile

import horovod.torch as hvd
import torch

from qfa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
from qfa.elastic_nn.networks import OFAMobileNetV3, QFAMobileNetV3
from qfa.imagenet_codebase.run_manager import DistributedImageNetRunConfig
from qfa.imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager
from qfa.imagenet_codebase.data_providers.base_provider import LoaderConfig
from qfa.utils import download_url
from qfa.elastic_nn.training.progressive_shrinking import load_models
from qfa.elastic_nn.utils import set_activation_statistics


# Unify all teacher specific arg diffs here, e.g. img size, interp method
def get_args(task, phase, teacher):
    args = argparse.Namespace()
    args.task = task
    args.phase = phase
    args.teacher = teacher

    args.base_batch_size = 64  # 128  # 128 has much worse result
    args.valid_size = 10000

    args.opt_type = 'sgd'
    args.momentum = 0.9
    args.no_nesterov = False
    args.weight_decay = 3e-5
    args.label_smoothing = 0.1
    args.no_decay_keys = 'bn#bias#gamma#beta#w_quantizer#a_quantizer#scale#offset'
    args.fp16_allreduce = False

    args.model_init = 'he_fout'
    args.validation_frequency = 1
    args.print_frequency = 10

    args.n_worker = 8
    args.resize_scale = 0.08
    args.distort_color = 'tf'
    args.continuous_size = True
    args.not_sync_distributed_image_size = False

    args.bn_momentum = 0.1
    args.bn_eps = 1e-5
    args.dropout = 0.1
    args.base_stage_width = 'proxyless'

    args.width_mult_list = '1.2'
    args.dy_conv_scaling_mode = 1
    args.independent_distributed_sampling = False

    args.kd_ratio = 1.0
    args.kd_type = 'ce'

    # whether to apply Sandwich Rule
    args.sandwich = True  # False  #

    args.total_dynamic_batch_size = 6

    if args.task == 'quantize':
        args.path = os.path.join('.', 'exp/phase%d' % args.phase)
        if args.sandwich:
            args.dynamic_batch_size = args.total_dynamic_batch_size - 2
        else:
            args.dynamic_batch_size = args.total_dynamic_batch_size
        if args.phase == 0:
            args.model_path = None
            args.n_epochs = 60
            args.base_lr = 2.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.ks_list = '3,5,7'
            args.expand_list = '3,4,6'
            args.depth_list = '2,3,4'
            args.bits_list = '2,3,4,32'
        elif args.phase == 1:
            args.model_path = os.path.join('.', 'exp/phase0/checkpoint/%s/quantize.pth.tar' % str(hvd.rank()))
            args.n_epochs = 120
            args.base_lr = 2.5e-3
            args.warmup_epochs = 5
            args.warmup_lr = -1
            args.ks_list = '3,5,7'
            args.expand_list = '3,4,6'
            args.depth_list = '2,3,4'
            args.bits_list = '2,3,4'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    args.manual_seed = 0

    # # new configs:
    args.teacher_image_size = None  # if None, teacher will operate at the same size as student
    args.interpolation = 2
    if args.teacher == 'OFA':
        args.image_size = '128,160,192,224'
    else:
        raise NotImplementedError

    return args


def main(task, phase, teacher):
    args = get_args(task, phase, teacher)
    os.makedirs(args.path, exist_ok=True)
    args.teacher_path = 'ofa_mbv3_d234_e346_k357_w%s' % args.width_mult_list

    num_gpus = hvd.size()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    LoaderConfig.DYNAMIC_BATCH_SIZE = args.dynamic_batch_size
    LoaderConfig.CONTINUOUS = args.continuous_size
    LoaderConfig.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size
    LoaderConfig.SANDWICH = args.sandwich

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4
    run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

    # print run config information
    if hvd.rank() == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    # build net from args
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [int(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]
    args.bits_list = [int(b) for b in args.bits_list.split(',')]

    net = QFAMobileNetV3(
        n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout, base_stage_width=args.base_stage_width, width_mult_list=args.width_mult_list,
        ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list, bits_list=args.bits_list,
    )

    init = torch.load(args.teacher_path, map_location='cpu')['state_dict']
    net.load_state_dict(init)
    net.cuda()
    set_activation_statistics(net)

    if args.model_path is not None:
        init = torch.load(args.model_path, map_location='cpu')['state_dict']
        net.load_state_dict(init)
        net.cuda()
    
    # teacher model
    if args.kd_ratio > 0:
        if args.teacher == 'OFA':
            args.teacher_model = OFAMobileNetV3(
                n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=0, width_mult_list=args.width_mult_list[0], ks_list=[3, 5, 7], expand_ratio_list=6,
                depth_list=4,
            )
            args.teacher_model.set_active_subnet(args.width_mult_list[0], 7, 6, 4)
        else:
            raise NotImplementedError
        LoaderConfig.INTERPOLATION = args.interpolation
        LoaderConfig.TEACHER_IMAGE_SIZE = args.teacher_image_size
        args.teacher_model.cuda()

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    distributed_run_manager = DistributedRunManager(
        args.path, net, run_config, compression,
        backward_steps=args.total_dynamic_batch_size,
        is_root=(hvd.rank() == 0), init=False
    )
    distributed_run_manager.save_config()
    # hvd broadcast
    distributed_run_manager.broadcast()

    # load teacher net weights
    if args.kd_ratio > 0 and isinstance(args.teacher_model, OFAMobileNetV3):
        load_models(distributed_run_manager, args.teacher_model, model_path=args.teacher_path)

    # training
    from qfa.elastic_nn.training.progressive_shrinking import validate, train

    validate_func_dict = {
        'image_size_list': {224} if isinstance(args.image_size, int) else sorted(set(args.image_size)),
        'width_mult_list': sorted({0, len(args.width_mult_list) - 1}),
        'ks_list': sorted(args.ks_list),
        'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
        'depth_list': sorted({min(net.depth_list), max(net.depth_list)}),
        'bits_list': sorted(net.bits_list)}
    if args.task == 'quantize':
        from qfa.elastic_nn.training.progressive_shrinking import supporting_elastic_quantize
        supporting_elastic_quantize(train, distributed_run_manager, args, validate_func_dict)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    # OFA training
    main('quantize', 0, 'OFA')
    main('quantize', 1, 'OFA')
