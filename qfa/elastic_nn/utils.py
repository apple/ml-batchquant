#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

import copy
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import horovod.torch as hvd

from qfa.utils import AverageMeter
from qfa.imagenet_codebase.utils import get_net_device, DistributedTensor, BatchMinMaxObserver, FakeRoundOp, \
    FakeQuantize
from qfa.elastic_nn.modules.dynamic_op import DynamicBatchNorm2d
from qfa.elastic_nn.modules.dynamic_q_op import DynamicActivationQuantizer,  DynamicQLinear,\
    DynamicSeparableQConv2d, DynamicPointQConv2d


def inv_softplus(x, beta, threshold=20):
    if x * beta > threshold:
        return x
    else:
        return x + torch.log(1. - torch.exp(-beta*x))


def set_running_statistics(model, data_loader, distributed=False):
    quantized = False
    for n, m in model.named_modules():
        if 'quantizer' in n:
            quantized = True
            break
    if quantized:
        set_running_statistics_q(model, data_loader, distributed)
    else:
        set_running_statistics_full(model, data_loader, distributed)


def set_running_statistics_full(model, data_loader, distributed=False):
    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if distributed:
                bn_mean[name] = DistributedTensor(name + '#mean')
                bn_var[name] = DistributedTensor(name + '#var')
            else:
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x, batch_mean, batch_var, bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim], False,
                        0.0, bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    with torch.no_grad():
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
        for images, labels in data_loader:
            images = images.to(get_net_device(forward_model))
            forward_model(images)
        DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def set_running_statistics_q(model, data_loader, distributed=False):
    bn_mean = {}
    bn_var = {}
    aq_min = {}
    aq_max = {}

    forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if (type(m) in [DynamicSeparableQConv2d, DynamicPointQConv2d] and m.use_bn):
            if distributed:
                bn_mean[name] = DistributedTensor(name + '#mean')
                bn_var[name] = DistributedTensor(name + '#var')
            else:
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

            def collect_batch_stats(mean_est, var_est):
                def lambda_forward(batch_size, batch_mean, batch_var):
                    mean_est.update(batch_mean.data, batch_size)
                    var_est.update(batch_var.data, batch_size)

                return lambda_forward

            # TODO: Put this as an extra hook somewhere that only collects stats and does not impact results
            # (e.g. m.accum(orig_conv) called whenever m.accum is not None, right after orig_conv is computed)
            m.collect_batch_stats = collect_batch_stats(bn_mean[name], bn_var[name])

        if isinstance(m, BatchMinMaxObserver):
            if distributed:
                aq_min[name] = DistributedTensor(name + '#min')
                aq_max[name] = DistributedTensor(name + '#max')
            else:
                aq_min[name] = AverageMeter()
                aq_max[name] = AverageMeter()

            def collect_batch_stats(min_est, max_est):
                def lambda_forward(batch_size, batch_min, batch_max):
                    min_est.update(batch_min.data, batch_size)
                    max_est.update(batch_max.data, batch_size)

                return lambda_forward

            m.collect_batch_stats = collect_batch_stats(aq_min[name], aq_max[name])

    with torch.no_grad():
        DynamicSeparableQConv2d.SET_RUNNING_STATISTICS = True
        DynamicPointQConv2d.SET_RUNNING_STATISTICS = True
        BatchMinMaxObserver.SET_RUNNING_STATISTICS = True
        for images, labels in data_loader:
            images = images.to(get_net_device(forward_model))
            forward_model(images)
        DynamicSeparableQConv2d.SET_RUNNING_STATISTICS = False
        DynamicPointQConv2d.SET_RUNNING_STATISTICS = False
        BatchMinMaxObserver.SET_RUNNING_STATISTICS = False

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert (type(m) in [DynamicSeparableQConv2d, DynamicPointQConv2d] and m.use_bn)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
            m.collect_batch_stats = None
        if name in aq_min and aq_min[name].count > 0:
            assert isinstance(m, BatchMinMaxObserver)
            m.running_min.data.copy_(aq_min[name].avg)
            m.running_max.data.copy_(aq_max[name].avg)
            m.collect_batch_stats = None


def set_activation_statistics(dynamic_net, data_loader=None, distributed=False):
    with torch.no_grad():
        if data_loader is None:
            for name, m in dynamic_net.named_modules():
                if isinstance(m, DynamicPointQConv2d):
                    old_bit = m.w_quantizer.active_bit
                    for bit in m.w_quantizer.quantizers.keys():
                        m.w_quantizer.active_bit = int(bit)
                        m.w_quantizer(m.conv.weight)
                    m.w_quantizer.active_bit = old_bit

                if isinstance(m, DynamicSeparableQConv2d):
                    for ks in m.w_quantizers.keys():
                        m.active_kernel_size = int(ks)
                        old_bit = m.w_quantizers[ks].active_bit
                        for bit in m.w_quantizers[ks].quantizers.keys():
                            m.w_quantizers[ks].active_bit = int(bit)
                            filters = m.get_active_filter(int(ks))
                            m.w_quantizers[ks](filters)
                        m.w_quantizers[ks].active_bit = old_bit

                if isinstance(m, DynamicQLinear):
                    old_bit = m.w_quantizer.active_bit
                    for bit in m.w_quantizer.quantizers.keys():
                        m.w_quantizer.active_bit = int(bit)
                        m.w_quantizer(m.linear.weight)
                    m.w_quantizer.active_bit = old_bit
        else:
            for images, labels in data_loader:
                images = images.to(get_net_device(dynamic_net))
                for batch_idx in ['min', 'max']:
                    for ks in dynamic_net.ks_list:
                        for bit in dynamic_net.bits_list:
                            dynamic_net.set_sandwich_subnet(batch_idx)
                            dynamic_net.set_active_subnet(ks=ks, b=bit)
                            dynamic_net(images)
                for _ in range(6):
                    dynamic_net.sample_active_subnet()
                    dynamic_net(images)


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
    bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def copy_bn(target_bn, src_bn):
    feature_dim = target_bn.num_features

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
    target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


def copy_bn_q(target, src):
    feature_dim = target.num_features

    target.gamma.data.copy_(src.gamma.data[:feature_dim])
    target.beta.data.copy_(src.beta.data[:feature_dim])
    target.running_mean.data.copy_(src.running_mean.data[:feature_dim])
    target.running_var.data.copy_(src.running_var.data[:feature_dim])


def adjust_bn_according_to_idx_q(bn, idx):
    bn.gamma.data = torch.index_select(bn.gamma.data, 0, idx)
    bn.beta.data = torch.index_select(bn.beta.data, 0, idx)
    bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
    bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def copy_ch_quantizer(target, src):
    "src is a dynamic quantizer, target is a fixed quantizer"
    if not isinstance(target, nn.Identity):
        bits = target.bits
        target.scale.data.copy_(src.quantizers[str(bits)].scale.data)
