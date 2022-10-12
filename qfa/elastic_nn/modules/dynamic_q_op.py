#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

from collections import OrderedDict

import copy
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from qfa.utils import get_same_padding
from qfa.imagenet_codebase.utils import sub_filter_start_end, weight_quantizer, activation_quantizer, \
    make_divisible, build_activation

import time


class DynamicActivationQuantizer(nn.Module):
    """
    Dynamic Quantizer
    """
    def __init__(self, n_channels, signed, bits_list, per_channel=True):
        super(DynamicActivationQuantizer, self).__init__()
        self.quantizers = nn.ModuleDict([
            ['2', activation_quantizer(n_channels=n_channels, bitwidth=2, signed=signed, per_channel=per_channel)],
            ['3', activation_quantizer(n_channels=n_channels, bitwidth=3, signed=signed, per_channel=per_channel)],
            ['4', activation_quantizer(n_channels=n_channels, bitwidth=4, signed=signed, per_channel=per_channel)],
            ['8', activation_quantizer(n_channels=n_channels, bitwidth=8, signed=signed, per_channel=per_channel)],
            ['32', nn.Identity()]
        ])
        self.n_channels = n_channels
        self.signed = signed
        self.bits_list = bits_list
        self.per_channel = per_channel
        self.active_bit = max(self.bits_list)

    def set_bit(self, bit):
        assert bit in self.bits_list
        self.active_bit = bit

    def forward(self, x):
        return self.quantizers[str(self.active_bit)](x)

    def get_active_subnet(self, preserve_weight=True):
        if preserve_weight:
            return copy.deepcopy(self.quantizers[str(self.active_bit)])
        else:
            return activation_quantizer(n_channels=self.n_channels, bitwidth=self.active_bit, signed=self.signed, per_channel=self.per_channel)


class DynamicWeightQuantizer(nn.Module):
    SHARE_FP = True

    """
    Dynamic Quantizer
    """
    def __init__(self, n_channels, ch_axis, signed, bits_list):
        super(DynamicWeightQuantizer, self).__init__()
        self.quantizers = nn.ModuleDict([
            ['2', weight_quantizer(n_channels=n_channels, ch_axis=ch_axis, bitwidth=2, signed=signed)],
            ['3', weight_quantizer(n_channels=n_channels, ch_axis=ch_axis, bitwidth=3, signed=signed)],
            ['4', weight_quantizer(n_channels=n_channels, ch_axis=ch_axis, bitwidth=4, signed=signed)],
            ['8', weight_quantizer(n_channels=n_channels, ch_axis=ch_axis, bitwidth=8, signed=signed)],
            ['32', nn.Identity()]
        ])
        self.n_channels = n_channels
        self.ch_axis = ch_axis
        self.signed = signed
        self.bits_list = bits_list
        self.active_bit = max(self.bits_list)

        self.initialized = False
        self.weight = None

    def set_bit(self, bit):
        assert bit in self.bits_list
        self.active_bit = bit

    def forward(self, x):
        if self.SHARE_FP:
            return self.quantizers[str(self.active_bit)](x)
        else:
            if not self.initialized:  # calibration, make sure inputs is pretrained weight
                self.weight = nn.Parameter(torch.zeros_like(x, requires_grad=True))
                self.weight.data = copy.deepcopy(x.detach())
                self.initialized = True
            if str(self.active_bit) == '32':
                return self.quantizers[str(self.active_bit)](self.weight)
            else:
                return self.quantizers[str(self.active_bit)](x)

    def re_organize_weights(self, sorted_idx):
        for k, q in self.quantizers.items():
            if int(k) < 32:
                q.re_organize_weights(sorted_idx)

    def get_active_subnet(self, preserve_weight=True):
        if preserve_weight:
            return copy.deepcopy(self.quantizers[str(self.active_bit)])
        else:
            return weight_quantizer(n_channels=self.n_channels, ch_axis=self.ch_axis, bitwidth=self.active_bit, signed=self.signed)


class DynamicSeparableQConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = 1
    SET_RUNNING_STATISTICS = False
    FREEZE_BN = False
    BITS_LIST = [2, 3, 4, 32]

    def __init__(self, max_in_channels, kernel_size_list, bits_list,
                 stride=1, dilation=1,
                 use_bn=True,
                 eps=1e-05, momentum=0.1,
                 act_func=None,
                 signed=False,
                 per_channel=True,
                 ):
        super(DynamicSeparableQConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.bits_list = bits_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.max_in_channels, bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        assert self.KERNEL_TRANSFORM_MODE is not None
        # register scaling parameters
        # 7to5_matrix, 5to3_matrix
        scale_params = {}
        for i in range(len(self._ks_set) - 1):
            ks_small = self._ks_set[i]
            ks_larger = self._ks_set[i + 1]
            param_name = '%dto%d' % (ks_larger, ks_small)
            scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
        for name, param in scale_params.items():
            self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

        self.use_bn = use_bn
        self.eps = eps
        self.momentum = momentum

        if use_bn:
            self.num_features = self.max_in_channels
            self.gamma = nn.Parameter(torch.Tensor(self.num_features))
            self.beta = nn.Parameter(torch.Tensor(self.num_features))
            self.affine = True
            self.track_running_stats = True
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            self.reset_bn_parameters()

        self.w_quantizers = nn.ModuleDict([
            [str(ks), DynamicWeightQuantizer(n_channels=max_in_channels, ch_axis=0, signed=True, bits_list=self.bits_list)]
            for ks in self.kernel_size_list
        ])

        self.signed = signed
        self.per_channel = per_channel
        self.a_quantizers = nn.ModuleDict([
            [str(ks), DynamicActivationQuantizer(n_channels=max_in_channels, signed=signed, bits_list=self.bits_list, per_channel=per_channel)]
            for ks in self.kernel_size_list
        ])
        self.act = build_activation(act_func)

        # Placeholder for batch stats collection function
        self.collect_batch_stats = None

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_bn_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    # Unlike original implimentation, we don't grab the active channels now, we grab it after quantization
    def get_active_filter(self, kernel_size):
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:, :, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:, :, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter

            filters = start_filter

        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        out_channel = in_channel = x.size(1)

        x = self.a_quantizers[str(kernel_size)](x)
        # x = self.a_quantizers(x)

        # Always return the max channel now, as we grab the actual active channel after bn scaling and quantization
        filters = self.get_active_filter(kernel_size).contiguous()
        padding = get_same_padding(kernel_size, self.dilation)

        if self.use_bn:
            quantized_weight = self.w_quantizers[str(kernel_size)](filters)
            active_quantized_weight = quantized_weight[:out_channel, :in_channel, :, :].contiguous()

            conv = F.conv2d(
                x, active_quantized_weight, None, self.stride, padding, self.dilation, in_channel
            )

            if self.training or self.SET_RUNNING_STATISTICS:
                batch_mean = torch.mean(conv, dim=[0, 2, 3])
                batch_var = torch.var(conv, dim=[0, 2, 3], unbiased=False)
                if self.SET_RUNNING_STATISTICS:
                    assert self.collect_batch_stats is not None
                    self.collect_batch_stats(conv.size(0), batch_mean, batch_var)
                else:
                    assert self.collect_batch_stats is None
                conv = (conv - batch_mean[None, :, None, None]) / (torch.sqrt(batch_var[None, :, None, None] + self.eps))
                conv = conv * self.gamma[None, :out_channel, None, None] + self.beta[None, :out_channel, None, None]
            else:
                conv = F.batch_norm(
                    conv, self.running_mean[:out_channel], self.running_var[:out_channel],
                    self.gamma[:out_channel], self.beta[:out_channel], False, 0.0, self.eps)
        else:
            quantized_weight = self.w_quantizers[str(kernel_size)](filters)
            active_quantized_weight = quantized_weight[:out_channel, :in_channel, :, :].contiguous()
            conv = F.conv2d(
                x, active_quantized_weight, None, self.stride, padding, self.dilation,
                in_channel
            )

        return self.act(conv)


class DynamicPointQConv2d(nn.Module):
    SET_RUNNING_STATISTICS = False
    FREEZE_BN = False

    def __init__(self, max_in_channels, max_out_channels,
                 bits_list, kernel_size=1, stride=1, dilation=1,
                 use_bn=True, eps=1e-05, momentum=0.1,
                 act_func=None, signed=False,
                 per_channel=True):
        super(DynamicPointQConv2d, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )

        self.active_out_channel = self.max_out_channels

        self.bits_list = bits_list

        self.use_bn = use_bn
        self.eps = eps
        self.momentum = momentum

        if use_bn:
            self.num_features = self.max_out_channels
            self.gamma = nn.Parameter(torch.Tensor(self.num_features))
            self.beta = nn.Parameter(torch.Tensor(self.num_features))
            self.affine = True
            self.track_running_stats = True
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            self.reset_bn_parameters()

        self.w_quantizer = DynamicWeightQuantizer(n_channels=self.max_out_channels, ch_axis=0, signed=True, bits_list=self.bits_list)

        self.signed = signed
        self.per_channel = per_channel
        self.a_quantizer = DynamicActivationQuantizer(n_channels=self.max_in_channels, signed=signed, bits_list=self.bits_list, per_channel=self.per_channel)
        self.act = build_activation(act_func)

        # Placeholder for batch stats collection function
        self.collect_batch_stats = None

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_bn_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    def forward(self, x, out_channel=None):
        x = self.a_quantizer(x)

        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.conv.weight
        padding = get_same_padding(self.kernel_size)

        if self.use_bn:
            quantized_weight = self.w_quantizer(filters)
            active_quantized_weight = quantized_weight[:out_channel, :in_channel, :, :].contiguous()

            conv = F.conv2d(
                x, active_quantized_weight, None, self.stride, padding, self.dilation, 1
            )

            if self.training or self.SET_RUNNING_STATISTICS:
                batch_mean = torch.mean(conv, dim=[0, 2, 3])
                batch_var = torch.var(conv, dim=[0, 2, 3], unbiased=False)
                if self.SET_RUNNING_STATISTICS:
                    assert self.collect_batch_stats is not None
                    self.collect_batch_stats(conv.size(0), batch_mean, batch_var)
                else:
                    assert self.collect_batch_stats is None
                conv = (conv - batch_mean[None, :, None, None]) / (
                    torch.sqrt(batch_var[None, :, None, None] + self.eps))
                conv = conv * self.gamma[None, :out_channel, None, None] + self.beta[None, :out_channel, None, None]
            else:
                conv = F.batch_norm(
                    conv, self.running_mean[:out_channel], self.running_var[:out_channel],
                    self.gamma[:out_channel], self.beta[:out_channel], False, 0.0, self.eps)
        else:
            quantized_weight = self.w_quantizer(filters)
            active_quantized_weight = quantized_weight[:out_channel, :in_channel, :, :].contiguous()
            conv = F.conv2d(
                x, active_quantized_weight, None, self.stride, padding, self.dilation, 1
            )

        return self.act(conv)


class DynamicQLinear(nn.Module):

    def __init__(self, max_in_features, max_out_features, bits_list, bias=True, act_func=None, signed=True):
        super(DynamicQLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

        self.bits_list = bits_list

        self.w_quantizer = DynamicWeightQuantizer(n_channels=None, ch_axis=None, signed=True, bits_list=self.bits_list)

        self.signed = signed
        self.a_quantizer = DynamicActivationQuantizer(n_channels=max_in_features, signed=signed, bits_list=self.bits_list, per_channel=False)
        self.act = build_activation(act_func)

    def forward(self, x, out_features=None):
        x = self.a_quantizer(x)
        if out_features is None:
            out_features = self.active_out_features
        in_features = x.size(1)
        quantized_weight = self.w_quantizer(self.linear.weight)
        active_quantized_weight = quantized_weight[:out_features, :in_features].contiguous()
        bias = self.linear.bias[:out_features] if self.bias else None
        y = F.linear(x, active_quantized_weight, bias)
        return self.act(y)
