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

from qfa.utils import SEModule
from qfa.layers import IdentityLayer
from qfa.q_layers import ZeroLayer, MBInvertedQConvLayer, QConvLayer, QLinearLayer
from qfa.imagenet_codebase.utils import MyModule, MyNetwork, int2list, get_net_device, build_activation
from qfa.elastic_nn.modules.dynamic_op import DynamicBatchNorm2d
from qfa.elastic_nn.modules.dynamic_op import DynamicSE
from qfa.elastic_nn.modules.dynamic_q_op import *
from qfa.elastic_nn.utils import adjust_bn_according_to_idx_q, copy_bn_q, copy_ch_quantizer

import time


class DynamicMBQConvLayer(MyModule):

    def __init__(self, in_channel_list, out_channel_list,
                 kernel_size_list=3, expand_ratio_list=6,
                 bits_list=32,
                 stride=1, dilation=1, act_func='relu6', use_se=False, signed=True):
        super(DynamicMBQConvLayer, self).__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = int2list(kernel_size_list, 1)
        self.expand_ratio_list = int2list(expand_ratio_list, 1)
        self.bits_list = int2list(bits_list, 1)

        self.stride = stride
        self.dilation = dilation
        self.act_func = act_func
        self.use_se = use_se
        self.signed = signed

        # build modules
        max_middle_channel = round(max(self.in_channel_list) * max(self.expand_ratio_list))
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', DynamicPointQConv2d(
                    max(self.in_channel_list), max_middle_channel,
                    self.bits_list, kernel_size=1, stride=1, dilation=1,
                    use_bn=True, eps=1e-05, momentum=0.1,
                    act_func=self.act_func, signed=signed))
            ]))

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicSeparableQConv2d(
                max_middle_channel, self.kernel_size_list, self.bits_list,
                stride=self.stride, dilation=self.dilation,
                use_bn=True, eps=1e-05, momentum=0.1,
                act_func=self.act_func,
                signed=True if self.inverted_bottleneck is None else False
            ))
        ]))

        if self.use_se:
            self.depth_conv.add_module('se', DynamicSE(max_middle_channel))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', DynamicPointQConv2d(
                max_middle_channel, max(self.out_channel_list),
                self.bits_list, kernel_size=1, stride=1, dilation=1,
                use_bn=True, eps=1e-05, momentum=0.1,
                act_func=None, signed=False))
        ]))

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = \
                make_divisible(round(in_channel * self.active_expand_ratio), 8)

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.use_se:
            return 'SE(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)
        else:
            return '(O%d, E%.1f, K%d)' % (self.active_out_channel, self.active_expand_ratio, self.active_kernel_size)

    @property
    def config(self):
        return {
            'name': DynamicMBQConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'kernel_size_list': self.kernel_size_list,
            'expand_ratio_list': self.expand_ratio_list,
            'bits_list': self.bits_list,
            'stride': self.stride,
            'act_func': self.act_func,
            'use_se': self.use_se,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicMBQConvLayer(**config)

    ############################################################################################

    def get_active_subnet(self, in_channel, preserve_weight=True):
        middle_channel = make_divisible(round(in_channel * self.active_expand_ratio), 8)

        # build the new layer
        active_bits = [
            self.inverted_bottleneck.conv.w_quantizer.active_bit if self.inverted_bottleneck else 32,
            self.inverted_bottleneck.conv.a_quantizer.active_bit if self.inverted_bottleneck else 32,
            self.depth_conv.conv.w_quantizers[str(self.active_kernel_size)].active_bit,
            self.depth_conv.conv.a_quantizers[str(self.active_kernel_size)].active_bit,
            self.point_linear.conv.w_quantizer.active_bit,
            self.point_linear.conv.a_quantizer.active_bit
        ]
        inverted_bottleneck_bits = active_bits[:2]
        depth_conv_bits = active_bits[2:4]
        point_linear_bits = active_bits[4:]
        sub_layer = MBInvertedQConvLayer(
            in_channel, self.active_out_channel, self.active_kernel_size, self.stride, self.active_expand_ratio,
            act_func=self.act_func, mid_channels=middle_channel, use_se=self.use_se, bits=active_bits,
            signed=self.signed
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        # copy weight from current layer
        if sub_layer.inverted_bottleneck is not None:
            sub_layer.inverted_bottleneck.conv.weight.data.copy_(
                self.inverted_bottleneck.conv.conv.weight.data[:middle_channel, :in_channel, :, :]
            )
            copy_bn_q(sub_layer.inverted_bottleneck.conv, self.inverted_bottleneck.conv)
            copy_ch_quantizer(sub_layer.inverted_bottleneck.conv.w_quantizer, self.inverted_bottleneck.conv.w_quantizer)
            sub_layer.inverted_bottleneck.conv.a_quantizer = copy.deepcopy(
                self.inverted_bottleneck.conv.a_quantizer.quantizers[str(inverted_bottleneck_bits[1])])

        sub_layer.depth_conv.conv.weight.data.copy_(
            self.depth_conv.conv.get_active_filter(self.active_kernel_size).data[:middle_channel, :middle_channel, :, :]
        )
        copy_bn_q(sub_layer.depth_conv.conv, self.depth_conv.conv)
        copy_ch_quantizer(sub_layer.depth_conv.conv.w_quantizer,
                          self.depth_conv.conv.w_quantizers[str(self.active_kernel_size)])
        sub_layer.depth_conv.conv.a_quantizer = copy.deepcopy(
            self.depth_conv.conv.a_quantizers[str(self.active_kernel_size)].quantizers[str(depth_conv_bits[1])])

        if self.use_se:
            se_mid = make_divisible(middle_channel // SEModule.REDUCTION, divisor=8)
            sub_layer.depth_conv.se.fc.reduce.weight.data.copy_(
                self.depth_conv.se.fc.reduce.weight.data[:se_mid, :middle_channel, :, :]
            )
            sub_layer.depth_conv.se.fc.reduce.bias.data.copy_(self.depth_conv.se.fc.reduce.bias.data[:se_mid])

            sub_layer.depth_conv.se.fc.expand.weight.data.copy_(
                self.depth_conv.se.fc.expand.weight.data[:middle_channel, :se_mid, :, :]
            )
            sub_layer.depth_conv.se.fc.expand.bias.data.copy_(self.depth_conv.se.fc.expand.bias.data[:middle_channel])

        sub_layer.point_linear.conv.weight.data.copy_(
            self.point_linear.conv.conv.weight.data[:self.active_out_channel, :middle_channel, :, :]
        )
        copy_bn_q(sub_layer.point_linear.conv, self.point_linear.conv)
        copy_ch_quantizer(sub_layer.point_linear.conv.w_quantizer, self.point_linear.conv.w_quantizer)
        sub_layer.point_linear.conv.a_quantizer = copy.deepcopy(
            self.point_linear.conv.a_quantizer.quantizers[str(point_linear_bits[1])])

        return sub_layer


class DynamicQConvLayer(MyModule):

    def __init__(self, in_channel_list, out_channel_list, bits_list, kernel_size=3, stride=1, dilation=1,
                 use_bn=True, eps=1e-05, momentum=0.1, act_func='relu6', signed=False, per_channel=True):
        super(DynamicQConvLayer, self).__init__()

        self.in_channel_list = int2list(in_channel_list, 1)
        self.out_channel_list = int2list(out_channel_list, 1)
        self.bits_list = int2list(bits_list, 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.eps = eps
        self.momentum = momentum
        self.act_func = act_func
        self.signed = signed
        self.per_channel = per_channel

        self.conv = DynamicPointQConv2d(
            max_in_channels=max(self.in_channel_list), max_out_channels=max(self.out_channel_list),
            bits_list=self.bits_list,
            kernel_size=self.kernel_size, stride=self.stride, dilation=self.dilation,
            use_bn=self.use_bn, eps=self.eps, momentum=self.momentum,
            act_func=self.act_func, signed=signed, per_channel=per_channel
        )

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        self.conv.active_out_channel = out_channel

        x = self.conv(x)
        return x

    @property
    def module_str(self):
        return 'DyConv(O%d, K%d, S%d)' % (self.active_out_channel, self.kernel_size, self.stride)

    @property
    def config(self):
        return {
            'name': DynamicQConvLayer.__name__,
            'in_channel_list': self.in_channel_list,
            'out_channel_list': self.out_channel_list,
            'bits_list': self.bits_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return DynamicConvLayer(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        pad = get_same_padding(self.kernel_size)
        active_w_bit = self.conv.w_quantizer.active_bit
        active_a_bit = self.conv.a_quantizer.active_bit
        sub_layer = QConvLayer(
            in_channel, self.active_out_channel, self.kernel_size,
            stride=self.stride, padding=pad, dilation=self.dilation,
            use_bn=self.use_bn, act_func=self.act_func,
            w_bits=active_w_bit, a_bits=active_a_bit, signed=self.signed
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.weight.data.copy_(self.conv.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        if self.use_bn:
            copy_bn_q(sub_layer, self.conv)

        copy_ch_quantizer(sub_layer.w_quantizer, self.conv.w_quantizer)
        sub_layer.a_quantizer = copy.deepcopy(self.conv.a_quantizer.quantizers[str(active_a_bit)])

        return sub_layer


class DynamicQLinearLayer(MyModule):

    def __init__(self, in_features_list, out_features, bits_list, bias=True, act_func=None, dropout_rate=0, signed=True):
        super(DynamicQLinearLayer, self).__init__()

        self.in_features_list = int2list(in_features_list, 1)
        self.out_features = out_features
        self.bits_list = int2list(bits_list, 1)
        self.bias = bias
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.signed = signed

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.linear = DynamicQLinear(
            max_in_features=max(self.in_features_list), max_out_features=self.out_features,
            bits_list=self.bits_list, bias=self.bias, act_func=self.act_func, signed=self.signed
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)

    @property
    def module_str(self):
        return 'DyLinear(%d)' % self.out_features

    @property
    def config(self):
        return {
            'name': DynamicQLinearLayer.__name__,
            'in_features_list': self.in_features_list,
            'out_features': self.out_features,
            'bits_list': self.bits_list,
            'bias': self.bias,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate
        }

    @staticmethod
    def build_from_config(config):
        return DynamicQLinearLayer(**config)

    def get_active_subnet(self, in_features, preserve_weight=True):
        active_w_bit = self.linear.w_quantizer.active_bit
        active_a_bit = self.linear.a_quantizer.active_bit
        sub_layer = QLinearLayer(in_features, self.out_features, bias=self.bias,
                                 act_func=self.act_func, dropout_rate=self.dropout_rate,
                                 w_bits=active_w_bit, a_bits=active_a_bit)
        sub_layer = sub_layer.to(get_net_device(self))
        if not preserve_weight:
            return sub_layer

        sub_layer.weight.data.copy_(self.linear.linear.weight.data[:self.out_features, :in_features])
        if self.bias:
            sub_layer.bias.data.copy_(self.linear.linear.bias.data[:self.out_features])
        sub_layer.w_quantizer = copy.deepcopy(self.linear.w_quantizer.quantizers[str(active_w_bit)])
        sub_layer.a_quantizer = copy.deepcopy(self.linear.a_quantizer.quantizers[str(active_a_bit)])
        return sub_layer
