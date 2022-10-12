#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from PyTorch (https://github.com/pytorch/pytorch)
#

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from qfa.utils import get_same_padding
from qfa.imagenet_codebase.utils import MyModule, MyNetwork, make_divisible,\
    build_activation, activation_quantizer, weight_quantizer


class ZeroLayer(MyModule):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        raise ValueError

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {
            'name': ZeroLayer.__name__,
            'stride': self.stride,
        }

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, act_func=None, w_bits=32, a_bits=32, signed=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.w_quantizer = weight_quantizer(n_channels=None, ch_axis=None, bitwidth=w_bits, signed=True) if w_bits < 32 else nn.Identity()
        signed = signed
        if a_bits < 32 and act_func is not None:
            self.a_quantizer = activation_quantizer(n_channels=in_features, bitwidth=a_bits, signed=signed, per_channel=False)
        else:
            self.a_quantizer = nn.Identity()
        self.act = build_activation(act_func)

    def quantized_linear(self, x, weight):
        x = F.linear(self.a_quantizer(x), self.w_quantizer(weight), self.bias)
        return self.act(x)

    def forward(self, x):
        return self.quantized_linear(x, self.weight)

    @classmethod
    def from_float(cls, mod, qconfig=None):
        pass


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, act_func=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, w_bits=32, a_bits=32, signed=False):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        if w_bits < 32:
            self.w_quantizer = weight_quantizer(n_channels=out_channels, ch_axis=0, bitwidth=w_bits, signed=True)
        else:
            self.w_quantizer = nn.Identity()
        signed = signed
        if a_bits < 32 and act_func is not None:
            self.a_quantizer = activation_quantizer(n_channels=in_channels, bitwidth=a_bits, signed=signed)
        else:
            self.a_quantizer = nn.Identity()
        self.act = build_activation(act_func)

    def quantized_conv2d_forward(self, x, weight):
        x = self._conv_forward(self.a_quantizer(x), self.w_quantizer(weight))
        return self.act(x)

    def forward(self, x):
        return self.quantized_conv2d_forward(x, self.weight)

    @classmethod
    def from_float(cls, mod, qconfig=None):
        pass


class _QConvBnNd(nn.modules.conv._ConvNd):
    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups,
                 padding_mode,
                 use_bn=True,
                 eps=1e-05, momentum=0.1,
                 act_func=None,
                 dropout_rate=0,
                 w_bits=32, a_bits=32, signed=False):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, False, padding_mode)
        self.eps = eps
        self.momentum = momentum
        self.use_bn = use_bn
        if use_bn:
            self.num_features = out_channels
            self.gamma = nn.Parameter(torch.Tensor(out_channels))
            self.beta = nn.Parameter(torch.Tensor(out_channels))
            self.affine = True
            self.track_running_stats = True
            self.register_buffer('running_mean', torch.zeros(out_channels))
            self.register_buffer('running_var', torch.ones(out_channels))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            self.dropout = None
        self.w_bits = w_bits
        self.a_bits = a_bits
        if w_bits < 32:
            self.w_quantizer = weight_quantizer(n_channels=out_channels, ch_axis=0, bitwidth=w_bits, signed=True)
        else:
            self.w_quantizer = nn.Identity()
        signed = signed
        if a_bits < 32 and act_func is not None:
            self.a_quantizer = activation_quantizer(n_channels=in_channels, bitwidth=a_bits, signed=signed)
        else:
            self.a_quantizer = nn.Identity()
        self.act = build_activation(act_func)
        if use_bn:
            self.reset_bn_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_bn_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    def reset_parameters(self):
        super(_QConvBnNd, self).reset_parameters()
        # A hack to avoid resetting on undefined parameters
        if hasattr(self, 'gamma'):
            self.reset_bn_parameters()

    def _forward(self, input):
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.use_bn:
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            conv = self._conv_forward(input, self.w_quantizer(self.weight))

            conv = F.batch_norm(
                conv,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.gamma, self.beta, self.training, exponential_average_factor, self.eps)
        else:
            conv = self._conv_forward(input, self.w_quantizer(self.weight))
        return conv

    def extra_repr(self):
        return super(_ConvBnNd, self).extra_repr()

    def forward(self, input):
        # dropout before weight operation
        if self.dropout:
            input = self.dropout(input)
        input = self.a_quantizer(input)
        return self.act(self._forward(input))

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'transposed': self.transposed,
            'output_padding': self.output_padding,
            'groups': self.groups,
            'padding_mode': self.padding_mode,
            'use_bn': self.use_bn,
            'eps': self.eps,
            'momentum': self.momentum,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'w_bits': self.w_bits,
            'a_bits': self.a_bits,
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    @classmethod
    def from_float(cls, mod, qconfig=None):
        pass


class QConvLayer(_QConvBnNd, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.
    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.
    Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2
    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.
    Attributes:
        freeze_bn:
        activation_post_process: fake quant module for output activation
        weight_fake_quant: fake quant module for weight
    """

    # _FLOAT_MODULE = torch.nn.intrinsic.ConvBn2d

    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 padding_mode='zeros',
                 use_bn=True,
                 eps=1e-05, momentum=0.1,
                 act_func=None,
                 dropout_rate=0,
                 w_bits=32, a_bits=32, signed=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _QConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                            padding, dilation, False, _pair(0), groups, padding_mode,
                            use_bn, eps, momentum, act_func, dropout_rate, w_bits, a_bits, signed)

    @property
    def module_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        if self.groups == 1:
            if self.dilation > 1:
                conv_str = '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
            else:
                conv_str = '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
        else:
            if self.dilation > 1:
                conv_str = '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
            else:
                conv_str = '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])
        conv_str += '_O%d' % self.out_channels
        return conv_str

    @property
    def config(self):
        return {
            'name': QConvLayer.__name__,
            **super(QConvLayer, self).config,
        }

    @staticmethod
    def build_from_config(config):
        return QConvLayer(**config)


class QLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, act_func=None, dropout_rate=0, w_bits=32, a_bits=32, signed=False):
        super(QLinearLayer, self).__init__(in_features, out_features, bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.w_quantizer = weight_quantizer(n_channels=None, ch_axis=None, bitwidth=w_bits, signed=True) if w_bits < 32 else nn.Identity()
        if a_bits < 32 and act_func is not None:
            self.a_quantizer = activation_quantizer(n_channels=in_features, bitwidth=a_bits, signed=signed, per_channel=False)
        else:
            self.a_quantizer = nn.Identity()
        self.act = build_activation(act_func)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate, inplace=True)
        else:
            self.dropout = None

    def quantized_linear(self, x, weight):
        if self.dropout:
            x = self.dropout(x)
        x = F.linear(self.a_quantizer(x), self.w_quantizer(weight), self.bias)
        return self.act(x)

    def forward(self, x):
        return self.quantized_linear(x, self.weight)

    @classmethod
    def from_float(cls, mod, qconfig=None):
        pass

    @property
    def module_str(self):
        return '%dx%d_Linear' % (self.in_features, self.out_features)

    @property
    def config(self):
        return {
            'name': LinearLayer.__name__,
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.bias,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'w_bits': self.w_bits,
            'a_bits': self.a_bits,
        }

    @staticmethod
    def build_from_config(config):
        return QLinearLayer(**config)


class MBInvertedQConvLayer(MyModule):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6,
                 mid_channels=None, act_func='relu6', use_se=False,
                 bits=[
                     32, 32,  # inverted_bottleneck_bits
                     32, 32,  # depth_conv_bits
                     32, 32,  # point_linear_bits
                 ],
                 signed=True,
                 ):
        super(MBInvertedQConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels
        self.act_func = act_func
        self.use_se = use_se
        self.signed = signed

        self.bits = bits
        self.inverted_bottleneck_bits = bits[:2]
        self.depth_conv_bits = bits[2:4]
        self.point_linear_bits = bits[4:]

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', QConvLayer(self.in_channels, feature_dim, 1, stride=1,
                                    padding=0, use_bn=True, act_func=self.act_func,
                                    w_bits=self.inverted_bottleneck_bits[0],
                                    a_bits=self.inverted_bottleneck_bits[1],
                                    signed=signed))
            ]))

        pad = get_same_padding(self.kernel_size)
        depth_conv_modules = [
            ('conv', QConvLayer(feature_dim, feature_dim, kernel_size, stride=stride,
                                padding=pad, groups=feature_dim, use_bn=True, act_func=self.act_func,
                                w_bits=self.depth_conv_bits[0],
                                a_bits=self.depth_conv_bits[1],
                                signed = False if self.inverted_bottleneck is not None else True))
        ]

        if self.use_se:
            from qfa.layers import SEModule
            depth_conv_modules.append(('se', SEModule(feature_dim)))
        self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', QConvLayer(feature_dim, out_channels, 1, stride=1,
                                padding=0, use_bn=True, act_func=None,
                                w_bits=self.point_linear_bits[0],
                                a_bits=self.point_linear_bits[1],
                                signed=False))
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @property
    def module_str(self):
        if self.mid_channels is None:
            expand_ratio = self.expand_ratio
        else:
            expand_ratio = self.mid_channels // self.in_channels
        layer_str = '%dx%d_MBConv%d_%s' % (self.kernel_size, self.kernel_size, expand_ratio, self.act_func.upper())
        if self.use_se:
            layer_str = 'SE_' + layer_str
        layer_str += '_O%d' % self.out_channels
        return layer_str

    @property
    def config(self):
        return {
            'name': MBInvertedQConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
            'act_func': self.act_func,
            'use_se': self.use_se,
            'bits': self.bits
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedQConvLayer(**config)
