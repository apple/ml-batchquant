#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn.functional as F
import copy
import time
import math

from .observer import *


WEIGHT_SHARING = True
USE_BQ = True


clamp = F.hardtanh  # softclamp


class FakeRoundOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None


# Learned Step Size Quantization: https://arxiv.org/abs/1902.08153
def gradscale(x, scale):
    yOut = x
    yGrad = x * scale
    y = (yOut - yGrad).detach() + yGrad
    return y


# Learned Step Size Quantization: https://arxiv.org/abs/1902.08153
def signpass(x):
    yOut = x.sign()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


# Learned Step Size Quantization: https://arxiv.org/abs/1902.08153
def roundpass(x):
    yOut = x.round()
    yGrad = x
    y = (yOut - yGrad).detach() + yGrad
    return y


class FakeQuantize(nn.Module):
    CALIBRATE = False
    CALIBRATION_CRITERION = 'kl'
    """
    Fake Quantizer that allows learnable scale and offset
    To use BatchQuant, set USE_BQ=True
    """

    def __init__(self, signed=True, bitwidth=8, use_bq=USE_BQ, affine=True, grad_scale=False):
        super(FakeQuantize, self).__init__()
        # the scale is initialized as 0 because it is passed through a softplus
        self.scale = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.offset = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.use_bq = use_bq
        self.affine = affine
        self.grad_scale = grad_scale
        self.signed = signed
        self.observer = BatchMinMaxObserver(
            signed=signed,
            bitwidth=bitwidth,
            qscheme=torch.per_tensor_affine)
        self.bitwidth = bitwidth
        self.calibrate_func = None

    def quantize(self, x):
        if self.signed:
            qmin, qmax = -2 ** (self.bitwidth - 1), 2 ** (self.bitwidth - 1) - 1
        else:
            qmin, qmax = 0, 2 ** self.bitwidth - 1

        if self.use_bq:
            observer_scale, zp = self.observer.calculate_qparams()
            observer_scale = observer_scale.to(self.scale.device)
            zp = zp.to(self.scale.device)
        else:
            observer_scale = 1.
            zp = 0.

        if self.CALIBRATE and self.affine and self.calibrate_func is not None:
            self.calibrate_func(x, observer_scale, zp, self.scale, self.offset, qmin, qmax)

        if self.affine:
            learned_scale = F.softplus(self.scale, beta=math.log(2.))
            learned_offset = self.offset
            if self.grad_scale:
                factor = 1. / math.sqrt(x.nelement() * qmax)
                learned_scale = gradscale(learned_scale, factor)
                learned_offset = gradscale(learned_offset, factor)
        else:
            learned_scale = 1.
            learned_offset = 0.

        # Scaled Quant with proper clipping, correct variant 2
        # Nicer gradient, better performance
        qx = x * learned_scale / observer_scale + zp + learned_offset
        qx = torch.clamp(qx, qmin, qmax)
        qx = FakeRoundOp.apply(qx)
        dqx = (qx - zp - learned_offset) * observer_scale / learned_scale

        return qx, dqx

    def forward(self, x):
        if self.training or isinstance(self.observer, BatchMinMaxObserver):
            if self.use_bq:
                x = self.observer(x)
        qx, dqx = self.quantize(x)
        return dqx


# Learned Step Size Quantization: https://arxiv.org/abs/1902.08153
def quantize(v, s, p, signed, per_channel=False):
    if signed:
        Qn = -2 ** (p - 1)
        Qp = 2 ** (p - 1) - 1
        if p == 1:
            Qp = 1
        nweight = v.nelement()
        if per_channel:
            nweight /= v.size(0)
        gradScaleFactor = 1 / math.sqrt(nweight * Qp)
    else:
        Qn = 0
        Qp = 2 ** p - 1
        nfeatures = v.nelement()
        gradScaleFactor = 1 / math.sqrt(nfeatures * Qp)

    s = gradscale(s, gradScaleFactor)
    v = v / s.abs()
    v = clamp(v, Qn, Qp)
    if signed and p == 1:
        vbar = signpass(v)
    else:
        vbar = roundpass(v)
    vhat = vbar * s.abs()
    return vhat


class LSQQuantize(nn.Module):

    def __init__(self, signed=True, bitwidth=8, ws=WEIGHT_SHARING, init='lsq'):
        super(LSQQuantize, self).__init__()
        self.scale = nn.Parameter(torch.Tensor(1))
        self.signed = signed
        self.bits = bitwidth
        self.ws = ws  # whether to use weight sharing
        self.init = init
        self.initialized = False
        self.weight = None

    def _quantize(self, x):
        if not self.initialized:  # calibration, make sure inputs is pretrained weight
            if not self.ws:
                self.weight = nn.Parameter(torch.zeros_like(x, requires_grad=True))
                self.weight.data = x.detach() if self.ws else copy.deepcopy(x.detach())
            if self.signed:
                q_max = 2 ** (self.bits - 1) - 1
                if self.bits == 1:
                    q_max = 1
            else:
                q_max = 2 ** (self.bits) - 1
            if self.init == 'lsq':
                self.scale.data = 2.0 / math.sqrt(q_max) * x.abs().mean().detach().view(1)
            else:
                assert self.init == 'lsq+'
                mu = x.mean().detach().view(1)
                std = x.std().detach().view(1)
                s_init = torch.max(torch.abs(mu - 3*std), torch.abs(mu + 3*std))
                self.scale.data = s_init / (q_max + 1.)
            self.initialized = True

        # Allow non Weight sharing
        if not self.ws:
            x = self.weight

        return quantize(x, self.scale, self.bits, self.signed)

    def forward(self, x):
        dqx = self._quantize(x)
        return dqx

    def re_organize_weights(self, sorted_idx):
        pass


def weight_quantizer(n_channels, ch_axis, bitwidth, signed):
    return LSQQuantize(signed=signed, bitwidth=bitwidth)


def activation_quantizer(n_channels, bitwidth, signed, per_channel=True):
    return FakeQuantize(signed=signed, bitwidth=bitwidth)
