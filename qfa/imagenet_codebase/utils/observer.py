#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from PyTorch (https://github.com/pytorch/pytorch)
#

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial

import torch
import torch.nn as nn
from torch._jit_internal import List, Optional

import numpy as np
import time


def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.
    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.
    .. Example::
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


ABC = ABCMeta(str("ABC"), (object,), {})  # compatible with Python 2 *and* 3:


class ObserverBase(ABC, nn.Module):
    r"""Base observer Module.
    Any observer implementation should derive from this class.
    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.
    Args:
        dtype: Quantized data type
    """
    def __init__(self, signed, bitwidth):
        super(ObserverBase, self).__init__()
        self.signed = signed
        self.bitwidth = bitwidth
        self.dtype = torch.qint8 if signed else torch.quint8

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    # Returns all quantization parameters that's needed
    # for a quantize function call
    # For instance, per channel obsserver will return
    # scales, zero_points and axis
    @abstractmethod
    def get_qparams(self, **kwargs):
        pass

    with_args = classmethod(_with_args)


class _ObserverBase(ObserverBase):
    r"""Internal common base for all qint/quint8 observers.
    This base is for commonly used paramters used internally.
    Users should use `~torch.quantization.observer.ObserverBase` as a base class
    for custom observers.
    Args:
        dtype: Quantized data type.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
                      This is sometimes required to avoid instruction overflow.
    .. warning::
        :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.
    .. warning::
        :attr:`qscheme` can only take one of the following options:
        - ``torch.per_tensor_affine``
        - ``torch.per_tensor_symmetric``
        - ``torch.per_channel_affine``
        - ``torch.per_channel_symmetric``
    """

    def __init__(self, signed=False, bitwidth=8, qscheme=torch.per_tensor_affine):
        super(_ObserverBase, self).__init__(signed=signed, bitwidth=bitwidth)
        self.qscheme = qscheme

        self.eps = torch.finfo(torch.float32).eps
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
            torch.per_channel_affine,
            torch.per_channel_symmetric,
        ), "Default Observer only works for per_tensor_affine, \
                per_tensor_symmetric, per_channel_affine and \
                per_channel_symmetric quantization scheme"
        
        if signed:
            assert bitwidth >= 1, "Must have at least 1 bit for sign"
        else:
            assert bitwidth >= 0, "Bitwidth must be non-negative"

    def _calculate_per_channel_qparams(self, min_vals, max_vals):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        r"""Calculates the per channel quantization parameters, given min and max
        value tensors.
        Args:
            min_vals: Minimum values per channel
            max_vals: Maximum values per channel
        Returns:
            scales: Per channel scales tensor of shape (#channels,)
            zero_points: Per channel zero points tensor of shape (#channels,)
        """
        if min_vals.numel() == 0 or max_vals.numel() == 0:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])

        diff = min_vals <= max_vals
        assert (torch.sum(diff) == len(diff)), "min_vals should be less than max_vals for indices."

        scales = torch.empty(min_vals.size(), dtype=torch.float32)
        zero_points = torch.empty(min_vals.size(), dtype=torch.int64)

        if self.signed:
            qmin, qmax = -2**(self.bitwidth - 1), 2**(self.bitwidth - 1) - 1
        else:
            qmin, qmax = 0, 2**self.bitwidth - 1

        max_vals, min_vals = max_vals.to(dtype=torch.float), min_vals.to(dtype=torch.float)

        min_vals = torch.min(min_vals, torch.tensor([0.], device=min_vals.device, dtype=torch.float))
        max_vals = torch.max(max_vals, torch.tensor([0.], device=max_vals.device, dtype=torch.float))
        if torch.equal(max_vals, min_vals):
            scales.fill_(1.0)
            zero_points.fill_(0)
        else:
            if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
                max_vals = torch.max(-min_vals, max_vals)
                scales = max_vals / ((qmax - qmin) / 2)
                scales = torch.max(scales, torch.tensor([self.eps], device=scales.device, dtype=scales.dtype))
                if self.signed:
                    zp = 0
                else:
                    zp = 2**(self.bitwidth - 1)
                zero_points.fill_(zp)
            else:
                scales = (max_vals - min_vals) / float(qmax - qmin)
                scales = torch.max(scales, torch.tensor([self.eps], device=scales.device))
                zero_points = qmin - torch.round(min_vals / scales)
                zero_points = torch.max(zero_points, torch.tensor([qmin], dtype=zero_points.dtype, device=zero_points.device))
                zero_points = torch.min(zero_points, torch.tensor([qmax], dtype=zero_points.dtype, device=zero_points.device))
                zero_points = zero_points.to(dtype=torch.int64)
        scales.to(dtype=torch.float)

        return scales, zero_points

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        r"""Calculates the per tensor quantization parameters, given the min/max.
        Args:
            min_val: Per tensor minimum value
            max_val: Per tensor maximum value
        Returns:
            scale: Scale as a tensor of shape (1,)
            zero_point: Zero point as a tensor of shape (1,)
        """

        if max_val.numel() == 0 or min_val.numel() == 0:
            warnings.warn("Must run observer before calling calculate_qparams.\
                           Returning default scale and zero point.")
            return torch.tensor([1.0]), torch.tensor([0])

        assert min_val <= max_val, "min {} should be less than max {}".format(
            min_val, max_val
        )

        if self.signed:
            qmin, qmax = -2**(self.bitwidth - 1), 2**(self.bitwidth - 1) - 1
        else:
            qmin, qmax = 0, 2**self.bitwidth - 1

        max_val, min_val = float(max_val), float(min_val)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
                max_val = max(-min_val, max_val)
                scale = max_val / ((qmax - qmin) / 2)
                scale = max(scale, self.eps)
                zero_point = 0 if self.signed else 2**(self.bitwidth - 1)
            else:
                scale = (max_val - min_val) / float(qmax - qmin)
                scale = max(scale, self.eps)
                zero_point = qmin - round(min_val / scale)
                zero_point = max(qmin, zero_point)
                zero_point = min(qmax, zero_point)
                zero_point = int(zero_point)

        return torch.tensor([scale]), torch.tensor([zero_point])

    @torch.jit.export
    def get_qparams(self):
        r"""Get all quantization parameters needed for quantize call"""
        return self.calculate_qparams()

class MinMaxObserver(_ObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running min and max values.
    This observer uses the tensor min/max statistics to compute the quantization
    parameters. The module records the running minimum and maximum of incoming
    tensors, and uses this statistic to compute the quantization parameters.
    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
    Given running min/max as :math:`x_\text{min}` and :math:`x_\text{max}`,
    scale :math:`s` and zero point :math:`z` are computed as:
    The running minimum/maximum :math:`x_\text{min/max}` is computed as:
    .. math::
        \begin{array}{ll}
        x_\text{min} &= \begin{cases}
            \min(X) & \text{if~}x_\text{min} = \text{None} \\
            \min\left(x_\text{min}, \min(X)\right) & \text{otherwise}
        \end{cases}\\
        x_\text{max} &= \begin{cases}
            \max(X) & \text{if~}x_\text{max} = \text{None} \\
            \max\left(x_\text{max}, \max(X)\right) & \text{otherwise}
        \end{cases}\\
        \end{array}
    where :math:`X` is the observed tensor.
    The scale :math:`s` and zero point :math:`z` are then computed as:
    .. math::
        \begin{aligned}
            \text{if Symmetric:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{if dtype is qint8} \\
                128 & \text{otherwise}
            \end{cases}\\
            \text{Otherwise:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}
    where :math:`Q_\text{min}` and :math:`Q_\text{max}` are the minimum and
    maximum of the quantized data type.
    .. warning:: Only works with ``torch.per_tensor_symmetric`` quantization scheme
    .. warning:: :attr:`dtype` can only take ``torch.qint8`` or ``torch.quint8``.
    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """

    def __init__(self, signed=False, bitwidth=8, qscheme=torch.per_tensor_affine):
        # For x86 quantized kernels, we need to ensure that the vpmaddubsw
        # instruction does not overflow. We allow for a reduce_range argument to
        # observers that reduces the quantized range to (0,127) or (-64, 63).
        # For more details see aten/src/ATen/native/quantized/cpu/qconv.cpp
        # This is not an optimal choice for non x86 backends as it loses a bit
        # of precision for activations.

        super(MinMaxObserver, self).__init__(signed=signed,
                                             bitwidth=bitwidth,
                                             qscheme=qscheme)
        self.register_buffer('min_val', torch.tensor([]))
        self.register_buffer('max_val', torch.tensor([]))

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            min_val = torch.min(torch.min(x), min_val)
            max_val = torch.max(torch.max(x), max_val)
        self.min_val = min_val
        self.max_val = max_val
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        r"""Calculates the quantization parameters."""
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(MinMaxObserver, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'min_val'] = self.min_val
        destination[prefix + 'max_val'] = self.max_val

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        local_state = ['min_val', 'max_val']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(MinMaxObserver, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                          missing_keys, unexpected_keys, error_msgs)


class MovingAverageMinMaxObserver(MinMaxObserver):
    r"""Observer module for computing the quantization parameters based on the
    moving average of the min and max values.
    This observer computes the quantization parameters based on the moving
    averages of minimums and maximums of the incoming tensors. The module
    records the average minimum and maximum of incoming tensors, and uses this
    statistic to compute the quantization parameters.
    Args:
        averaging_constant: Averaging constant for min/max.
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
    The moving average min/max is computed as follows
    .. math::
        \begin{array}{ll}
                x_\text{min} = \begin{cases}
                    \min(X) & \text{if~}x_\text{min} = \text{None} \\
                    (1 - c) x_\text{min} + c \min(X) & \text{otherwise}
                \end{cases}\\
                x_\text{max} = \begin{cases}
                    \max(X) & \text{if~}x_\text{max} = \text{None} \\
                    (1 - c) x_\text{max} + c \max(X) & \text{otherwise}
                \end{cases}\\
        \end{array}
    where :math:`x_\text{min/max}` is the running average min/max, :math:`X` is
    is the incoming tensor, and :math:`c` is the ``averaging_constant``.
    The scale and zero point are then computed as in
    :class:`~torch.quantization.observer.MinMaxObserver`.
    .. note:: Only works with ``torch.per_tensor_affine`` quantization shceme.
    .. note:: If the running minimum equals to the running maximum, the scale
              and zero_point are set to 1.0 and 0.
    """
    def __init__(self, averaging_constant=0.01, signed=False, bitwidth=8,
                 qscheme=torch.per_tensor_affine):
        self.averaging_constant = averaging_constant
        super(MovingAverageMinMaxObserver, self).__init__(signed=signed,
                                                          bitwidth=bitwidth,
                                                          qscheme=qscheme)

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val
        max_val = self.max_val
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            min_val = min_val + self.averaging_constant * (torch.min(x) - min_val)
            max_val = max_val + self.averaging_constant * (torch.max(x) - max_val)
        self.min_val = min_val
        self.max_val = max_val
        return x_orig


class BatchMinMaxObserver(MinMaxObserver):
    SET_RUNNING_STATISTICS = False
    TEST_MODE = 'running'  # 'batch'  #

    """Use this for activation only"""
    def __init__(self, signed=False, bitwidth=8,
                 qscheme=torch.per_tensor_affine):
        super(BatchMinMaxObserver, self).__init__(signed=signed,
                                                  bitwidth=bitwidth,
                                                  qscheme=qscheme)
        self.criteria = 'mean'
        self.track_running_stats = True
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Placeholder for batch stats collection function
        self.collect_batch_stats = None

    def get_minmax(self, x):
        if self.criteria == 'mean':  # avg along channel dim
            if x.dim() == 4:
                y = x.permute((1, 0, 2, 3))
            else:
                y = x.permute((1, 0))
            y = torch.flatten(y, start_dim=1)
            batch_min = torch.min(y, 1)[0].mean()
            batch_max = torch.max(y, 1)[0].mean()
        elif self.criteria == 'bmean':  # avg along batch dim
            y = torch.flatten(x, start_dim=1)
            batch_min = torch.min(y, 1)[0].mean()
            batch_max = torch.max(y, 1)[0].mean()
        elif self.criteria == 'minmax':
            batch_min = torch.min(x)
            batch_max = torch.max(x)
        else:
            raise NotImplementedError
        return batch_min.to(x.device), batch_max.to(x.device)


    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        batch_min, batch_max = self.get_minmax(x)

        if self.training or self.TEST_MODE == 'batch' or self.SET_RUNNING_STATISTICS:  # and not self.freeze_bn
            # recovering original conv to get original batch_mean and batch_var
            min_val = batch_min
            max_val = batch_max
            if self.SET_RUNNING_STATISTICS:
                assert self.collect_batch_stats is not None
                # Important!!!! We may want to use unbiased estimator here
                self.collect_batch_stats(x.size(0), batch_min, batch_max)
            else:
                assert self.collect_batch_stats is None
        else:
            assert self.TEST_MODE == 'running'
            min_val = self.running_min
            max_val = self.running_max

        self.min_val = min_val
        self.max_val = max_val

        return x_orig
