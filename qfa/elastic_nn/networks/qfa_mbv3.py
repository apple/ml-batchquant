#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

import torch
import random
import numpy as np
from qfa.layers import IdentityLayer
from qfa.elastic_nn.modules.dynamic_q_op import DynamicWeightQuantizer, DynamicActivationQuantizer
from qfa.elastic_nn.modules.dynamic_q_layers import *
from qfa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3, MobileInvertedResidualBlock
from qfa.imagenet_codebase.utils import FLOPsTable
import horovod.torch as hvd


class QFAMobileNetV3(MobileNetV3):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None,
                 width_mult_list=1.0, ks_list=3, expand_ratio_list=6, depth_list=4, bits_list=32):

        self.flops_table = FLOPsTable(width_mult_list[0])
        self.width_mult_list = int2list(width_mult_list, 1)
        self.ks_list = int2list(ks_list, 1)
        self.expand_ratio_list = int2list(expand_ratio_list, 1)
        self.depth_list = int2list(depth_list, 1)
        self.bits_list = int2list(bits_list, 1)
        self.base_stage_width = base_stage_width

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()
        self.bits_list.sort()

        base_stage_width = [16, 24, 40, 80, 112, 160, 960, 1280]

        final_expand_width = [
            make_divisible(base_stage_width[-2] * max(self.width_mult_list), 8) for _ in self.width_mult_list
        ]
        last_channel = [
            make_divisible(base_stage_width[-1] * max(self.width_mult_list), 8) for _ in self.width_mult_list
        ]

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
        se_stages = [False, False, True, False, True, True]
        if depth_list is None:
            n_block_list = [1, 2, 3, 4, 2, 3]
            self.depth_list = [4, 4]
            print('Use MobileNetV3 Depth Setting')
        else:
            n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = [make_divisible(base_width * width_mult, 8) for width_mult in self.width_mult_list]
            width_list.append(width)

        input_channel = width_list[0]
        # first conv layer
        first_conv = DynamicQConvLayer(
            in_channel_list=3, out_channel_list=input_channel, bits_list=self.bits_list,
            kernel_size=3, stride=2, use_bn=True, act_func='h_swish', signed=True
        )
        first_block_conv = DynamicMBQConvLayer(
            in_channel_list=input_channel, out_channel_list=input_channel, bits_list=self.bits_list,
            kernel_size_list=3, expand_ratio_list=1,
            stride=stride_stages[0], act_func=act_stages[0], use_se=se_stages[0], signed=False
        )
        first_block = MobileInvertedResidualBlock(first_block_conv, IdentityLayer(input_channel, input_channel))

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1
        feature_dim = input_channel

        for width, n_block, s, act_func, use_se in zip(width_list[1:], n_block_list[1:],
                                                       stride_stages[1:], act_stages[1:], se_stages[1:]):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1

                mobile_inverted_conv = DynamicMBQConvLayer(
                    in_channel_list=feature_dim, out_channel_list=output_channel, kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list, bits_list=self.bits_list,
                    stride=stride, act_func=act_func, use_se=use_se, signed=True
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

        # final expand layer, feature mix layer & classifier
        final_expand_layer = DynamicQConvLayer(
            in_channel_list=feature_dim, out_channel_list=final_expand_width, bits_list=self.bits_list,
            kernel_size=1, use_bn=True, act_func='h_swish', signed=True
        )
        feature_mix_layer = DynamicQConvLayer(
            in_channel_list=final_expand_width, out_channel_list=last_channel, bits_list=self.bits_list,
            kernel_size=1, use_bn=False, act_func='h_swish', signed=False, per_channel=False
        )

        classifier = DynamicQLinearLayer(
            in_features_list=last_channel, out_features=n_classes, bits_list=self.bits_list,
            bias=True, dropout_rate=dropout_rate, signed=False
        )
        super(QFAMobileNetV3, self).__init__(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]
        self.quantizers = []
        self.quantizer_dict = {}
        for n, m in self.named_modules():
            if type(m) in [DynamicWeightQuantizer, DynamicActivationQuantizer]:
                self.quantizers.append(m)
                self.quantizer_dict[n] = m

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'QFAMobileNetV3'

    def forward(self, x):
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'

        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': QFAMobileNetV3.__name__,
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_state_dict(self, src_model_dict):
        self.load_weights_from_float_net(src_model_dict)

    def load_weights_from_float_net(self, src_model_dict):
        model_dict = self.state_dict()
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            elif 'first_conv' in key or 'blocks.0' in key or 'final_expand_layer' in key or 'feature_mix_layer' in key:
                rep_list = [
                    ('conv.weight', 'conv.conv.weight'),
                    ('bn.weight', 'conv.gamma'),
                    ('bn.bias', 'conv.beta'),
                    ('bn', 'conv')
                ]
                new_key = key
                for v1, v2 in rep_list:
                    new_key = new_key.replace(v1, v2)
            elif 'blocks' in key:
                rep_list = [
                    ('bn.bn.weight', 'conv.gamma'),
                    ('bn.bn.bias', 'conv.beta'),
                    ('bn.bn', 'conv')
                ]
                new_key = key
                for v1, v2 in rep_list:
                    new_key = new_key.replace(v1, v2)
            elif 'classifier' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            else:
                raise ValueError(key)
            if new_key not in model_dict:
                assert 'w_q' in key, '%s' % new_key
            else:
                model_dict[new_key] = src_model_dict[key]
            # assert new_key in model_dict, '%s' % new_key
            # model_dict[new_key] = src_model_dict[key]
        super(QFAMobileNetV3, self).load_state_dict(model_dict)

    def load_weights_from_net(self, src_model_dict):
        model_dict = self.state_dict()
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = src_model_dict[key]
        self.load_state_dict(model_dict)

    """ set, sample and get active sub-networks """
    def override_quantizer(self):
        for k, q in self.quantizer_dict.items():
            if 'first_conv' in k:
                q.active_bit = 32
            if 'classifier' in k:
                if q.active_bit < 8:
                    q.active_bit = 8

        bit_list = [b for b in self.bits_list if b != 32]
        for n, m in self.named_modules():
            if isinstance(m, DynamicPointQConv2d):
                w_bits = m.w_quantizer.active_bit
                a_bits = m.a_quantizer.active_bit
                if w_bits != a_bits and 32 in [w_bits, a_bits]:
                    if random.random() < 0.2:
                        m.w_quantizer.active_bit = 32
                        m.a_quantizer.active_bit = 32
                    else:
                        if m.w_quantizer.active_bit == 32:
                            m.w_quantizer.active_bit = random.choice(bit_list)
                        if m.a_quantizer.active_bit == 32:
                            m.a_quantizer.active_bit = random.choice(bit_list)

            if isinstance(m, DynamicSeparableQConv2d):
                for ks in m.w_quantizers.keys():
                    w_bits = m.w_quantizers[str(ks)].active_bit
                    a_bits = m.a_quantizers[str(ks)].active_bit
                    if w_bits != a_bits and 32 in [w_bits, a_bits]:
                        if random.random() < 0.2:
                            m.w_quantizers[str(ks)].active_bit = 32
                            m.a_quantizers[str(ks)].active_bit = 32
                        else:
                            if m.w_quantizers[str(ks)].active_bit == 32:
                                m.w_quantizers[str(ks)].active_bit = random.choice(bit_list)
                            if m.a_quantizers[str(ks)].active_bit == 32:
                                m.a_quantizers[str(ks)].active_bit = random.choice(bit_list)

            if isinstance(m, DynamicQLinear):
                w_bits = m.w_quantizer.active_bit
                a_bits = m.a_quantizer.active_bit
                if w_bits != a_bits and 32 in [w_bits, a_bits]:
                    if random.random() < 0.666:
                        m.w_quantizer.active_bit = 32
                        m.a_quantizer.active_bit = 32
                    else:
                        m.w_quantizer.active_bit = 8
                        m.a_quantizer.active_bit = 8

        bits_list = [q.active_bit for q in self.quantizers]
        return bits_list
    
    def set_max_net(self):
        return self.set_sandwich_subnet('max')

    def set_active_subnet(self, wid=None, ks=None, e=None, d=None, b=None):
        width_mult_id = int2list(wid, 4 + len(self.block_group_info))
        ks = int2list(ks, len(self.blocks) - 1)
        expand_ratio = int2list(e, len(self.blocks) - 1)
        depth = int2list(d, len(self.block_group_info))
        bits = int2list(b, len(self.quantizers))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.mobile_inverted_conv.active_kernel_size = k
            if e is not None:
                block.mobile_inverted_conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        for i, q in enumerate(self.quantizers):
            if bits[i] is not None:
                q.active_bit = bits[i]

        return self.override_quantizer()

    def set_sandwich_subnet(self, mode):
        assert mode in ['min', 'max']
        aggregate = min if mode == 'min' else max
        width_mult_setting = aggregate(self.width_mult_list)
        ks_setting = random.choice(self.ks_list)
        expand_setting = aggregate(self.expand_ratio_list)
        depth_setting = aggregate(self.depth_list)
        bits_setting = random.choice(self.bits_list)

        bits_setting = self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting, bits_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
            'b': bits_setting,
        }

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'bits':
            self.__dict__['_bits_include_list'] = include_list.copy()
        elif constraint_type == 'depth':
            self.__dict__['_depth_include_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        elif constraint_type == 'width_mult':
            self.__dict__['_widthMult_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_bits_include_list'] = None
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_ks_include_list'] = None
        self.__dict__['_widthMult_include_list'] = None

    def get_pareto_config(self, res):
        ks_setting = []
        expand_setting = []
        for block in self.blocks[1:]:
            ks_setting.append(block.mobile_inverted_conv.active_kernel_size)
            expand_setting.append(block.mobile_inverted_conv.active_expand_ratio)

        bits_setting = []
        for i, q in enumerate(self.quantizers):
            bits_setting.append(q.active_bit)

        pareto_cfg = {
            'r': [res],
            'wid': self.width_mult_list,
            'ks': ks_setting,
            'e': expand_setting,
            'd': self.runtime_depth,
            'b': bits_setting,
        }

        bs = [[self.first_conv.conv.w_quantizer.active_bit, self.first_conv.conv.a_quantizer.active_bit]]
        for block in self.blocks:
            inv_w, inv_a = 4, 4
            inv = block.mobile_inverted_conv.inverted_bottleneck
            if inv:
                inv_w = inv.conv.w_quantizer.active_bit
                inv_a = inv.conv.a_quantizer.active_bit
            dep = block.mobile_inverted_conv.depth_conv.conv
            kernel_size = dep.active_kernel_size
            dep_w = dep.w_quantizers[str(kernel_size)].active_bit
            dep_a = dep.a_quantizers[str(kernel_size)].active_bit
            pw = block.mobile_inverted_conv.point_linear.conv
            pw_w = pw.w_quantizer.active_bit
            pw_a = pw.a_quantizer.active_bit
            bs.append([inv_w, inv_a, dep_w, dep_a, pw_w, pw_a])
        bs += [
            [self.final_expand_layer.conv.w_quantizer.active_bit,
             self.final_expand_layer.conv.a_quantizer.active_bit],
            [self.feature_mix_layer.conv.w_quantizer.active_bit,
             self.feature_mix_layer.conv.a_quantizer.active_bit],
            [self.classifier.linear.w_quantizer.active_bit,
             self.classifier.linear.a_quantizer.active_bit],
        ]

        pareto_cfg['bs'] = bs

        return pareto_cfg

    def sample_active_subnet(self, res=None, subnet_seed=None):
        if subnet_seed is not None:
            random.seed(subnet_seed)

        return self.sample_active_subnet_helper()

    def sample_active_subnet_helper(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']
        bits_candidates = self.bits_list if self.__dict__.get('_bits_include_list', None) is None else \
            self.__dict__['_bits_include_list']

        # sample width_mult
        width_mult_setting = None

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        # sample bits
        bits_setting = []
        if not isinstance(bits_candidates[0], list):
            bits_candidates = [bits_candidates for _ in range(len(self.quantizers))]
        for b_set in bits_candidates:
            b = random.choice(b_set)
            bits_setting.append(b)

        bits_setting = self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting, bits_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
            'b': bits_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        first_conv = self.first_conv.get_active_subnet(in_channel=3, preserve_weight=preserve_weight)
        blocks = [MobileInvertedResidualBlock(
            self.blocks[0].mobile_inverted_conv.get_active_subnet(
                in_channel=self.first_conv.active_out_channel,
                preserve_weight=preserve_weight),
            copy.deepcopy(self.blocks[0].shortcut))]

        final_expand_layer = self.final_expand_layer.get_active_subnet(
            in_channel=self.blocks[-1].mobile_inverted_conv.active_out_channel, preserve_weight=preserve_weight)
        feature_mix_layer = self.feature_mix_layer.get_active_subnet(
            in_channel=self.final_expand_layer.active_out_channel, preserve_weight=preserve_weight)
        classifier = self.classifier.get_active_subnet(
            in_features=self.feature_mix_layer.active_out_channel, preserve_weight=preserve_weight)

        input_channel = blocks[0].mobile_inverted_conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    MobileInvertedResidualBlock(
                        self.blocks[idx].mobile_inverted_conv.get_active_subnet(in_channel=input_channel,
                                                                                preserve_weight=preserve_weight),
                        copy.deepcopy(self.blocks[idx].shortcut)
                    )
                )
                input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
            blocks += stage_blocks

        _subnet = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        return _subnet
