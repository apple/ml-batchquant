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
import numpy as np


class FLOPsTable:
    def __init__(self, multiplier=1.2):
        self.multiplier = multiplier
        self.efficiency_dict = np.load('lut_flops_%.2f.npy' % multiplier, allow_pickle=True).item()

    def predict_efficiency(self, sample):
        input_size = sample.get('r', [224])
        input_size = input_size[0]
        assert 'ks' in sample and 'e' in sample and 'd' in sample
        assert len(sample['bs']) == 25
        assert len(sample['ks']) == len(sample['e']) and len(sample['ks']) == 20
        assert len(sample['d']) == 5
        total_stats = 0.
        for i in range(20):
            stage = i // 4
            depth_max = sample['d'][stage]
            depth = i % 4 + 1
            if depth > depth_max:
                continue
            ks, e, bs = sample['ks'][i], sample['e'][i], sample['bs'][i + 2]
            total, fpi, fpd, fpp = self.efficiency_dict[input_size]['mobile_inverted_blocks'][i + 1][(ks, e)]
            iw, ia, dw, da, pw, pa = [min(b, 8) for b in bs]
            total_stats += total - (fpi + fpd + fpp) + fpi * iw * ia / 64. + fpd * dw * da / 64. + fpp * pw * pa / 64.

        for key in self.efficiency_dict[input_size]['other_blocks']:
            w, a = [min(b, 8) for b in sample['bs'][key]]
            total_stats += self.efficiency_dict[input_size]['other_blocks'][key] * w * a / 64.

        total, fpi, fpd, fpp = self.efficiency_dict[input_size]['mobile_inverted_blocks'][0][(3, 1)]
        iw, ia, dw, da, pw, pa = [min(b, 8) for b in sample['bs'][1]]
        total_stats += total - (fpi + fpd + fpp) + fpi * iw * ia / 64. + fpd * dw * da / 64. + fpp * pw * pa / 64.
        return total_stats
