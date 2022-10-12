#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import argparse
import pickle
import random
from glob import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch

import uuid

from qfa.imagenet_codebase.utils import get_net_info
from qfa.elastic_nn.utils import set_running_statistics, set_activation_statistics
from qfa.elastic_nn.networks import OFAMobileNetV3, QFAMobileNetV3
from qfa.elastic_nn.modules.dynamic_q_layers import *
from qfa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
from qfa.imagenet_codebase.run_manager import ImagenetRunConfig, RunManager


parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--id',
    help='The model id to evaluate, model id range = [0, 180]',
    type=int,
    default=0)
args = parser.parse_args()


def set_seed():
    seed = 2021
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # This can slow down training
    
    
def set_active_net(run_config, model, gene):
    ks, e, d, r, bs = gene[:20], gene[20:40], gene[40:45], int(gene[45]), gene[46:]
    run_config.data_provider.assign_active_img_size(r)
    model.set_active_subnet(ks=ks, e=e, d=d)

    for block in model.blocks:
        inv = block.mobile_inverted_conv.inverted_bottleneck
        if inv:
            inv.conv.w_quantizer.active_bit = bs.pop(0)
            inv.conv.a_quantizer.active_bit = bs.pop(0)
        else:
            bs.pop(0)
            bs.pop(0)
        dep = block.mobile_inverted_conv.depth_conv.conv
        kernel_size = dep.active_kernel_size
        dep.w_quantizers[str(kernel_size)].active_bit = bs.pop(0)
        dep.a_quantizers[str(kernel_size)].active_bit = bs.pop(0)
        pw = block.mobile_inverted_conv.point_linear.conv
        pw.w_quantizer.active_bit = bs.pop(0)
        pw.a_quantizer.active_bit = bs.pop(0)
    model.final_expand_layer.conv.w_quantizer.active_bit = bs.pop(0)
    model.final_expand_layer.conv.a_quantizer.active_bit = bs.pop(0)
    model.feature_mix_layer.conv.w_quantizer.active_bit = bs.pop(0)
    model.feature_mix_layer.conv.a_quantizer.active_bit = bs.pop(0)
    assert len(bs) == 0, len(bs)
    net_cfg = model.get_pareto_config(r)
    flops = model.flops_table.predict_efficiency(net_cfg)
    return flops


def eval_gene(population, idx):
    set_seed()
    qfa_network = QFAMobileNetV3(n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.,
                                 width_mult_list=[1.2], ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6],
                                 depth_list=[2, 3, 4], bits_list=[2, 3, 4])
    qfa_network.set_active_subnet(1.2, 7, 6, 4, 4)
    set_activation_statistics(qfa_network)
    model_path = 'b234_ps.pth'
    init = torch.load(model_path, map_location='cpu')['state_dict']
    qfa_network.load_state_dict(init)
    qfa_network.cuda()
    qfa_network.eval()

    run_config = ImagenetRunConfig(train_batch_size=64, test_batch_size=256, valid_size=10000, n_worker=10)
    run_config.data_provider.assign_active_img_size(224)
    run_manager = RunManager('.tmp/', qfa_network, run_config, init=False)
    
    gene = population[idx]
    if type(gene) == np.ndarray:
        gene = gene.astype(int).tolist()
    flops = set_active_net(run_manager.run_config, qfa_network, gene)
    qfa_network.eval()
    run_manager.reset_running_statistics(net=qfa_network)
    loss, metric_dict = run_manager.validate(net=qfa_network, no_logs=True, is_test=True)
    acc1 = metric_dict['acc1']
    print("| QFA_%s | %.3f | %.3fM |" % (str(idx).zfill(3), acc1, flops))


def main():
    set_seed()
    population = np.load('pareto_archs.npy')
    eval_gene(population, args.id)
    
    
if __name__ == '__main__':
    main()
