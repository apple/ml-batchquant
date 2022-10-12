#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os
import argparse
import json
import uuid
import random
import numpy as np

import torch
from qfa.elastic_nn.networks import QFAMobileNetV3
from qfa.imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from qfa.imagenet_codebase.run_manager import ImagenetRunConfig, RunManager
from qfa.imagenet_codebase.utils import get_net_info
from qfa.elastic_nn.utils import set_activation_statistics


parser = argparse.ArgumentParser()
parser.add_argument(
    '-g',
    '--gpu',
    help='The gpu(s) to use',
    type=str,
    default='all')
parser.add_argument(
    '-b',
    '--batch-size',
    help='The batch on every device for validation',
    type=int,
    default=64)
parser.add_argument(
    '-v',
    '--valid-size',
    help='The number of images used for validation',
    type=int,
    default=10000)
parser.add_argument(
    '-j',
    '--workers',
    help='Number of workers',
    type=int,
    default=5)
parser.add_argument(
    '-w',
    '--weight',
    help='The weight to load',
    type=str)
parser.add_argument(
    '-o',
    '--output-path',
    help='The path of outputs',
    type=str,
    default='./samples')
parser.add_argument(
    '-s',
    '--seed',
    help='base of the random seed',
    type=int,
    default=0)
args = parser.parse_args()

_a = 0.4  # full res prob
_b = 0.3  # full depth prob
_c = 0.3  # fixed prec prob

CASES = [
    # Full Res, Full Depth, Fixed Prec
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True),
    (True, False, False),
    (True, False, True),
    (True, True, False),
    (True, True, True),
]


def func(b, x):
    return x if b else 1. - x


CASE_PROBS = [func(x, _a) * func(y, _b) * func(z, _c) for x, y, z in CASES]


def sample(args, ofa_network, run_config):
    full_res, full_depth, fixed_prec = random.choices(CASES, weights=CASE_PROBS, k=1)[0]
    if full_res:
        img_size = 224
    else:
        img_size = int(128 + 4 * np.random.randint(25))
    run_config.data_provider.assign_active_img_size(img_size)

    ofa_network.sample_active_subnet(res=img_size)
    if full_depth and fixed_prec:
        ofa_network.set_active_subnet(d=4, b=random.choice([2, 3, 4]))
    elif full_depth:
        ofa_network.set_active_subnet(d=4)
    elif fixed_prec:
        ofa_network.set_active_subnet(b=random.choice([2, 3, 4]))
    """ Test sampled subnet 
    """
    run_manager = RunManager('.tmp/%s' % uuid.uuid4().hex, ofa_network, run_config, init=False)

    # net info: get_net_info handles the active subnet automatically
    net_info = get_net_info(ofa_network, (3, img_size, img_size), print_info=False)

    run_manager.reset_running_statistics(net=ofa_network)
    loss, metrics = run_manager.validate(net=ofa_network, no_logs=True, is_test=True)
    net_info['loss'] = loss
    result_log = 'Results: res=%d,\t loss=%.5f,\t flops=%.5f' % (
        img_size, loss, net_info['flops'])
    for k, v in metrics.items():
        net_info[k] = v
        result_log += ',\t %s=%.4f' % (k, v)
    print(result_log)

    net_info['r'] = img_size
    net_info['ks'] = [block.mobile_inverted_conv.active_kernel_size for block in ofa_network.blocks[1:]]
    net_info['e'] = [block.mobile_inverted_conv.active_expand_ratio for block in ofa_network.blocks[1:]]
    net_info['d'] = ofa_network.runtime_depth
    net_info['b'] = {k: q.active_bit for k, q in ofa_network.quantizer_dict.items()}

    with open('%s/%s.json' % (args.output_path, uuid.uuid4().hex), 'w') as fout:
        fout.write(json.dumps(net_info, indent=4) + '\n')


def main(args):
    bits_list = [2, 3, 4]
    model = QFAMobileNetV3(n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.,
                           width_mult_list=[1.2], ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6],
                           depth_list=[2, 3, 4], bits_list=bits_list)
    set_activation_statistics(model)
    model.set_max_net()
    model_path = args.weight
    init = torch.load(model_path, map_location='cpu')['state_dict']
    model.load_state_dict(init)

    run_config = ImagenetRunConfig(train_batch_size=64, test_batch_size=args.batch_size,
                                   valid_size=args.valid_size, n_worker=args.workers)
    sample(args, model, run_config)


if __name__ == '__main__':
    if args.gpu == 'all':
        device_list = range(torch.cuda.device_count())
        args.gpu = ','.join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.batch_size = args.batch_size * max(len(device_list), 1)
    os.makedirs(args.output_path, exist_ok=True)
    main(args)
