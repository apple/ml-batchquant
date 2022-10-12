#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

from qfa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3


def get_net_by_name(name):
    if name == MobileNetV3.__name__:
        return MobileNetV3
    else:
        raise ValueError('unrecognized type of network: %s' % name)
