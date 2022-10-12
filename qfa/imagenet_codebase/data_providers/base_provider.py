#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

import numpy as np
import time
import random
import os
import math

import torch
from torch._six import container_abcs
import torch.nn.functional as TF
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


class DataProvider:
    SUB_SEED = 937162211  # random seed for sampling subset
    VALID_SEED = 2147483647  # random seed for the validation set

    @staticmethod
    def name():
        """ Return name of the dataset """
        raise NotImplementedError

    @property
    def data_shape(self):
        """ Return shape as python list of one data entry """
        raise NotImplementedError

    @property
    def n_classes(self):
        """ Return `int` of num classes """
        raise NotImplementedError

    @property
    def save_path(self):
        """ local path to save the data """
        raise NotImplementedError

    @property
    def data_url(self):
        """ link to download the data """
        raise NotImplementedError

    @staticmethod
    def random_sample_valid_set(train_size, valid_size):
        assert train_size > valid_size

        g = torch.Generator()
        g.manual_seed(DataProvider.VALID_SEED)  # set random seed before sampling validation set
        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        valid_indexes = rand_indexes[:valid_size]
        train_indexes = rand_indexes[valid_size:]
        return train_indexes, valid_indexes

    @staticmethod
    def labels_to_one_hot(n_classes, labels):
        new_labels = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels


def dict_apply(func, data):
    if isinstance(data, container_abcs.Mapping):
        return {k: func(v) for k, v in data.items()}
    return func(data)


class DictTransform(torch.nn.Module):
    def __init__(self, transform):
        super(DictTransform, self).__init__()
        self.transform = transform

    def forward(self, data):
        return dict_apply(self.transform, data)


class MyRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, data):
        if torch.rand(1) < self.p:
            return dict_apply(F.hflip, data)
        return data


class MyColorJitter(transforms.ColorJitter):
    def __call__(self, img):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = dict_apply(lambda x: F.adjust_brightness(x, brightness_factor), img)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = dict_apply(lambda x: F.adjust_contrast(x, contrast_factor), img)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = dict_apply(lambda x: F.adjust_saturation(x, saturation_factor), img)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = dict_apply(lambda x: F.adjust_hue(x, hue_factor), img)

        return img


class LoaderConfig:
    ACTIVE_SIZES = {224}
    IMAGE_SIZE_LIST = [224]
    MIN_SIZE = 224
    MAX_SIZE = 224

    CONTINUOUS = False
    INCREMENT = 4
    SYNC_DISTRIBUTED = False

    DYNAMIC_BATCH_SIZE = 1
    TEACHER_IMAGE_SIZE = None
    INTERPOLATION = 2  # 2 for bilinear, 3 for bicubic

    SANDWICH = False  # True if need to generate max and min image size for sandwich rule training

    EPOCH = 0

    @staticmethod
    def get_candidate_image_size():
        if LoaderConfig.CONTINUOUS:
            min_size = min(LoaderConfig.IMAGE_SIZE_LIST)
            max_size = max(LoaderConfig.IMAGE_SIZE_LIST)
            candidate_sizes = []
            for i in range(min_size, max_size + 1):
                if (i-min_size) % LoaderConfig.INCREMENT == 0:
                    candidate_sizes.append(i)
        else:
            candidate_sizes = LoaderConfig.IMAGE_SIZE_LIST

        relative_probs = None
        return candidate_sizes, relative_probs

    @staticmethod
    def sample_image_size(batch_id):
        if LoaderConfig.SYNC_DISTRIBUTED:
            _seed = int('%d%.3d' % (batch_id, LoaderConfig.EPOCH))
        else:
            _seed = os.getpid() + time.time()
        random.seed(_seed)
        candidate_sizes, relative_probs = LoaderConfig.get_candidate_image_size()
        LoaderConfig.ACTIVE_SIZES = random.choices(candidate_sizes,
                                                   k=LoaderConfig.DYNAMIC_BATCH_SIZE,
                                                   weights=relative_probs)
        if LoaderConfig.SANDWICH:
            LoaderConfig.MIN_SIZE = min(candidate_sizes)
            LoaderConfig.MAX_SIZE = max(candidate_sizes)


class MyResize(torch.nn.Module):
    def __init__(self):
        super(MyResize, self).__init__()
        self.mode = ['nearest', 'linear', 'bilinear', 'bicubic'][LoaderConfig.INTERPOLATION]

    def __call__(self, img):
        imgs = {}
        if LoaderConfig.TEACHER_IMAGE_SIZE is not None:
            imgs['teacher'] = TF.interpolate(img.unsqueeze(0), size=(LoaderConfig.TEACHER_IMAGE_SIZE, LoaderConfig.TEACHER_IMAGE_SIZE),
                                             mode=self.mode, align_corners=True).squeeze().detach()
        if LoaderConfig.SANDWICH:
            imgs['min'] = TF.interpolate(img.unsqueeze(0), size=(LoaderConfig.MIN_SIZE, LoaderConfig.MIN_SIZE), mode=self.mode, align_corners=True).squeeze().detach()
            imgs['max'] = TF.interpolate(img.unsqueeze(0), size=(LoaderConfig.MAX_SIZE, LoaderConfig.MAX_SIZE), mode=self.mode, align_corners=True).squeeze().detach()
        for i, size in enumerate(LoaderConfig.ACTIVE_SIZES):
            imgs[i] = TF.interpolate(img.unsqueeze(0), size=(size, size), mode=self.mode, align_corners=True).squeeze().detach()
        return imgs


class MyRandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        # TODO: generate a dict here
        imgs = {}
        if LoaderConfig.TEACHER_IMAGE_SIZE is not None:
            imgs['teacher'] = F.resized_crop(
                img, i, j, h, w,
                (LoaderConfig.TEACHER_IMAGE_SIZE, LoaderConfig.TEACHER_IMAGE_SIZE),
                self.interpolation
            )
        if LoaderConfig.SANDWICH:
            min_size = min(LoaderConfig.IMAGE_SIZE_LIST)
            max_size = max(LoaderConfig.IMAGE_SIZE_LIST)
            imgs['min'] = F.resized_crop(
                img, i, j, h, w, (min_size, min_size), self.interpolation
            )
            imgs['max'] = F.resized_crop(
                img, i, j, h, w, (max_size, max_size), self.interpolation
            )
        for i, size in enumerate(LoaderConfig.ACTIVE_SIZES):
            imgs[i] = F.resized_crop(
                img, i, j, h, w, (size, size), self.interpolation
            )
        return imgs


class MyDistributedSampler(DistributedSampler):
    """ Allow Subset Sampler in Distributed Training """

    def __init__(self, dataset, num_replicas=None, rank=None, sub_index_list=None):
        super(MyDistributedSampler, self).__init__(dataset, num_replicas, rank)
        self.sub_index_list = sub_index_list  # numpy

        self.num_samples = int(math.ceil(len(self.sub_index_list) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        print('Use MyDistributedSampler: %d, %d' % (self.num_samples, self.total_size))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.sub_index_list), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        indices = self.sub_index_list[indices].tolist()
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
