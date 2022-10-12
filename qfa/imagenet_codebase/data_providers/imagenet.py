#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Modified from Once for All: Train One Network and Specialize it for Efficient Deployment (https://github.com/mit-han-lab/once-for-all)
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
#

import warnings
import os
import os.path
import math
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from qfa.imagenet_codebase.data_providers.base_provider import DataProvider, DictTransform, LoaderConfig,\
    MyColorJitter, MyRandomHorizontalFlip, MyRandomResizedCrop, MyDistributedSampler


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_valid_file(f, extensions):
    valid = False
    if has_file_allowed_extension(f, extensions):
        # try:
        #     pil_loader(f)
        # except Exception as e:
        #     print(f)
        #     print(e)
        #     valid = False
        # else:
        #     valid = True
        valid = True
    return valid


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path, extensions):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = None
        while sample is None:
            try:
                path, target = self.samples[index]
                sample = self.loader(path)
                break
            except Exception as e:
                print(path)
                print(e)
                index = np.random.randint(len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples


class ImagenetDataProvider(DataProvider):
    DEFAULT_PATH = '{YOUR IMAGENET PATH HERE}'
    
    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):
        
        warnings.filterwarnings('ignore')
        self._save_path = save_path
        
        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)
            from qfa.imagenet_codebase.data_providers.my_data_loader import MyDataLoader
            self.image_size.sort()  # e.g., 160 -> 224
            LoaderConfig.IMAGE_SIZE_LIST = self.image_size.copy()
            LoaderConfig.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = data.DataLoader

        train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(train_transforms)
        
        if valid_size is not None:
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset.samples) * valid_size)
            
            valid_dataset = self.train_dataset(valid_transforms)
            train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.samples), valid_size)
            
            if num_replicas is not None:
                train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
                valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
            else:
                train_sampler = data.sampler.SubsetRandomSampler(train_indexes)
                valid_sampler = data.sampler.SubsetRandomSampler(valid_indexes)
            
            self.train = train_loader_class(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            if num_replicas is not None:
                train_sampler = data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
                self.train = train_loader_class(
                    train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=n_worker, pin_memory=True
                )
            else:
                self.train = train_loader_class(
                    train_dataset, batch_size=train_batch_size, shuffle=True,
                    num_workers=n_worker, pin_memory=True,
                )
            self.valid = None
        
        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
            self.test = data.DataLoader(
                test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
            )
        else:
            self.test = data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
            )
        
        if self.valid is None:
            self.valid = self.test
    
    @staticmethod
    def name():
        return 'imagenet'
    
    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W
    
    @property
    def n_classes(self):
        return 1000
    
    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
        return self._save_path
    
    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())
    
    def train_dataset(self, _transforms):
        dataset = ImageFolder(self.train_path, _transforms)
        return dataset
    
    def test_dataset(self, _transforms):
        dataset = ImageFolder(self.valid_path, _transforms)
        return dataset
    
    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')
    
    @property
    def valid_path(self):
        return os.path.join(self.save_path, 'val')
    
    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))

        if self.distort_color == 'torch':
            color_transform = MyColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif self.distort_color == 'tf':
            color_transform = MyColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print('Use MyRandomResizedCrop: %s, \t %s' % LoaderConfig.get_candidate_image_size(),
                  'sync=%s, continuous=%s' % (LoaderConfig.SYNC_DISTRIBUTED, LoaderConfig.CONTINUOUS))
        else:
            resize_transform_class = transforms.RandomResizedCrop

        if not isinstance(image_size, int):
            image_size = image_size[0]
        train_transforms = [
            resize_transform_class(image_size,
                                   scale=(self.resize_scale, 1.0),
                                   interpolation=LoaderConfig.INTERPOLATION),
            MyRandomHorizontalFlip(),
        ]
        if color_transform is not None:
            train_transforms.append(color_transform)
        train_transforms += [
            DictTransform(transforms.ToTensor()),
            DictTransform(self.normalize),
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]
    
    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None, cache=True):
        # used for resetting running statistics
        if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None or not cache:
            if num_worker is None:
                num_worker = self.train.num_workers
            
            n_samples = len(self.train.dataset.samples)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            
            new_train_dataset = self.train_dataset(
                self.build_train_transform(image_size=self.active_img_size, print_log=False))
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, np.array(chosen_indexes))
            else:
                sub_sampler = data.sampler.SubsetRandomSampler(chosen_indexes)
            sub_data_loader = data.DataLoader(
                new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=True,
            )
            if not cache:
                return sub_data_loader
            self.__dict__['sub_train_%d' % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
        return self.__dict__['sub_train_%d' % self.active_img_size]
