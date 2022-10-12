#!/usr/bin/env python
import os, sys
import shutil
import datetime

from setuptools import setup, find_packages
from setuptools.command.install import install

# readme = open('README.md').read()
readme = '''
# BatchQuant: Quantized-for-all Architecture Search with Robust Quantizer [[arXiv]](https://arxiv.org/abs/2105.08952)

```BibTex
@inproceedings{NEURIPS2021_08aee627,
    author = {Bai, Haoping and Cao, Meng and Huang, Ping and Shan, Jiulong},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
    pages = {1074--1085},
    publisher = {Curran Associates, Inc.},
    title = {BatchQuant: Quantized-for-all Architecture Search with Robust Quantizer},
    url = {https://proceedings.neurips.cc/paper/2021/file/08aee6276db142f4b8ac98fb8ee0ed1b-Paper.pdf},
    volume = {34},
    year = {2021}
}
```

## Check our [GitHub](https://github.com/apple/ml-batchquant) for more details.
'''
VERSION = '0.0.1'

requirements = [
    'torch',
    'torchvision',
    'scikit-learn',
    'skorch',
]

VERSION += "_" + datetime.datetime.now().strftime('%Y%m%d%H%M')[2:]

setup(
    # Metadata
    name='QFA',
    version=VERSION,
    author='',
    author_email='',
    url='https://github.com/apple/ml-batchquant',
    description='BatchQuant: Quantized-for-all Architecture Search with Robust Quantizer',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)