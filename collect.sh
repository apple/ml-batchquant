#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

while [ 1 ]
do
    python3 collect_network_info.py -b 250 -w b234_ps_7qtpgzacmi.pth -o samples -j 5 -g $1;
    sleep 1
done
