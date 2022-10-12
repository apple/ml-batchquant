# BatchQuant: Quantized-for-all Architecture Search with Robust Quantizer

This software project accompanies the research paper, [BatchQuant: Quantized-for-all Architecture Search with Robust Quantizer](https://arxiv.org/abs/2105.08952).

We propose **BatchQuant**, a novel quantizer to stabilize single-shot supernet training for joint mixed-precision quantization and architecture search. Our approach discovers quantized architectures with SOTA efficiency within fewer GPU hours than previous methods.

## How to use / evaluate **QFA Networks**
### Use

```python
from qfa.elastic_nn.networks import QFAMobileNetV3


model = QFAMobileNetV3(n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.,
                       width_mult_list=[1.2], ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6],
                       depth_list=[2, 3, 4], bits_list=[2, 3, 4])
set_activation_statistics(model)
model.set_max_net()
model_path = 'b234_ps.pth'
init = torch.load(model_path, map_location='cpu')['state_dict']
model.load_state_dict(init)

# Randomly sample sub-networks from OFA network
model.sample_active_subnet()

# Manually set the sub-network
model.set_active_subnet(ks=7, e=6, d=4, b=2)
```


### Evaluate

`python qfa_eval.py --id [0-180] 'Larger model id associates with model with larger FLOPs and higher Top 1 accuracy' `


## How to train **QFA Networks**
```bash
horovodrun -np 32 -H <server1_ip>:8,<server2_ip>:8,<server3_ip>:8,<server4_ip>:8 python main.py
```

## How to collect training data for **Accuracy Predictor**
```bash
bash collect_master.sh qfa
```

## How to run NSGA-II search procedure
```bash
python nsgaii_search.py
```

## Search space for NSGA-II crossover probability and mutation probability
```yaml
crossover_probability:
  type: CATEGORICAL
  range: [ 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5 ]
mutation_probability:
  type: CATEGORICAL
  range: [ 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
```

## Requirement
* Python 3.6+
* Pytorch 1.4.0+
* ImageNet Dataset (Set path in `qfa/imagenet_codebase/data_providers/imagenet.py#L213`)
* Horovod
* Scikit-Learn
* Skorch

## Citation
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