#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import os

import pickle
import random
import json
import numpy as np
from tqdm import tqdm

import torch

from qfa.elastic_nn.utils import set_running_statistics, set_activation_statistics
from qfa.elastic_nn.networks import QFAMobileNetV3
from qfa.elastic_nn.modules.dynamic_q_layers import *
from qfa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
from qfa.imagenet_codebase.run_manager import ImagenetRunConfig, RunManager
from qfa.imagenet_codebase.utils import FLOPsTable
from skorch import NeuralNetRegressor
from sklearn.neighbors import NearestNeighbors


# Seeding
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
set_seed()


# Here are our complete hyperparameter search space FOR EACH LAYER, so the space size is (3*3)^20 * 3^5 * 17
K_CHOICES = np.array([3, 5, 7])
E_CHOICES = np.array([3, 4, 6])
D_CHOICES = np.array([2, 3, 4])
R_CHOICES = np.array([128 + 4 * i for i in range(25)])
B_CHOICES = np.array([2, 3, 4])
B_CHUNCKS = [6] * 21 + [2, 2]
_std = np.load('std@256.npy') * 100.

TOLERANCE = 6.66
PATIENCE = 15


# One hot encode genotype
def encode_onehot(g):
    """
    Can contain feature: ks, e, d, flops, (res)
    """
    onehot = []
    depth = g[40:45]
    for pos in range(20):
        code = [0., 0., 0.]
        idx, lpos = pos // 4, pos % 4
        if lpos < depth[idx]:
            code[np.where(K_CHOICES == g[pos])[0][0]] = 1.
        onehot += code
    for pos in range(20, 40):
        code = [0., 0., 0.]
        idx, lpos = (pos - 20) // 4, pos % 4
        if lpos < depth[idx]:
            code[np.where(E_CHOICES == g[pos])[0][0]] = 1.
        onehot += code
    for pos in range(40, 45):
        code = [0., 0., 0., 0.]
        for i in range(int(g[pos])):
            code[i] = 1.
        onehot += code
    code = [0. for _ in R_CHOICES]
    code[np.where(R_CHOICES == g[45])[0][0]] = 1.
    onehot += code
    for b in g[46:52]:
        code = [0., 0., 0.]
        code[np.where(B_CHOICES == b)[0][0]] = 1.
        onehot += code
    bblocks = np.array_split(g[52:52 + 6 * 20], 20)
    for pos, bblock in enumerate(bblocks):
        idx, lpos = pos // 4, pos % 4
        for b in bblock:
            code = [0., 0., 0.]
            if lpos < depth[idx]:
                code[np.where(B_CHOICES == b)[0][0]] = 1.
            onehot += code
    for b in g[52 + 6 * 20:]:
        code = [0., 0., 0.]
        code[np.where(B_CHOICES == b)[0][0]] = 1.
        onehot += code
    return np.array(onehot)


# Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
# Useful helper function, get the multiobjective pareto front of a population
# !!! Important this function works with arbitrary number of objectives
def is_pareto_efficient(scores, return_mask=True):
    """
    !!! Important this function works with arbitrary number of objectives
    Find the pareto-efficient points
    :param scores: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(scores.shape[0])
    n_points = scores.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(scores):
        nondominated_point_mask = np.any(scores > scores[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        scores = scores[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def get_gene(info):
    bs = np.concatenate(info['bs'][1:-1])
    return np.concatenate([
        info['ks'],
        info['e'],
        info['d'],
        [info['r']],
        bs
    ]).astype(np.float32)


# Load MLP trained on all samples
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(555, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 1),
        )

    def forward(self, X, **kwargs):
        return self.model(X)


# Helper function for getting accuracy prediction
def get_accuracy(regr, population):
    onehots = [encode_onehot(gene).astype(np.float32) for gene in population]
    return regr.predict(np.vstack(onehots))


def get_network():
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
    return run_manager, qfa_network


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


def get_accuracy_exact(population, flops):
    scores = None
    results = []
    for gene, flop in tqdm(list(zip(population, flops))):
        set_seed()
        run_manager, qfa_network = get_network()
        if type(gene) == np.ndarray:
            gene = gene.astype(int).tolist()
        set_active_net(run_manager.run_config, qfa_network, gene)
        qfa_network.eval()
        run_manager.reset_running_statistics(net=qfa_network)

        uwaited = 0
        total = 0.
        correct = 0.
        stopped = False
        with torch.no_grad():
            for i, (data, target) in enumerate(run_manager.run_config.test_loader):
                data, target = data.cuda(), target.cuda()
                output = qfa_network(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total += data.size(0)
                correct += pred.eq(target.view_as(pred)).sum().item()
                acc = 100. * correct / total

                if scores is not None:
                    uscore = np.array([acc + TOLERANCE * _std[i], -flop])
                    if np.any(np.all(scores >= uscore, axis=1)):
                        uwaited += 1
                        if uwaited >= PATIENCE:
                            stopped = True
                            break
                    else:
                        uwaited = 0

        if not stopped:
            if scores is None:
                scores = np.array([[acc, -flop]])
            else:
                scores = np.vstack([scores, np.array([acc, -flop])])
            print("Discovered: ", acc, flop)
            ckpt_path = './%.6f_%.6f.pth' % (acc, flop)
            ckpt = {
                'state_dict': qfa_network.state_dict(),
                'gene': gene,
                'acc': acc,
                'flops': flop,
            }
            torch.save(ckpt, ckpt_path)

        results.append(acc)
    return np.array(results)


def get_flops(flops_table, population):
    flopss = []
    for gene in population:
        sect = np.cumsum(B_CHUNCKS)
        bs = np.split(gene[46:], sect)
        bs = [[32] * 2] + bs[:-1] + [[8] * 2]
        spec = {'ks': gene[:20], 'e': gene[20:40], 'd': gene[40:45], 'r': [gene[45]], 'bs': bs}
        flopss.append(flops_table.predict_efficiency(spec))
    return np.array(flopss)


def mutate(g, mutation_probability):
    child = g.copy()
    for pos in range(len(child)):
        if np.random.rand() <= mutation_probability:
            if pos < 20:
                child[pos] = np.random.choice(K_CHOICES[K_CHOICES != g[pos]])
            elif pos < 40:
                child[pos] = np.random.choice(E_CHOICES[E_CHOICES != g[pos]])
            elif pos < 45:
                child[pos] = np.random.choice(D_CHOICES[D_CHOICES != g[pos]])
            elif pos < 46:
                child[pos] = np.random.choice(R_CHOICES[R_CHOICES != g[pos]])
            else:
                child[pos] = np.random.choice(B_CHOICES[B_CHOICES != g[pos]])
    return child


########################################################################################################################
# NSGA-II Implementation below is based on: https://github.com/MichaelAllen1966
########################################################################################################################
def cross_over(g1, g2, crossover_probability):
    child1 = g1.copy()
    child2 = g2.copy()
    distinct = np.nonzero(g1 != g2)
    for pos in distinct[0]:
        if np.random.rand() <= crossover_probability:
            child1[pos] = g2[pos]
            child2[pos] = g1[pos]
    return child1, child2


def score_population(flops_table, regr, population, exact=False):
    flops = get_flops(flops_table, population)
    if exact:
        top1 = get_accuracy_exact(population, flops)
    else:
        top1 = np.squeeze(get_accuracy(regr, population))
    scores = np.vstack([top1, -flops]).T  # We use fitness here
    return scores


def calculate_crowding(scores):
    population_size, number_of_scores = scores.shape
    crowding_matrix = np.zeros((population_size, number_of_scores))
    normed_scores = (scores - scores.min(0)) / scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])
        sorted_scores_index = np.argsort(normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1:population_size - 1] = sorted_scores[2:population_size] - sorted_scores[0:population_size - 2]

        # resort to original order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances


def reduce_by_crowding(scores, number_to_select):
    """
    This function selects a number of solutions based on tournament of
    crowding distances. Two members of the population are picked at
    random. The one with the higher crowding distance is always picked
    """
    population_ids = np.arange(scores.shape[0])
    crowding_distances = calculate_crowding(scores)
    picked_population_ids = np.zeros(number_to_select)
    picked_scores = np.zeros((number_to_select, len(scores[0, :])))
    for i in range(number_to_select):
        population_size = population_ids.shape[0]
        fighter1_id = np.random.randint(0, population_size - 1)
        fighter2_id = np.random.randint(0, population_size - 1)

        # If fighter # 1 is better
        if crowding_distances[fighter1_id] >= crowding_distances[fighter2_id]:
            # add solution to picked solutions array
            picked_population_ids[i] = population_ids[fighter1_id]

            # Add score to picked scores array
            picked_scores[i, :] = scores[fighter1_id, :]

            # remove selected solution from available solutions
            population_ids = np.delete(population_ids, fighter1_id, axis=0)
            scores = np.delete(scores, fighter1_id, axis=0)
            crowding_distances = np.delete(crowding_distances, fighter1_id, axis=0)
        else:
            picked_population_ids[i] = population_ids[fighter2_id]
            picked_scores[i, :] = scores[fighter2_id, :]
            population_ids = np.delete(population_ids, fighter2_id, axis=0)
            scores = np.delete(scores, fighter2_id, axis=0)
            crowding_distances = np.delete(crowding_distances, fighter2_id, axis=0)

    # Convert to integer
    picked_population_ids = np.asarray(picked_population_ids, dtype=int)
    return picked_population_ids


def build_pareto_population(
        population, scores, minimum_population_size, maximum_population_size):
    """
    Repeats Pareto front selection to build a population within
    defined size limits. Will reduce a Pareto front by applying crowding
    selection as necessary.
    """
    unselected_population_ids = np.arange(population.shape[0])
    all_population_ids = np.arange(population.shape[0])
    pareto_front = []
    while len(pareto_front) < minimum_population_size:
        temp_pareto_front_mask = is_pareto_efficient(
            scores[unselected_population_ids, :])
        temp_pareto_front = unselected_population_ids[temp_pareto_front_mask]

        # Check size of total parteo front.
        # If larger than maximum size reduce new pareto front by crowding
        combined_pareto_size = len(pareto_front) + len(temp_pareto_front)
        if combined_pareto_size > maximum_population_size:
            number_to_select = combined_pareto_size - maximum_population_size
            selected_individuals = (reduce_by_crowding(
                scores[temp_pareto_front], number_to_select))
            temp_pareto_front = temp_pareto_front[selected_individuals]

        # Add latest pareto front to full Pareto front
        pareto_front = np.hstack((pareto_front, temp_pareto_front))

        # Update unselected population ID by using sets to find IDs in all
        # ids that are not in the selected front
        unselected_set = set(all_population_ids) - set(pareto_front)
        unselected_population_ids = np.array(list(unselected_set))

    population = population[pareto_front.astype(int)]
    scores = scores[pareto_front.astype(int)]
    return population, scores


def breed_by_crossover(parent_1, parent_2, crossover_probability):
    return cross_over(parent_1, parent_2, crossover_probability)


def randomly_mutate_population(population, mutation_probability):
    """
    Randomly mutate population with a given individual gene mutation
    probability. Individual gene may switch between 0/1.
    """
    for i in range(len(population)):
        population[i] = mutate(population[i], mutation_probability)
    return population


def breed_population(population, crossover_probability, mutation_probability):
    """
    Create child population by repetedly calling breeding function (two parents
    producing two children), applying genetic mutation to the child population,
    combining parent and child population, and removing duplicatee chromosomes.
    """
    # Create an empty list for new population
    new_population = []
    population_size = population.shape[0]
    population_list = population.tolist()
    # Create new popualtion generating two children at a time
    while len(new_population) < population_size:
        idx_1, idx_2 = np.random.choice(len(population), 2, replace=False)
        parent_1 = population[idx_1]
        parent_2 = population[idx_2]
        child_1, child_2 = breed_by_crossover(parent_1, parent_2, crossover_probability)
        child_1, child_2 = randomly_mutate_population([child_1, child_2], mutation_probability)
        child_1, child_2 = child_1.tolist(), child_2.tolist()
        if child_1 not in new_population and child_1 not in population_list:
            new_population.append(child_1)
        if child_2 not in new_population and child_2 not in population_list:
            new_population.append(child_2)

    new_population = randomly_mutate_population(new_population, mutation_probability)

    new_population = np.array(new_population)
    new_population = np.unique(new_population, axis=0)

    return population, new_population


def main():
    set_seed()

    max_epochs = 100
    lr = 0.003
    batch_size = 2048
    optimizer = torch.optim.Adam
    maximum_generation = 1000
    population_size = 500
    crossover_probability = 0.007
    mutation_probability = 0.02

    flops_table = FLOPsTable()

    with open('qsamples_8k.json') as f:
        samples = json.load(f)
    samples = np.array(samples)

    sample_top1 = np.array([s['acc1'] for s in samples])
    sample_flops = np.array([s['lut_flops'] for s in samples])
    sample_gene = np.array([get_gene(s) for s in samples])

    X = [get_gene(info) for info in samples]
    X = np.array([encode_onehot(gene) for gene in X], dtype=np.float32)
    y = np.array([s['acc1'] for s in samples], dtype=np.float32)
    flops = np.array([s['lut_flops'] for s in samples], dtype=np.float32)

    _X = X.copy()
    _y = y.copy() / 100.
    _flops = flops.copy()
    data = np.vstack([flops, y]).T
    thresholds = [0.55, 0.4, 0.4, 0.4]
    for t in thresholds:
        nbrs = NearestNeighbors(n_neighbors = 5)
        nbrs.fit(data)
        distances, indexes = nbrs.kneighbors(data)
        inlier_index = np.where(distances.mean(axis=1) <= t)[0]
        outlier_index = np.where(distances.mean(axis=1) > t)[0]
        y_pred = np.zeros(data.shape[0]).astype(int)
        y_pred[outlier_index] = 1
        data = data[inlier_index, :]
        _X = _X[inlier_index, :]
        _y = _y[inlier_index]
        _flops = _flops[inlier_index]

    np.save('filtered_gene', _X)
    np.save('filtered_acc', _y)
    np.save('filtered_flops', _flops)

    X = np.load('filtered_gene.npy')
    y = np.load('filtered_acc.npy')
    y = y[:, None]

    net = NeuralNetRegressor(
                MyModule,
                max_epochs=max_epochs,
                lr=lr,
                batch_size=batch_size,
                optimizer=optimizer,
                train_split=None,
                # Shuffle training data on each epoch
                iterator_train__shuffle=True,
                device='cuda',
            )
    model = net.fit(X, y)
    mpath = './accuracy_predictor.sav'
    pickle.dump(model, open(mpath, 'wb'))

    regr = pickle.load(open(mpath, 'rb'))

    # Set general parameters
    minimum_population_size = population_size
    maximum_population_size = population_size

    # Create starting population
    sample_fitness = np.vstack([sample_top1, -sample_flops]).T
    init_population = sample_gene.astype(int)
    population, scores = build_pareto_population(init_population, sample_fitness, minimum_population_size,
                                                 maximum_population_size)

    # Loop through the generations of genetic algorithm
    for _ in tqdm(range(maximum_generation)):
        # Breed
        population, new_population = breed_population(population, crossover_probability, mutation_probability)
        # Score population
        new_scores = score_population(flops_table, regr, new_population, exact=False)
        population = np.vstack([population, new_population])
        scores = np.vstack([scores, new_scores])
        # Build pareto front
        population, scores = build_pareto_population(population, scores, minimum_population_size,
                                                     maximum_population_size)

    scores = score_population(flops_table, regr, population, exact=True)
    new_pareto_front_mask = is_pareto_efficient(scores)
    population = population[new_pareto_front_mask]
    scores = scores[new_pareto_front_mask]
    order = np.argsort(scores[:, 0])
    scores = scores[order]
    scores = scores.tolist()
    population = population[order].tolist()

    spath = 'search_result'
    np.save(spath, {'population': population, 'scores': scores})


if __name__ == '__main__':
    main()
