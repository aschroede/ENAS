# File: nasbench201_loader.py

import pickle
import os
import numpy as np
from nas_201_api import NASBench201API
import random
import torch
from nas_201_api import NASBench201API as API

HP_EPOCH = 12

def load_nasbench201(path='data/NAS-Bench-201-v1_1-096897.pth'):

    assert os.path.isfile(path), f"NASBench-201 file not found: {path}"

    original_load = torch.load

    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = safe_load


    # # Later reuse
    # with open('nasbench201_api.pkl', 'rb') as f:
    #     api = pickle.load(f)

    try:
        api = API('data/NAS-Bench-201-v1_1-096897.pth', verbose=True)
    finally:

        torch.load = original_load

    # with open('nasbench201_api.pkl', 'wb') as f:
    #     pickle.dump(api, f)

    arch_list = list(api)
    print("Finished loading API...")
    return api, arch_list

def query_nasbench201(api, arch_str, hp='200'):
    index = api.query_index_by_arch(arch_str)
    # info = api.query_meta_info_by_index(index)
    gt = api.get_more_info(index, dataset='cifar10', iepoch=None, hp=str(HP_EPOCH), is_random=False)
    train_acc = gt['train-accuracy']
    # val_acc = gt['valid-accuracy']
    test_acc = gt['test-accuracy']
    return train_acc, test_acc

def mutate_architecture(arch_str):
    ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    parts = arch_str.split('+')
    node_to_mutate = np.random.randint(len(parts))
    ops_in_node = parts[node_to_mutate].split('|')[1:-1]
    op_to_mutate = np.random.randint(len(ops_in_node))

    current_op = ops_in_node[op_to_mutate].split('~')[0]
    new_op = random.choice([op for op in ops if op != current_op])
    parts[node_to_mutate] = parts[node_to_mutate].replace(current_op, new_op)
    return '+'.join(parts)
