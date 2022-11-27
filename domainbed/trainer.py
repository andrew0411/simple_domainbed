import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module

def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f'`{type(v)}` is not JSON Serializable')


def train(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info('')

    #####################################
    ########## Dataset, loader ##########
    #####################################

    args.real_test_envs = test_envs
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []

    if target_env is not None:
        testenv_name = f'te_{dataset.environments[target_env]}'
        logger.info(f'Target env = {target_env}')

    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = 'te_' + '_'.join(testenv_properties)
    
    logger.info(f'Testenv name escaping {testenv_name} -> {testenv_name.replace(".", "")}')
    testenv_name = testenv_name.replace(".", "")
    logger.info(f'Test envs = {test_envs} | name = {testenv_name}') # ex) Test envs = [0] | name = te_art_painting

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams['batch_size'], dtype=np.int)