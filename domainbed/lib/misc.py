import hashlib
import sys
import random
import os
import shutil
import errno
from itertools import chain
from datetime import datetime
from collections import Counter
from typing import List
from contextlib import contextmanager
from subprocess import call

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_weights_for_balanced_classes(dataset):
    '''
    데이터 셋 내에 클래스당 데이터가 몇개인지 세주고, 그 데이터에 부여되는 가중치를 반환
    '''
    counts = Counter() # 각 클래스에 요소 개수를 빠르게 세줌, 클래스가 key, 개수가 value 값으로 저장되는 dictionary 형태
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)
    
    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def seed_hash(*args):
    '''
    Random seed를 만들때 사용되는 integer hash를 만듦
    '''
    args_str = str(args)
    return int(hashlib.md5(args_str.encode('utf-8')).hexdigest(), 16) % (2 ** 31) # hashlib 을 사용해서 md5 암호화


def to_row(row, colwidth=10, latex=False):
    if latex:
        sep = ' & '
        end_ = '\\\\'

    else:
        sep = ' '
        end_ = ''

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = '{:.6f}'.format(x)
        return str(x).ljust(colwidth)[:colwidth]

    return sep.join([format_val(x) for x in row]) + ' ' + end_


def timestamp(fmt='%y%m%d_%H-%M-%S'):
    return datetime.now().strftime(fmt)


def index_conditional_iterate(skip_condition, iterable, index):
    for i, x in enumerate(iterable):
        if skip_condition(i):
            continue

        if index:
            yield i, x

        else:
            yield x


class SplitIterator:
    def __init__(self, test_envs):
        self.test_envs = test_envs

    def train(self, iterable, index=False):
        '''
        만약 test_env에 속해있으면 넘어가고 아니면 그 index를 반환
        '''
        return index_conditional_iterate(lambda idx: idx in self.test_envs, iterable, index) 

    def test(self, iterable, index=False):
        return index_conditional_iterate(lambda idx: idx not in self.test_envs, iterable, index)


def merge_dictlist(dictlist):
    ret = {
        k: []
        for k in dictlist[0].keys()
    }
    for dic in dictlist:
        for data_key, v in dic.items():
            ret[data_key].append(v)
    return ret