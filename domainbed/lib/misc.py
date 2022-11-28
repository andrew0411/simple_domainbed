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


def timestamp(fmt='%y%m%d_%H-%M-%S'):
    return datetime.now().strftime(fmt)


print(timestamp())
