import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)
from domainbed.optimizers import get_optimizer

from domainbed.models.resnet_mixstyle import (
    resnet18_mixstyle_L234_p0d5_a0d1,
    resnet50_mixstyle_L234_p0d5_a0d1,
)
from domainbed.models.resnet_mixstyle2 import (
    resnet18_mixstyle2_L234_p0d5_a0d1,
    resnet50_mixstyle2_L234_p0d5_a0d1,
)
