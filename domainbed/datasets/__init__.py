import torch
import numpy as np

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT

def set_transforms(dataset, data_type, hparams, algorithms_class=None):
    '''
    Parameters:
        data_type (str) -- ['train', 'valid', 'test']
    '''

    assert hparams['data_augmentation']

    additional_data = False
    if data_type == 'train':
        dataset.transforms = {'x': DBT.aug}
        additional_data = True

    elif data_type == 'valid':
        if hparams['val_augment'] is False:
            dataset.transforms = {'x': DBT.basic}

        else:
            dataset.transforms = {'x': DBT.aug}

    elif data_type == 'test':
        dataset.transforms = {'x': DBT.basic}

    else:
        raise ValueError(data_type)


    if additional_data and algorithms_class is not None:
        for key, transform in algorithms_class.transforms.items():
            dataset.transforms[key] = transform


def get_dataset(test_envs, args, hparams, algorithm_class=None):
    '''
    Parameters:
        test_envs -- ex) [0] : [env0, env1, ...] 로 구성되어있는 multidomain dataset에서 몇번째 domain를 test로 할 것인지

    Returns:
        dataset -- args로 주어진 dataset 자체
        
        in_splits -- 특정 domain dataset을 args의 holdout_fraction만큼 자르고 남은 datapoint가 모여있음
                  -- [(env0의 잘린 data에 transform 거친 data, 각 data의 weight), (env1의 ...)]

        out_splits -- 특정 domain dataset을 holdout_fraction만큼 자른 datapoint가 모여있음
    '''
    dataset = vars(datasets)[args.dataset](args.data_dir)

    in_splits = []
    out_splits = []

    for env_i, env in enumerate(dataset): # dataset 자체가 [env1, env2, env3, ...]의 형태임 (-> domainbed/datasets/datasets.py)
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i)
        )
        if env_i in test_envs:
            in_type = 'test'
            out_type = 'test'
        else:
            in_type = 'train'
            out_type = 'train'

        set_transforms(in_, in_type, hparams, algorithm_class)
        set_transforms(out, out_type, hparams, algorithm_class)

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)

        else:
            in_weights, out_weights = None, None

        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    return dataset, in_splits, out_splits





class _SplitDataset(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}

        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        ret = {'y': y}

        for key, transform in self.transforms.items():
            ret[key] = transform(x)

        return ret

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    '''
    dataset 두개를 만드는데
    첫번째는 n개의 datapoint, 두번째는 나머지 datapoint가 있음
    '''
    assert n <= len(dataset) # 여기서 dataset은 하나의 domain dataset을 의미함
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)