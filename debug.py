import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
# from domainbed.trainer import train

import warnings
warnings.filterwarnings(action='ignore')

print('import error -> None')

parser = argparse.ArgumentParser(description='Domain Generalization', allow_abbrev=False) # 축약 사용 금지
parser.add_argument('name', type=str, help='train_output에 저장될 실험의 이름')
parser.add_argument('configs', nargs='*') # 소비되어야하는 명령행의 인자 수, '*' -> 모든 명령행 인자를 리스트로 수집
parser.add_argument('--data_dir', type=str, default='datadir/')
parser.add_argument('--dataset', type=str, default='PACS')
parser.add_argument('--algorithm', type=str, default='ERM')
parser.add_argument('--trial_seed', type=int, default=0, help='Split_dataset과 random_hparams에 영향을 미치는 trial seed')
parser.add_argument('--seed', type=int, default=0, help='학습 및 나머지와 관련된 seed')
parser.add_argument('--steps', type=int, default=None, help='Epoch 수, default는 dataset 내부에 정해진 값')
parser.add_argument('--checkpoint_freq', type=int, default=None, help='N-step마다 check, default는 dataset 내부에 정해진 값')
parser.add_argument('--test_envs', type=int, nargs='+', default=None, help='특별하게 지정된 값 없으면 모든 environment에 대해서 실험')
parser.add_argument('--holdout_fraction', type=float, default=0.2)
parser.add_argument('--model_save', default=None, type=int, help='Model save를 시작할 epoch')
parser.add_argument('--tb_freq', default=10, help='TensorBoard에 몇 iteration마다 step value를 log할지')
parser.add_argument('--show', action='store_true', help='학습 시에 argument와 hyperparameter를 보여줄지 말지')
parser.add_argument('--evalmode', default='fast', help='["fast", "all"] fast이면 train_in dataset을 evaluation 때 무시함')
parser.add_argument('--prebuilder_loader', action='store_true', help='Pre-build eval loaders')
parser.add_argument('--ld_reg', type=float, default=0.01, help='정규화 impact')
args, left_argv = parser.parse_known_args()
args.deterministic = True

# Hyperparameter setting
hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
#     hparams = hparams_registry.random_hparams(args.algorithm, args.dataset)
print('Hparams')
print(hparams, '\n')

keys = ['config.yaml'] + args.configs # parser에 넣었던 config와 합치기, parser에 넣기 싫으면 그냥 config.yaml에 저장
keys = [open(key, encoding='utf8') for key in keys]
hparams = Config(*keys, default=hparams)
hparams.argv_update(left_argv)
print('Updated Hparams')
print(hparams, '\n')

timestamp = misc.timestamp() 
args.unique_name = f'{timestamp}_{args.name}'
print('Unique Name')
print(args.unique_name, '\n')

args.work_dir = Path('.') # 현재 workspace 디렉토리를 기준으로 Path 객체 생성
args.data_dir = Path(args.data_dir)

args.out_root = args.work_dir / Path('train_output') / args.dataset # os.path.join 처럼 '/' 로 경로를 연결

print('Out Root')
print(args.out_root, '\n')

args.out_dir = args.out_root / args.unique_name
args.out_dir.mkdir(exist_ok=True, parents=True) # 누락된 부모 디렉토리가 있으면 만듦

print('Out Directory')
print(args.out_dir, '\n')

writer = get_writer(args.out_root / 'runs' / args.unique_name)

logger = Logger.get(args.out_dir / 'log.txt')

cmd = ' '.join(sys.argv)

