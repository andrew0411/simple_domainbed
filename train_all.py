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
from domainbed.trainer import train

import warnings
warnings.filterwarnings(action='ignore')


def main():
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

    keys = ['config.yaml'] + args.configs # parser에 넣었던 config와 합치기, parser에 넣기 싫으면 그냥 config.yaml에 저장

    keys = [open(key, encoding='utf8') for key in keys]

    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv) # 실시간으로 CLI update

    timestamp = misc.timestamp() 
    args.unique_name = f'{timestamp}_{args.name}' # 현 시간 기준으로 실험 이름 결정

    # Path Setting
    args.work_dir = Path('.') # 현재 workspace 디렉토리를 기준으로 Path 객체 생성
    args.data_dir = Path(args.data_dir) 

    args.out_root = args.work_dir / Path('train_output') / args.dataset # os.path.join 처럼 '/' 로 경로를 연결
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True) # 누락된 부모 디렉토리가 있으면 만듦

    writer = get_writer(args.out_root / 'runs' / args.unique_name) # 
    logger = Logger.get(args.out_dir / 'log.txt')

    cmd = ' '.join(sys.argv)
    
    # 콘솔 창에 입력한 command
    logger.info(f'Command :: {cmd}')
    
    # 가상환경 패키지 정보
    logger.nofmt('Environment:')
    logger.nofmt(f'\tPython : {sys.version.split(" ")[0]}')
    logger.nofmt(f'\tPytorch : {torch.__version__}')
    logger.nofmt(f'\tTorchvision : {torchvision.__version__}')
    logger.nofmt(f'\tCUDA : {torch.version.cuda}')
    logger.nofmt(f'\tCUDNN : {torch.backends.cudnn.version()}')
    logger.nofmt(f'\tNumpy : {np.__version__}')
    logger.nofmt(f'\tPIL : {PIL.__version__}')

    assert torch.cuda.is_available(), 'CUDA is not available'

    # Argument 정보
    logger.nofmt('Args:')
    for k, v in sorted(vars(args).items()):
        logger.nofmt(f'\t{k} : {v}')

    # Hyperparameter 정보
    logger.nofmt('Hparams:')
    for line in hparams.dumps().split('\n'):
        logger.nofmt('\t' + line)


    if args.show:
        exit()

    # Seed 고정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Dataset 정보 확인
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)
    
    logger.nofmt('Dataset:')
    logger.nofmt(f'\t[{args.dataset}] #ENVS = {len(dataset)} | #CLASSES = {dataset.num_classes}')
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f'\tENV{i} : {env_property} (#{len(dataset[i])})')
    logger.nofmt('')

    # 현재 실험에 대한 정보 확인 (ex. 에폭수, 체크포인트, test domain 들)
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f'n_steps = {n_steps}')
    logger.info(f'checkpoint_freq = {checkpoint_freq}')

    if not args.test_envs:
        args.test_envs = [[te] for te in range(len(dataset))]
    logger.info(f'Target test envs = {args.test_envs}') # 예를 들면 args.test_envs = [[0], [1], [2], [3]] 


    #####################################
    ############# 학습 시작 #############
    #####################################

    all_records = []
    results = collections.defaultdict(list)

    for test_env in args.test_envs:
        res, records = train(
            test_env,
            args=args,
            hparams=hparams,
            n_steps=n_steps,
            checkpoint_freq=checkpoint_freq,
            logger=logger,
            writer=writer
        )
        all_records.append(records)
        for k, v in res.items():
            results[k].append(v)

    # log summary table
    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)

    table = PrettyTable(["Selection"] + dataset.environments + ["Avg."])
    for key, row in results.items():
        row.append(np.mean(row))
        row = [f"{acc:.3%}" for acc in row]
        table.add_row([key] + row)
    logger.nofmt(table)


if __name__ == '__main__':
    main()