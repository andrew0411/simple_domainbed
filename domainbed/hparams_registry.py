import numpy as np

def _hparams(algorithm, dataset, random_state):
    '''
    각 entry는 (default, random) 모양의 tuple 형태
    '''

    hparams = {}

    hparams['data_augmentation'] = (True, True)
    hparams['val_augment'] = (False, False)
    hparams['resnet18'] = (False, False)
    hparams['resnet_dropout'] = (0.0, random_state.choice([0.0, 0.1, 0.5]))
    hparams['class_balanced'] = (False, False)
    hparams['optimizer'] = ('adam', 'adam')

    hparams['freeze_bn'] = (True, True)
    hparams['pretrained'] = (True, True) # ResNet 사용할 때만
 
    hparams['lr'] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
    if dataset == 'DomainNet':
        hparams['batch_size'] = (32, int(2 ** random_state.uniform(3, 5)))
    else:
        hparams['batch_size'] = (32, int(2 ** random_state.uniform(3, 5.5)))
    if algorithm == 'ARM':
        hparams['batch_size'] = (8, 8)

    hparams['weight_decay'] = (0.0, 10 ** random_state.uniform(-6, -2))


    if algorithm in ['DANN', 'CDANN']:
        hparams['lr_g'] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        hparams['lr_d'] = (5e-5, 10 ** random_state.uniform(-5, -3.5))
        hparams['weight_decay_g'] = (0.0, 10 ** random_state.uniform(-6, -2))
        hparams['lambda'] = (1.0, 10 ** random_state.uniform(-2, 2))
        hparams['weight_decay_d'] = (0.0, 10 ** random_state.uniform(-6, -2))
        hparams['d_steps_per_g_step'] = (1, int(2 ** random_state.uniform(0, 3)))
        hparams['grad_penalty'] = (0.0, 10 ** random_state.uniform(-2, 1))
        hparams['beta1'] = (0.5, random_state.choice([0.0, 0.5]))
        hparams['mlp_width'] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams['mlp_depth'] = (3, int(random_state.choice([3, 4, 5])))
        hparams['mlp_dropout'] = (0.0, random_state.choice([0.0, 0.1, 0.5]))

    elif algorithm == 'TTA':
        hparams['tta_mix'] = (0.1, random_state.uniform(0., 0.1))
        hparams['tta_cut'] = (0.2, random_state.uniform(0., 0.25))
        hparams['cellsize'] = (10, int(10 ** random_state.uniform(1, 1.5)))

    elif algorithm == 'RSC':
        hparams['rsc_f_drop_factor'] = (1 / 3, random_state.uniform(0, 0.5))
        hparams['rsc_b_drop_factor'] = (1 / 3, random_state.uniform(0, 0.5))

    elif algorithm == 'SagNet':
        hparams['sag_w_adv'] = (0.1, 10 ** random_state.uniform(-2, 1))

    elif algorithm == 'IRM':
        hparams['irm_lambda'] = (1e2, 10 ** random_state.uniform(-1, 5))
        hparams['irm_penalty_anneal_iters'] = (500, int(10 ** random_state.uniform(0, 4)))

    elif algorithm in ['Mixup', 'OrgMixup']:
        hparams['mixup_alpha'] = (0.2, 10 ** random_state.uniform(-1, -1))

    elif algorithm == 'GroupDRO':
        hparams['groupdro_eta'] = (1e-2, 10 ** random_state.uniform(-3, -1))

    elif algorithm in ['MMD', 'CORAL']:
        hparams['mmd_gamma'] = (1.0, 10 ** random_state.uniform(-1, 1))

    elif algorithm in ['MLDG', 'SOMLDG']:
        hparams['mldg_beta'] = (1.0, 10 ** random_state.uniform(-1, 1))

    elif algorithm == 'MTL':
        hparams['mtl_ema'] = (0.99, random_state.choice([0.5, 0.9, 0.99, 1.0]))

    elif algorithm == 'VREx':
        hparams['vrex_lambda'] = (1e1, 10 ** random_state.uniform(-1, 5))
        hparams['vrex_penalty_anneal_iters'] = (500, int(10 ** random_state.uniform(0, 4)))

    elif algorithm == 'SAM':
        hparams['rho'] = (0.05, random_state.choice([0.01, 0.02, 0.05, 0.1]))

    elif algorithm == 'CutMix':
        hparams['beta'] = (1.0, 1.0)
        hparams['cutmix_prob'] = (1.0, 1.0)

    elif algorithm == 'Fish':
        hparams['meta_lr'] = (0.5, random_state.choice([0.05, 0.1, 0.5]))

    elif algorithm == 'SD':
        hparams['sd_reg'] = (0.1, 10 ** random_state.uniform(-5, -1))

    elif algorithm == 'ANDMask':
        hparams['tau'] = (1, random_state.uniform(0.5, 1.0))

    elif algorithm == 'IGA':
        hparams['penalty'] = (1000, 10 ** random_state.uniform(1, 5))

    elif algorithm == 'SANDMask':
        hparams['tau'] = (1.0, random_state.uniform(0.0, 1.0))
        hparams['k'] = (1e1, 10 ** random_state.uniform(-3, 5))

    elif algorithm == 'TRM':
        hparams['cos_lambda'] = (1e-4, 10 ** random_state.uniform(-5, 0))
        hparams['iters'] = (200, int(10 ** random_state.uniform(0, 4)))

    elif algorithm in ['IB_IRM', 'IB_ERM']:
        hparams['ib_lambda'] = (1e2, 10 ** random_state.uniform(-1, 5))
        hparams['ib_penalty_anneal_iters'] = (500, int(10 ** random_state.uniform(0, 4)))

    return hparams


def default_hparams(algorithm, dataset):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, dummy_random_state).items()}


def random_hparams(algorithm, dataset, seed):
    random_state = np.random.RandomState(seed)
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, random_state).items()}


    



