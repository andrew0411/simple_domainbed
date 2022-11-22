import os
from PIL import ImageFile
from torchvision.datasets import ImageFolder

# 잘려있는 이미지가 있어도 그냥 불러오기
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Big Images
    'VLCS',
    'PACS',
    'OfficeHome',
    'DomainNet'
]

def get_dataset_class(dataset_name):
    '''
    주어진 이름에 맞게 dataset 클래스를 return
    '''
    if dataset_name not in globals():
        raise NotImplementedError(f'Dataset not found : {dataset_name}')
    return globals()[dataset_name]


def num_environments(dataset_name):
    '''
    주어진 이름의 dataset 클래스에 environmnet가 몇개 있는지 반환
    '''
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001 # Default, subclass에서 override 해도댐
    CHECKPOINT_FREQ = 100 # Default, subclass에서 override 해도댐
    N_WORKERS = 4 # Default, subclass에서 override 해도댐
    ENVIRONMENTS = None # Subclass에서 반드시 override
    INPUT_SHAPE = None # Subclass에서 반드시 override

    def __getitem__(self, index):
        '''
        sub-dataset을 반환, 여기서 index는 몇번째 domain 인지를 나타냄
        '''
        return self.datasets[index]

    def __len__(self):
        '''
        sub-dataset의 개수를 반환, 즉 몇개의 domain으로 구성되어있는지 반환
        '''
        return len(self.datsets)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()] # `os.scandir()` 을 통해 sub-directory 의 파일을 entry로 받아옴
        environments = sorted(environments)
        self.environments = environments

        self.datasets = []
        for env in environments:
            path = os.path.join(root, env)
            env_dataset = ImageFolder(path)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['C', 'L', 'S', 'V']

    def __init__(self, root):
        self.dir = os.path.join(root, 'VLCS/')
        super().__init__(self.dir)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['A', 'C', 'P', 'S']

    def __init__(self, root):
        self.dir = os.path.join(root, 'PACS/')
        super().__init__(self.dir)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['A', 'C', 'P', 'R']

    def __init__(self, root):
        self.dir = os.path.join(root, 'office_home/')
        super().__init__(self.dir)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ['clip', 'info', 'paint', 'quick', 'real', 'sketch']

    def __init__(self, root):
        self.dir = os.path.join(root, 'domain_net/')
        super().__init__(self.dir)
