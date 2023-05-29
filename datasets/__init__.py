from .dtu import DTUDataset
from .tanks import TanksDataset
from .blendedmvs import BlendedMVSDataset
from .eth3d import ETH3Dataset

dataset_dict = {'dtu': DTUDataset,
                'tanks': TanksDataset,
                'blendedmvs': BlendedMVSDataset, 
                'eth3d': ETH3Dataset}
