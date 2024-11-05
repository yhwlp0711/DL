import platform

import torch


def get_device():
    if platform.system() == 'Windows' or platform.system() == 'Linux':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif platform.system() == 'Darwin':
        return torch.device('mps')
