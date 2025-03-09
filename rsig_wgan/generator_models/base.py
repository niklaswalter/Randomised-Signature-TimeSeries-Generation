import torch.nn as nn
from torch.types import Device

class GeneratorBase(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GeneratorBase, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, batch_size: int, n_lags: int, device: str=Device):
        """
        to be specified for the individual generator
        """
        pass