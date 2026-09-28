from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf


def load_config() -> dict[str, Any]:
    """
    return: 
        Dictionary with the config
    """
    try:
        config = OmegaConf.load(Path(__file__).parent / "config.yml")
        return config
    except FileNotFoundError:
        print("Error: The file at config.yml was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None
    
class ShiftedSigmoid(nn.Module):
    """Sigmoid shifted to satisfy sigma(0) = 0."""

    def forward(self, x):
        return torch.sigmoid(x) - 0.5


ACTIVATION_REGISTRY = {
    "Sigmoid": nn.Sigmoid(),
    "ShiftedSigmoid": ShiftedSigmoid(),
    "Tanh": nn.Tanh(),
}
    
__all__ = ["load_config", "ACTIVATION_REGISTRY"]

