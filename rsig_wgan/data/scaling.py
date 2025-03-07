import torch


class Standardiser:
    """
    Class for the standardisation of data
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.shift_by = None

    def transform(self, x: torch.tensor) -> torch.tensor:
        self.mean = x.mean()
        self.std = x.std()
        return (x - self.mean) / self.std

    def inverse(self, x: torch.tensor) -> torch.tensor:
        return x * self.std + self.mean


class IDScaler:
    """
    Class for scaler applying the identity function
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.shift_by = None

    def transform(self, x: torch.tensor) -> torch.tensor:
        return x

    def inverse(self, x: torch.tensor) -> torch.tensor:
        return x