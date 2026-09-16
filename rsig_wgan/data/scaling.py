import torch


class Standardiser:
    """
    Class for the standardisation of data
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.shift_by = None

    def fit(self, x: torch.tensor) -> None:
        self.mean = x.mean()
        self.std = x.std()

    def transform(self, x: torch.tensor) -> torch.tensor:
        if self.mean is None:
            raise RuntimeError("Standardiser must be fitted before transforming")
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

    def fit(self, x: torch.tensor) -> None:
        pass

    def transform(self, x: torch.tensor) -> torch.tensor:
        return x

    def inverse(self, x: torch.tensor) -> torch.tensor:
        return x