"""
Main file for model training
"""

import torch

from rsig_wgan.config import load_config
from rsig_wgan.data import get_data
from rsig_wgan.discriminator_models import RSigWGANTraining, SigWGANTraining
from rsig_wgan.evaluator import Evaluator
from rsig_wgan.generator_models import LSTMGenerator, NeuralSDEGenerator


def sample_reservoir_matrices(config, device):
    """
    Random matrices and biases of the RSig-W1 reservoir. Shared with the generator
    when config.others.same_matrices is set, which assumes data_dim == brownian_dim.
    """
    res_dim = config.rsigw1.reservoir_dim_metric
    data_dim = config.timeseries.data_dim

    A1 = torch.randn(res_dim, res_dim, device=device)
    A2 = torch.randn(data_dim, res_dim, res_dim, device=device)
    xi1 = torch.randn(res_dim, 1, device=device)
    xi2 = torch.randn(data_dim, res_dim, 1, device=device)

    return A1, A2, xi1, xi2


def get_generator(config, device, A1, A2, xi1, xi2):
    if config.generator.id == "NeuralSDE":
        return NeuralSDEGenerator(
            config=config,
            device=device,
            A1=A1,
            A2=A2,
            xi1=xi1,
            xi2=xi2
        )
    elif config.generator.id == "LSTM":
        return LSTMGenerator(config=config, device=device)
    raise ValueError(f"Unknown generator id: {config.generator.id}")


def get_training(config, generator, x_train, x_val, device, A1, A2, xi1, xi2):
    if config.discriminator.id == "RSigW1":
        return RSigWGANTraining(
            x_train=x_train,
            x_val=x_val,
            generator=generator,
            config=config,
            device=device,
            A1=A1,
            A2=A2,
            xi1=xi1,
            xi2=xi2
        )
    elif config.discriminator.id == "SigW1":
        return SigWGANTraining(
            x_train=x_train,
            x_val=x_val,
            generator=generator,
            config=config,
            device=device
        )
    raise ValueError(f"Unknown discriminator id: {config.discriminator.id}")


def main():
    torch.autograd.set_detect_anomaly(True)

    config = load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data, (data_train, data_val, data_test) = get_data(config)

    A1, A2, xi1, xi2 = sample_reservoir_matrices(config, device)
    generator = get_generator(config, device, A1, A2, xi1, xi2)
    training = get_training(config, generator, data_train, data_val, device, A1, A2, xi1, xi2)
    training.fit()

    evaluator = Evaluator(
        training=training,
        x_train=data_train,
        x_test=data_test,
        config=config,
        scaler=data.scaler,
        device=device
    )
    evaluator.log_to_mlflow()


if __name__ == "__main__":
    main()
