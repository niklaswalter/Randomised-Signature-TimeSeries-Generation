"""
Main file for model training
"""

import torch

from rsig_wgan.config import ACTIVATION_REGISTRY, load_config
from rsig_wgan.data import get_data
from rsig_wgan.discriminator_models import RSigWGANTraining, SigWGANTraining
from rsig_wgan.discriminator_models.rsigcw1 import RSigCWGANTraining
from rsig_wgan.discriminator_models.sigcw1 import SigCWGANTraining
from rsig_wgan.evaluator import ConditionalEvaluator, Evaluator
from rsig_wgan.generator_models import ConditionalNeuralSDEGenerator, LSTMGenerator, NeuralSDEGenerator

CONDITIONAL_DISCRIMINATORS = ("RSigCW1", "SigCW1")


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
    elif config.generator.id == "ConditionalNeuralSDE":
        return ConditionalNeuralSDEGenerator(
            config=config,
            device=device,
            A1=A1,
            A2=A2,
            xi1=xi1,
            xi2=xi2
        )
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
    elif config.discriminator.id == "RSigCW1":
        return RSigCWGANTraining(
            x_train=x_train,
            x_val=x_val,
            batch_size=config.hyperparameters.batch_size,
            generator=generator,
            p=config.timeseries.p,
            q=config.timeseries.q,
            dim_res=config.rsigcw1.reservoir_dim_metric,
            mc_num=config.hyperparameters.mc_num,
            num_grad_steps=config.hyperparameters.gradient_steps,
            learning_rate=config.hyperparameters.learning_rate,
            activation=ACTIVATION_REGISTRY[config.cond_neural_sde.activation],
            device=device,
            A1=A1,
            A2=A2,
            xi1=xi1,
            xi2=xi2,
            past_chunk=config.hyperparameters.past_chunk
        )
    elif config.discriminator.id == "SigCW1":
        return SigCWGANTraining(
            x_train=x_train,
            x_val=x_val,
            batch_size=config.hyperparameters.batch_size,
            generator=generator,
            p=config.timeseries.p,
            q=config.timeseries.q,
            mc_num=config.hyperparameters.mc_num,
            num_grad_steps=config.hyperparameters.gradient_steps,
            learning_rate=config.hyperparameters.learning_rate,
            trunc=config.sigcw1.truncation_depth,
            device=device,
            augmented=config.sigcw1.augmented,
            past_chunk=config.hyperparameters.past_chunk
        )
    raise ValueError(f"Unknown discriminator id: {config.discriminator.id}")


def get_evaluator(config, training, x_train, x_test, scaler, device):
    evaluator = ConditionalEvaluator if config.discriminator.id in CONDITIONAL_DISCRIMINATORS else Evaluator
    return evaluator(
        training=training,
        x_train=x_train,
        x_test=x_test,
        config=config,
        scaler=scaler,
        device=device
    )


def main():
    torch.autograd.set_detect_anomaly(True)

    config = load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config.discriminator.id in CONDITIONAL_DISCRIMINATORS:
        expected_lags = config.timeseries.p + config.timeseries.q
        if config.timeseries.n_lags != expected_lags:
            raise ValueError(
                f"conditional training needs n_lags == p + q, but n_lags is "
                f"{config.timeseries.n_lags} and p + q is {expected_lags}"
            )

    data, (data_train, data_val, data_test) = get_data(config)

    A1, A2, xi1, xi2 = sample_reservoir_matrices(config, device)
    generator = get_generator(config, device, A1, A2, xi1, xi2)
    training = get_training(config, generator, data_train, data_val, device, A1, A2, xi1, xi2)
    training.fit()

    evaluator = get_evaluator(config, training, data_train, data_test, data.scaler, device)
    evaluator.log_to_mlflow()


if __name__ == "__main__":
    main()
