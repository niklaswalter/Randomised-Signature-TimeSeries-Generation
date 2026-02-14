# %% 
import os

import torch

from rsig_wgan.config import load_config
from rsig_wgan.data import get_data
from rsig_wgan.discriminator_models import RSigWGANTraining
from rsig_wgan.evaluator import Evaluator
from rsig_wgan.generator_models import NeuralSDEGenerator

# %% 
config = load_config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

A1, A2 = torch.randn(
            config.rsigw1.reservoir_dim_metric,
            config.rsigw1.reservoir_dim_metric, 
            device=device,
            requires_grad=False
        ), torch.randn(
            config.rsigw1.reservoir_dim_metric,
            config.rsigw1.reservoir_dim_metric,
            device=device,
            requires_grad=False
        )

xi1, xi2 = torch.randn(
            config.rsigw1.reservoir_dim_metric,
            1,
            device=device,
            requires_grad=False
        ), torch.randn(
            config.rsigw1.reservoir_dim_metric,
            1,
            device=device,
            requires_grad=False
        )
    
generator = NeuralSDEGenerator(
                config=config,
                device=device,
                A1=A1,
                A2=A2,
                xi1=xi1,
                xi2=xi2
            )

data = get_data(config)[0]
data_train, data_val, data_test = get_data(config)[1]

# %% 
training = RSigWGANTraining(
                    x_train=data_train,
                    x_val=data_val,
                    generator=generator,   
                    config=config,
                    device=device,
                    A1=A1,
                    A2=A2,
                    xi1=xi1,
                    xi2=xi2
                )

torch.autograd.set_detect_anomaly(True)
training.fit()

evaluator = Evaluator(
                training=training,
                x_train=data_train,
                x_test=data_test,
                config=config,
                scaler=None,
                device=device
            )

evaluator.log_to_mlflow()

# %%
