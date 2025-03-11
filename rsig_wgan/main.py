# %%
import torch
import torch.nn as nn
from rsig_wgan.config import load_config
from rsig_wgan.data import get_data
from rsig_wgan.generator_models import NeuralSDEGenerator

# %% 
config = load_config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# Sample random matrices for rsig-w1 metric
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
# %%

def get_activation(id):
    if id == "Sigmoid":
        return nn.Sigmoid()
    elif id == "Tanh":
        return nn.Tanh()
    
#%%

generator = NeuralSDEGenerator(
                config=config,
                activation=nn.Sigmoid(),
                device=device,
                A1=A1,
                A2=A2,
                xi1=xi1,
                xi2=xi2
            )

# %%

data = get_data(config)[0]
data_train, data_val, data_test = get_data(config)[1]
# %%
