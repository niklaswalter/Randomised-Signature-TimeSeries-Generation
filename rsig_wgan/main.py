# %%
import torch
from rsig_wgan.config import load_config

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% 

# Sample random matrices for rsig-w1 metric
A1, A2 = torch.randn(
            load_config()["rsigw1"]["reservoir_dim_metric"],
            load_config()["rsigw1"]["reservoir_dim_metric"], 
            device=DEVICE,
            requires_grad=False
        ), torch.randn(
            load_config()["rsigw1"]["reservoir_dim_metric"],
            load_config()["rsigw1"]["reservoir_dim_metric"],
            device=DEVICE,
            requires_grad=False
        )

xi1, xi2 = torch.randn(
            load_config()["rsigw1"]["reservoir_dim_metric"],
            1,
            device=DEVICE,
            requires_grad=False
        ), torch.randn(
            load_config()["rsigw1"]["reservoir_dim_metric"],
            1,
            device=DEVICE,
            requires_grad=False
        )
# %%
