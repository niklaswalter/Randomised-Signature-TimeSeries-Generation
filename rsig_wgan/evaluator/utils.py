import os
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow.pytorch
import seaborn as sns
import torch

from rsig_wgan.data import to_numpy


def load_model_from_mlflow(run_id: str, name: str):
    model_uri = f"runs:/{run_id}/{name}"
    model = mlflow.pytorch.load_model(model_uri)
    return model

def plot_data_test(
    x_test: torch.tensor, x_fake: torch.tensor, data_type: str, num_paths: int = 50
) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.set_theme()
    for i in range(min(num_paths, x_test.shape[0], x_fake.shape[0])):
        plt.plot(to_numpy(x_fake)[i], color="darkblue", linewidth=0.7)
        plt.plot(to_numpy(x_test)[i], color="dimgrey", linewidth=0.7)
    ax.set_title("Real and generated {} paths".format(data_type))
    ax.legend(["Fake", "Real"])
    ax.set_xlabel("Time")
    os.makedirs("plots", exist_ok=True)
    plot_file_path = "plots/test_fake_plot-{}.pdf".format(datetime.now().strftime("%d%m%Y-%H%M%S"))
    fig.savefig(plot_file_path)
    plt.close()
    return plot_file_path
          