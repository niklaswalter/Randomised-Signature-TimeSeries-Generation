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
    x_test: torch.tensor, x_fake: torch.tensor, data_type: str
) -> str:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.set_theme()
    for i in range(x_test.shape[0]//100):
        plt.plot(to_numpy(x_fake)[i], color="darkblue", linewidth=0.7)
        plt.plot(to_numpy(x_test)[i], color="dimgrey", linewidth=0.7)
    ax.set_title("Real and generated {} paths".format(data_type))
    ax.legend(["Fake", "Real"])
    ax.set_xlabel("Time")
    plot_file_path = f"plots/test_fake_plot-{datetime.now()}.pdf"
    fig.savefig(plot_file_path)
    plt.close()
    return plot_file_path
          