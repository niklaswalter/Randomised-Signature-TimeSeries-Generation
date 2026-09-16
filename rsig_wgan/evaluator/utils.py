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
          

def plot_conditional_paths(
    x_past: torch.tensor,
    x_future: torch.tensor,
    generator,
    p: int,
    q: int,
    data_type: str,
    device: str,
    num_pasts: int = 4,
    num_draws: int = 30
) -> str:
    """
    For a few pasts, the real continuation against several generated ones. Pooled path plots
    cannot show whether a generator reacts to its conditioning; this can.
    """
    num_pasts = min(num_pasts, x_past.shape[0])
    fig, axes = plt.subplots(1, num_pasts, figsize=(4 * num_pasts, 3.2), sharey=True)
    sns.set_theme()
    axes = [axes] if num_pasts == 1 else list(axes)

    past_steps = range(p)
    future_steps = range(p, p + q)

    for ax, i in zip(axes, range(num_pasts)):
        past = x_past[i].reshape(1, p, 1)
        with torch.no_grad():
            fakes = generator(num_draws, q, past.to(device))
        for draw in to_numpy(fakes):
            ax.plot(future_steps, draw.reshape(-1), color="darkblue", linewidth=0.5, alpha=0.4)
        ax.plot(past_steps, to_numpy(x_past[i]).reshape(-1), color="black", linewidth=1.2)
        ax.plot(future_steps, to_numpy(x_future[i]).reshape(-1), color="dimgrey", linewidth=1.2)
        ax.axvline(p - 0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Time")

    axes[0].set_ylabel("Value")
    axes[0].legend(["Generated", "Past", "Real future"])
    fig.suptitle("Conditional {} futures given the past".format(data_type))
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plot_file_path = "plots/conditional_plot-{}.pdf".format(datetime.now().strftime("%d%m%Y-%H%M%S"))
    fig.savefig(plot_file_path)
    plt.close()
    return plot_file_path
