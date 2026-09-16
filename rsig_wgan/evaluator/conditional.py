"""
Out-of-sample evaluation for the conditional generators
"""

import os
import tempfile

import mlflow
import mlflow.pytorch
import torch

from rsig_wgan.config import ACTIVATION_REGISTRY
from rsig_wgan.discriminator_models.sigcw1 import compute_sig, fit_lr_sig, predict_lr_sig
from rsig_wgan.utils import compute_rsig, fit_lr_rsig, predict_lr_rsig, sample_indices

from .metrics import acf_diff, cov_diff
from .utils import plot_conditional_paths


class ConditionalEvaluator:
    """
    Evaluates a conditional generator on held-out pasts.

    The conditional expectation is fitted on the training pasts only and then applied to the
    test pasts, so the reported metric is out-of-sample rather than the best training loss.
    """

    def __init__(
        self,
        training,
        x_train: torch.tensor,
        x_test: torch.tensor,
        config,
        scaler,
        device: str
    ):
        self.config = config
        self.training = training
        self.scaler = scaler
        self.best_generator = self.training.generator
        self.generator_id = config.generator.id
        self.discriminator_id = config.discriminator.id
        self.activation_id = config.cond_neural_sde.activation
        self.num_epochs = config.hyperparameters.gradient_steps
        self.learning_rate = config.hyperparameters.learning_rate
        self.activation = ACTIVATION_REGISTRY[self.activation_id]
        self.data_type = config.data.id
        self.batch_size = config.hyperparameters.batch_size
        self.device = device

        self.p = config.timeseries.p
        self.q = config.timeseries.q
        self.mc_num = config.hyperparameters.mc_num
        self.n_eval = config.conditional_evaluation.n_eval
        self.samples_per_past = config.conditional_evaluation.samples_per_past

        self.dim_res = config.rsigcw1.reservoir_dim_metric
        self.trunc = config.sigcw1.truncation_depth
        self.augmented = config.sigcw1.augmented
        self.A1, self.A2 = getattr(training, "A1", None), getattr(training, "A2", None)
        self.xi1, self.xi2 = getattr(training, "xi1", None), getattr(training, "xi2", None)

        self.x_fit_past = x_train[:, :self.p].to(self.device)
        self.x_fit_future = x_train[:, self.p:].to(self.device)

        self.x_train_past, self.x_train_future = self.evaluation_subset(x_train)
        self.x_test_past, self.x_test_future = self.evaluation_subset(x_test)

        self.estimator = self.fit_estimator()

        with torch.no_grad():
            self.x_fake_train = self.generate_conditional(self.x_train_past)
            self.x_fake_test = self.generate_conditional(self.x_test_past)

            self.train_error = self.conditional_error(self.x_train_past)
            self.test_error = self.conditional_error(self.x_test_past)

            self.acf_train_error = acf_diff(self.x_train_future, self.x_fake_train, lag=self.q // 2)
            self.acf_test_error = acf_diff(self.x_test_future, self.x_fake_test, lag=self.q // 2)

            self.cov_train_error = cov_diff(self.x_train_future, self.x_fake_train)
            self.cov_test_error = cov_diff(self.x_test_future, self.x_fake_test)

    def evaluation_subset(self, x):
        if self.n_eval is not None and self.n_eval < x.shape[0]:
            x = x[sample_indices(x.shape[0], self.n_eval)]
        return x[:, :self.p].to(self.device), x[:, self.p:].to(self.device)

    def fit_estimator(self):
        if self.discriminator_id == "RSigCW1":
            return fit_lr_rsig(self.x_fit_future, self.x_fit_past, self.A1, self.A2, self.xi1, self.xi2,
                               self.dim_res, self.activation, self.device)
        elif self.discriminator_id == "SigCW1":
            return fit_lr_sig(self.x_fit_future, self.x_fit_past, self.trunc, self.device, self.augmented)
        raise ValueError(f"Unknown conditional discriminator id: {self.discriminator_id}")

    def predict(self, x_past):
        if self.discriminator_id == "RSigCW1":
            return predict_lr_rsig(self.estimator, x_past, self.A1, self.A2, self.xi1, self.xi2,
                                   self.dim_res, self.activation, self.device)
        return predict_lr_sig(self.estimator, x_past, self.trunc, self.device, self.augmented)

    def features(self, x):
        if self.discriminator_id == "RSigCW1":
            return compute_rsig(x, self.A1, self.A2, self.xi1, self.xi2, self.dim_res,
                                self.activation, self.device).reshape([x.shape[0], self.dim_res])
        return compute_sig(x, self.trunc, self.device, self.augmented)

    def generate_conditional(self, x_past, samples_per_past=None):
        """
        Many futures per past. One draw each would only compare pooled marginals,
        which says nothing about the conditional law.
        """
        n = samples_per_past or self.samples_per_past
        paths = [self.best_generator(n, self.q, past.reshape(1, self.p, 1)).to(self.device) for past in x_past]
        return torch.cat(paths, dim=0)

    def conditional_moments(self, x_past, samples_per_past=None):
        """
        Mean and standard deviation of the generated futures for each past, shape
        """
        n = samples_per_past or self.samples_per_past
        means, stds = [], []
        for past in x_past:
            fakes = self.best_generator(n, self.q, past.reshape(1, self.p, 1)).to(self.device)
            means.append(fakes.mean(0).reshape(-1))
            stds.append(fakes.std(0).reshape(-1))
        return torch.stack(means), torch.stack(stds)

    def brownian_conditional_error(self, x_past, samples_per_past=None):
        """
        Exact conditional check for Brownian motion: given the past, the future level k steps
        ahead is N
        """
        drift, std = self.config.bm.drift, self.config.bm.std
        h = 1.0 / self.config.timeseries.n_lags
        with torch.no_grad():
            gen_mean, gen_std = self.conditional_moments(x_past, samples_per_past)
            last = x_past[:, -1, 0].to(self.device).unsqueeze(1)
            k = torch.arange(1, self.q + 1, device=self.device, dtype=gen_mean.dtype).unsqueeze(0)
            true_mean = last + k * drift * h
            true_std = (k * h).sqrt() * std
            return (gen_mean - true_mean).abs().mean(), (gen_std - true_std).abs().mean()

    def monte_carlo_features(self, x_past):
        """
        Monte-Carlo estimate of the conditional expected (randomised) signature per past.
        """
        fakes = [self.best_generator(self.mc_num, self.q, past.reshape(1, self.p, 1)).to(self.device)
                 for past in x_past]
        features = self.features(torch.cat(fakes, dim=0))
        return features.reshape(x_past.shape[0], self.mc_num, -1).mean(1)

    def conditional_error(self, x_past):
        predicted = self.predict(x_past).to(self.device)
        realised = self.monte_carlo_features(x_past)
        return torch.norm(predicted - realised, p=2, dim=1).mean()

    def log_to_mlflow(self):
        os.environ.setdefault("MLFLOW_TRACKING_URI", self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        model_name = f"{self.generator_id}-{self.discriminator_id}-{self.p}-{self.q}"

        with mlflow.start_run(run_name=model_name):
            mlflow.pytorch.log_model(self.best_generator, model_name)

            mlflow.log_param("gradient steps", self.num_epochs)
            mlflow.log_param("learning rate", self.learning_rate)
            mlflow.log_param("activation", self.activation_id)
            mlflow.log_param("data type", self.data_type)
            mlflow.log_param("p", self.p)
            mlflow.log_param("q", self.q)
            mlflow.log_param("batch size", self.batch_size)
            mlflow.log_param("mc num", self.mc_num)
            mlflow.log_param("evaluation pasts", self.x_test_past.shape[0])
            mlflow.log_param("samples per past", self.samples_per_past)
            mlflow.log_params(self.data_process_params())

            mlflow.log_metric(f"{self.discriminator_id}-error-train", float(self.train_error))
            mlflow.log_metric(f"{self.discriminator_id}-error-test", float(self.test_error))
            mlflow.log_metric("corr-error-train", float(self.cov_train_error))
            mlflow.log_metric("corr-error-test", float(self.cov_test_error))
            mlflow.log_metric("acf-error-train", float(self.acf_train_error))
            mlflow.log_metric("acf-error-test", float(self.acf_test_error))

            self.log_losses()

            if self.data_type == "BM":
                mean_error, std_error = self.brownian_conditional_error(self.x_test_past)
                mlflow.log_metric("bm-conditional-mean-error", float(mean_error))
                mlflow.log_metric("bm-conditional-std-error", float(std_error))

            plot_path = plot_conditional_paths(self.x_test_past, self.x_test_future, self.best_generator,
                                               self.p, self.q, self.data_type, self.device)
            mlflow.log_artifact(plot_path)
            self.log_paths()

    def data_process_params(self):
        params = {
            "BM": {"drift": self.config.bm.drift, "std": self.config.bm.std},
            "AR": {"phi": self.config.ar.phi, "std": self.config.ar.std}
        }
        return params.get(self.data_type, {})

    def log_losses(self):
        for history in (self.training.train_losses_history, self.training.val_losses_history):
            for name, losses in history.items():
                for step, loss in enumerate(losses):
                    mlflow.log_metric(name, loss, step=step)

    def log_paths(self):
        paths = {
            "fake_test": self.x_fake_test,
            "past_test": self.x_test_past,
            "future_test": self.x_test_future
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, path in paths.items():
                file_path = os.path.join(tmp_dir, f"{name}.pt")
                torch.save(path.detach(), file_path)
                mlflow.log_artifact(file_path, artifact_path="paths")
