import os
import tempfile

import mlflow
import mlflow.pytorch
import torch

from rsig_wgan.config import ACTIVATION_REGISTRY
from rsig_wgan.discriminator_models import RSigW1Metric, SigW1Metric

from .metrics import acf_diff, cov_diff, p_val_normaltest
from .utils import plot_data_test


class Evaluator:
    def __init__(
        self,
        training,
        x_train,
        x_test,
        config,
        scaler,
        device
    ):
        self.config = config
        self.training = training
        self.x_train = x_train
        self.x_test = x_test
        self.scaler = scaler
        self.best_generator = self.training.generator
        self.generator_id = config.generator.id
        self.discriminator_id = config.discriminator.id
        self.activation_id = config.neural_sde.activation
        self.num_epochs = config.hyperparameters.gradient_steps
        self.learning_rate = config.hyperparameters.learning_rate
        self.activation = ACTIVATION_REGISTRY[self.activation_id]
        self.data_type = config.data.id
        self.n_lags = config.timeseries.n_lags
        self.batch_size = config.hyperparameters.batch_size
        self.device = device

        with torch.no_grad():
            self.x_fake = self.best_generator(batch_size=self.batch_size, n_lags=self.n_lags).to(self.device)

            self.train_error = self.discriminator_error(self.x_train)
            self.test_error = self.discriminator_error(self.x_test)

            self.corr_train_error = cov_diff(self.x_train, self.x_fake)
            self.corr_test_error = cov_diff(self.x_test, self.x_fake)

            self.acf_train_error = acf_diff(self.x_train, self.x_fake, lag=self.n_lags // 2)
            self.acf_test_error = acf_diff(self.x_test, self.x_fake, lag=self.n_lags // 2)

            self.x_train_scale_inverse = self.scaler.inverse(self.x_train)
            self.x_test_scale_inverse = self.scaler.inverse(self.x_test)
            self.x_fake_scale_inverse = self.scaler.inverse(self.x_fake)

    def discriminator_error(self, x_real):
        """
        The metric the generator was trained against, evaluated against held-out paths
        """
        if self.discriminator_id == "RSigW1":
            metric = RSigW1Metric(
                x_real=x_real,
                config=self.config,
                A1=self.training.A1,
                A2=self.training.A2,
                xi1=self.training.xi1,
                xi2=self.training.xi2,
                device=self.device
            )
        elif self.discriminator_id == "SigW1":
            metric = SigW1Metric(x_real=x_real, config=self.config, device=self.device)
        else:
            raise ValueError(f"Unknown discriminator id: {self.discriminator_id}")
        return metric(self.x_fake)

    def normality_p_values(self):
        return [p_val_normaltest(self.x_fake, i) for i in range(1, self.n_lags)]

    def log_to_mlflow(self):
        os.environ.setdefault("MLFLOW_TRACKING_URI", self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        model_name = f"{self.generator_id}-{self.discriminator_id}-{self.n_lags}"

        with mlflow.start_run(run_name=model_name):
            mlflow.pytorch.log_model(self.best_generator, model_name)

            mlflow.log_param("gradient steps", self.num_epochs)
            mlflow.log_param("learning rate", self.learning_rate)
            mlflow.log_param("activation", self.activation_id)
            mlflow.log_param("data type", self.data_type)
            mlflow.log_param("time-steps", self.n_lags)
            mlflow.log_param("batch size", self.batch_size)
            mlflow.log_param("data dimension", self.config.timeseries.data_dim)
            mlflow.log_param("reservoir dimension metric", self.config.rsigw1.reservoir_dim_metric)
            mlflow.log_params(self.data_process_params())

            mlflow.log_metric(f"{self.discriminator_id}-error-train", float(self.train_error))
            mlflow.log_metric(f"{self.discriminator_id}-error-test", float(self.test_error))
            mlflow.log_metric("corr-error-train", float(self.corr_train_error))
            mlflow.log_metric("corr-error-test", float(self.corr_test_error))
            mlflow.log_metric("acf-error-train", float(self.acf_train_error))
            mlflow.log_metric("acf-error-test", float(self.acf_test_error))

            self.log_losses()

            if self.data_type == "BM":
                for i, p_value in enumerate(self.normality_p_values(), start=1):
                    mlflow.log_metric("normaltest-p-value", p_value, step=i)

            plot_path = plot_data_test(self.x_test_scale_inverse, self.x_fake_scale_inverse, self.data_type)
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
            "fake": self.x_fake_scale_inverse,
            "train": self.x_train_scale_inverse,
            "test": self.x_test_scale_inverse
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, path in paths.items():
                file_path = os.path.join(tmp_dir, f"{name}.pt")
                torch.save(path, file_path)
                mlflow.log_artifact(file_path, artifact_path="paths")
