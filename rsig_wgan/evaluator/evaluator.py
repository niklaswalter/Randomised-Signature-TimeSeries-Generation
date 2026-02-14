import mlflow
import mlflow.pytorch
import os

from rsig_wgan.config import ACTIVATION_REGISTRY
from rsig_wgan.discriminator_models import RSigW1Metric
from .metrics import cov_diff, acf_diff
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
        self.x_fake = self.best_generator(batch_size=self.batch_size, n_lags=self.n_lags).to(self.device)

        self.corr_train_error = cov_diff(self.x_train, self.x_fake)
        self.corr_test_error = cov_diff(self.x_test, self.x_fake)

        self.acf_train_error = acf_diff(self.x_train, self.x_fake, lag=self.n_lags // 2)
        self.acf_test_error = acf_diff(self.x_test, self.x_fake, lag=self.n_lags // 2)

        self.metric_test = RSigW1Metric(
                            x_real=self.x_test,    
                            config=config,
                            A1=self.training.A1,
                            A2=self.training.A2, 
                            xi1=self.training.xi1, 
                            xi2=self.training.xi2,
                            device=device
                        )
        
        self.training_metric_loss = self.metric_test(self.x_fake)

    def log_to_mlflow(self):
        os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow.tracking_uri
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        model_name = f"{self.generator_id}-{self.discriminator_id}-{self.n_lags}"

        with mlflow.start_run(run_name=model_name):    
            mlflow.pytorch.log_model(self.best_generator, model_name)

            mlflow.log_param("gradient steps", self.num_epochs)
            mlflow.log_param("learning rate", self.num_epochs)
            mlflow.log_param("activation", self.activation_id)
            mlflow.log_param("data type", self.data_type)
            mlflow.log_param("time-steps", self.n_lags)
            mlflow.log_param("batch size", self.batch_size)

            mlflow.log_metric("training metric on test", self.training_metric_loss)
            mlflow.log_metric("corr-error-train", self.corr_train_error)
            mlflow.log_metric("corr-error-test", self.corr_test_error)
            mlflow.log_metric("acf-error-train", self.acf_train_error)
            mlflow.log_metric("acf-error-test", self.acf_test_error)

            plot_path = plot_data_test(self.x_test, self.x_fake, self.data_type)
            mlflow.log_artifact(plot_path)