"""
Defines the evaluation environment
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from rsigw1 import *
from sigw1 import *
from utils import *


class Evaluation:
    """
    Class implementing the evaluation environment
    """
    def __init__(self, training, x_train, x_test, scaler, generator_id, discriminator_id, activation_id, data_type, device=DEVICE):
        self.training = training
        self.x_train = x_train
        self.x_test = x_test
        self.scaler = scaler
        self.best_generator = self.training.generator
        self.generator_id = generator_id
        self.discriminator_id = discriminator_id
        self.activation_id = activation_id
        self.num_epochs = self.training.num_grad_steps
        self.learning_rate = self.training.learning_rate
        self.activation = get_activation(ACTIVATION_ID)
        self.data_type = data_type
        self.n_lags = self.training.n_lags
        self.batch_size = self.training.batch_size
        self.device = device
        self.x_fake = self.best_generator(batch_size=self.batch_size, n_lags=self.n_lags).to(self.device)
        
        if self.discriminator_id == "RSigW1":
            self.expected_rsig_x_train = compute_rsig_td(self.x_train, self.training.A1, self.training.A2,
                                                     self.training.xi1,
                                                     self.training.xi2, RESERVOIR_DIM_METRIC, self.activation).mean(0).to(self.device)
            self.expected_rsig_x_test = compute_rsig_td(self.x_test, self.training.A1, self.training.A2, self.training.xi1,
                                                    self.training.xi2, RESERVOIR_DIM_METRIC, self.activation).mean(0).to(self.device)
            self.expected_rsig_x_fake = compute_rsig_td(self.x_fake, self.training.A1, self.training.A2, self.training.xi1,
                                                    self.training.xi2, RESERVOIR_DIM_METRIC, self.activation).mean(0).to(self.device)

            self.train_error = l2_dist(self.expected_rsig_x_train, self.expected_rsig_x_fake)
            self.test_error = l2_dist(self.expected_rsig_x_test, self.expected_rsig_x_fake)

        if self.discriminator_id == "SigW1":
            self.expected_sig_x_train = compute_exp_sig(self.x_train, TRUNCATION_DEPTH, NORMALISE_SIG).to(self.device)
            self.expected_sig_x_test = compute_exp_sig(self.x_test, TRUNCATION_DEPTH, NORMALISE_SIG).to(self.device)
            self.expected_sig_x_fake = compute_exp_sig(self.x_fake, TRUNCATION_DEPTH, NORMALISE_SIG).to(self.device)

            self.train_error = l2_dist(self.expected_sig_x_train, self.expected_sig_x_fake)
            self.test_error = l2_dist(self.expected_sig_x_test, self.expected_sig_x_fake)

        self.corr_train_error = cov_diff(self.x_train, self.x_fake)
        self.corr_test_error = cov_diff(self.x_test, self.x_fake)

        self.acf_train_error = acf_diff(self.x_train, self.x_fake, lag=self.n_lags // 2)
        self.acf_test_error = acf_diff(self.x_test, self.x_fake, lag=self.n_lags // 2)

        self.x_train_scale_inverse = self.scaler.inverse(x_train)
        self.x_test_scale_inverse = self.scaler.inverse(x_test)
        self.x_fake_scale_inverse = self.scaler.inverse(self.x_fake)

    def print_summary(self):
        file = open("path/{}/{}-{}-{}.txt".format(self.data_type, self.generator_id, self.discriminator_id,
                                                        datetime.now().strftime("%d%m%Y-%H%M%S")), 'w+')
        file.write("-------------------------------\nTRAINING SUMMARY\n------------------------------\n")
        file.write("Generator: {}\nDiscriminator: {}\nActivation: {}\nGradient steps: {}\nLearning rate:"
                   " {}\nReservoir dimension: {}\nData dimension: {}\n".format(self.generator_id,
                                                                               self.discriminator_id,
                                                                               self.activation_id, self.num_epochs,
                                                                               self.learning_rate,
                                                                               RESERVOIR_DIM_METRIC,
                                                                               DATA_DIM))
        file.write("-------------------------------\n")
        file.write("Drift BM: {}\nStd BM: {}\nDrift GBM: {}\nStd GBM: {}\nPhi AR: {}\nStd AR: {}\n".format(
            DRIFT_BM, STD_BM, DRIFT_GBM, STD_GBM, PHI, STD_AR))
        file.write("-------------------------------\n")
        file.write("{}:\n".format(self.discriminator_id))
        file.write("Training error: {:.4e}\nTest error: {:.4e}\n".format(self.train_error, self.test_error))
        file.write("-------------------------------\n")
        file.write("Correlation Metric:\n")
        file.write("Training error: {:.4e}\nTest error: {:.4e}\n".format(self.corr_train_error,
                                                                         self.corr_test_error))
        file.write("-------------------------------\n")
        file.write("Autocorrelation Metric:\n")
        file.write("Training error: {:.4e}\nTest error: {:.4e}\n".format(self.acf_train_error,
                                                                         self.acf_test_error))
        if self.data_type == "BM":
            file.write("-------------------------------\n")
            file.write("Results of Normality Test:\n")
            for i in range(1, self.n_lags):
                file.write("Normaltest #{}: {}\n".format(i, p_val_normaltest(self.x_fake, i) > 0.05))

        file.write("-------------------------------\n")
        sys.stdout = file
        print(self.training.train_losses_history)
        file.write("-------------------------------\n")
        sys.stdout = file
        print(self.training.val_losses_history)

    def save_best_generator(self):
        torch.save(self.best_generator.state_dict(), "path/{}/{}-{}-{}-{}-{}.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))

    def save_paths(self):
        torch.save(self.x_fake_scale_inverse.detach(), "path/{}/{}-{}-{}-{}-{}-fake.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
        torch.save(self.x_train_scale_inverse.detach(), "path/{}/{}-{}-{}-{}-{}-train.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
        torch.save(self.x_test_scale_inverse.detach(), "path/{}/{}-{}-{}-{}-{}-test.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))

    def plot_paths(self, num_paths=50):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.set_theme()
        for i in range(num_paths):
            plt.plot(to_numpy(self.x_fake_scale_inverse)[i], color="darkblue", linewidth=0.7)
            plt.plot(to_numpy(self.x_train_scale_inverse)[i], color="dimgrey", linewidth=0.7)
        ax.set_title("Real and generated {} paths".format(self.data_type))
        ax.legend(["Fake", "Real"])
        ax.set_xlabel("Time")
        plt.show()
        fig.savefig("path/{}/{}-{}-{}-{}-{}-plot.pdf".
                    format(self.data_type, self.generator_id, self.discriminator_id,
                           self.activation, self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
