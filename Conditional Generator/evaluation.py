"""
Defines the evaluation environment
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from rsigcw1 import *
from utils import *

class Evaluation:
    def __init__(self, training, x_train, x_test, scaler, generator_id, discriminator_id, activation_id, data_type):
        self.training = training
        self.scaler = scaler
        self.x_train = self.scaler.inverse(x_train)
        self.x_test = self.scaler.inverse(x_test)
        self.indices_train = sample_indices(x_train.shape[0], 100)
        self.indices_test = sample_indices(x_test.shape[0], 100)
        self.x_past_train = self.x_train[self.indices_train][:, :P, :]
        self.x_future_train = self.x_train[self.indices_train][:, P:, :]
        self.x_past_test = self.x_test[self.indices_test][:, :P, :]
        self.x_future_test = self.x_test[self.indices_test][:, P:, :]
        self.best_generator = self.training.generator
        self.generator_id = generator_id
        self.discriminator_id = discriminator_id
        self.activation_id = activation_id
        self.num_epochs = self.training.num_grad_steps
        self.learning_rate = self.training.learning_rate
        self.data_type = data_type
        self.batch_size = self.training.batch_size
        self.reservoir_dim = RESERVOIR_DIM
        self.activation = ACTIVATION_ID
        self.x_fake_train = torch.zeros([100, 10, 1])
        self.x_fake_test = torch.zeros([100, 10, 1])
        for i in range(100):
            self.x_fake_train[i] = self.best_generator(batch_size=1, n_lags=Q, x_past=self.x_past_train[i].reshape(1, P, 1))
            self.x_fake_test[i] = self.best_generator(batch_size=1, n_lags=Q, x_past=self.x_past_test[i].reshape(1, P, 1))


    def print_summary(self):
        try:
          os.makedirs("evaluation/{}".format(self.data_type))
          file = open("evaluation/{}/{}-{}-{}.txt".format(self.data_type, self.generator_id, self.discriminator_id,
                                                          datetime.now().strftime("%d%m%Y-%H%M%S")), 'w+')
          file.write("-------------------------------\nTRAINING SUMMARY\n------------------------------\n")
          file.write("Generator: {}\nDiscriminator: {}\nActivation: {}\nGradient steps: {}\nLearning rate:"
                    " {}\nReservoir dimension: {}\n".format(self.generator_id, self.discriminator_id,
                                                            self.activation_id, self.num_epochs, self.learning_rate,
                                                            self.reservoir_dim))
          file.write("-------------------------------\n")
          file.write("Training Error:\n")
          file.write("{:.4e}\n".format(self.training.best_loss))
          file.write("-------------------------------\n")
          file.write("Autocorrelation Metric:\n")
          file.write("Training Error: {:.4e}\n".format(acf_diff(self.x_future_train, self.x_fake_train, Q//2)))
          file.write("Test Error: {:.4e}\n".format(acf_diff(self.x_future_test, self.x_fake_test, Q // 2)))
          file.write("-------------------------------\n")
          file.write("Covariance Metric:\n")
          file.write("Training Error: {:.4e}\n".format(acf_diff(self.x_future_train, self.x_fake_train, Q // 2)))
          file.write("Test Error: {:.4e}\n".format(acf_diff(self.x_future_test, self.x_fake_test, Q // 2)))
          sys.stdout = file
          print(self.training.train_losses_history)
          file.write("-------------------------------\n")
          sys.stdout = file
          print(self.training.val_losses_history)
        except FileExistsError:
          file = open("evaluation/{}/{}-{}-{}.txt".format(self.data_type, self.generator_id, self.discriminator_id,
                                                          datetime.now().strftime("%d%m%Y-%H%M%S")), 'w+')
          file.write("-------------------------------\nTRAINING SUMMARY\n------------------------------\n")
          file.write("Generator: {}\nDiscriminator: {}\nActivation: {}\nGradient steps: {}\nLearning rate:"
                    " {}\nReservoir dimension: {}\n".format(self.generator_id, self.discriminator_id,
                                                            self.activation_id, self.num_epochs, self.learning_rate,
                                                            self.reservoir_dim))
          file.write("-------------------------------\n")
          file.write("Training Error:\n")
          file.write("{:.4e}\n".format(self.training.best_loss))
          file.write("-------------------------------\n")
          file.write("Autocorrelation Metric:\n")
          file.write("Training Error: {:.4e}\n".format(acf_diff(self.x_future_train, self.x_fake_train, Q//2)))
          file.write("Test Error: {:.4e}\n".format(acf_diff(self.x_future_test, self.x_fake_test, Q // 2)))
          file.write("-------------------------------\n")
          file.write("Covariance Metric:\n")
          file.write("Training Error: {:.4e}\n".format(acf_diff(self.x_future_train, self.x_fake_train, Q // 2)))
          file.write("Test Error: {:.4e}\n".format(acf_diff(self.x_future_test, self.x_fake_test, Q // 2)))
          sys.stdout = file
          print(self.training.train_losses_history)
          file.write("-------------------------------\n")
          sys.stdout = file
          print(self.training.val_losses_history)

    def save_best_generator(self):
        try:
          os.makedirs("best_generators/{}".format(self.data_type))
          torch.save(self.best_generator.state_dict(), "best_generators/{}/{}-{}-{}-{}-{}.pt".
               format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                      self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
        except FileExistsError:
          torch.save(self.best_generator.state_dict(), "best_generators/{}/{}-{}-{}-{}-{}.pt".
                   format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                          self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))

    def save_paths(self):
        try:
          os.makedirs("best_generators/{}".format(self.data_type))
          torch.save(self.x_fake_train.detach(), "best_generators/{}/{}-{}-{}-{}-{}-fake.pt".
                    format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                            self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
          torch.save(self.x_fake_test.detach(), "best_generators/{}/{}-{}-{}-{}-{}-train.pt".
                    format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                            self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
          torch.save(self.x_train.detach(), "best_generators/{}/{}-{}-{}-{}-{}-test.pt".
                    format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                            self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
        except FileExistsError:
          torch.save(self.x_fake_train.detach(), "best_generators/{}/{}-{}-{}-{}-{}-fake.pt".
                    format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                            self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
          torch.save(self.x_fake_test.detach(), "best_generators/{}/{}-{}-{}-{}-{}-train.pt".
                    format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                            self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
          torch.save(self.x_train.detach(), "best_generators/{}/{}-{}-{}-{}-{}-test.pt".
                    format(self.data_type, self.generator_id, self.discriminator_id, self.activation,
                            self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))

    def plot_paths(self, num_paths=50):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.set_theme()
        for i in range(num_paths):
            temp = torch.cat([self.x_past_train[0].reshape(-1, 1), self.x_fake_train[i]])
            ax.plot(to_numpy(temp),color ='gray', lw=0.5)
            ax.plot(to_numpy(self.x_train[self.indices_train[0]].reshape(-1, 1)), color ='lightskyblue', lw=0.5)
        ax.axvline(x=P-1, color='red', ls=':')
        ax.set_xlabel("Time")
        ax.set_title("Generated {} paths conditioned on past path".format(self.data_type))
        plt.show()
        try:
          os.makedirs("best_generators/{}".format(self.data_type))
          fig.savefig("best_generators/{}/{}-{}-{}-{}-{}-plot.pdf".
                    format(self.data_type, self.generator_id, self.discriminator_id,
                           self.activation,self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
        except FileExistsError:
          fig.savefig("best_generators/{}/{}-{}-{}-{}-{}-plot.pdf".
                    format(self.data_type, self.generator_id, self.discriminator_id,
                           self.activation,self.num_epochs, datetime.now().strftime("%d%m%Y-%H%M%S")))
