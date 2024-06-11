"""
Main file for model training
"""

import torch
from utils import *
from rsigw1 import *
from sigw1 import *
from generators import *
from evaluation import *
from data import *
from config import *
import config
import generators


def get_generator(name):
    if name == "NeuralSDE":
        return NeuralSDEGenerator(INPUT_DIM_NSDE, DATA_DIM, RESERVOIR_DIM_GEN, BROWNIAN_DIM, get_activation(ACTIVATION_ID))
    elif name == "LSTM":
        return LSTMGenerator(INPUT_DIM_LSTM, DATA_DIM, HIDDEN_DIM_LSTM, NUM_LAYERS_LSTM)

def get_training(name, data_train, data_val):
    if name == "RSigW1":
        return RSigWGANTraining(data_train, data_val, BATCH_SIZE, get_generator(GENERATOR_ID), GRADIENT_STEPS,
                                LEARNING_RATE, RESERVOIR_DIM_METRIC, DATA_DIM, get_activation(ACTIVATION_ID))
    elif name == "SigW1":
        return SigWGANTraining(data_train, data_val, BATCH_SIZE, get_generator(GENERATOR_ID), GRADIENT_STEPS,
                               LEARNING_RATE, TRUNCATION_DEPTH, NORMALISE_SIG)

def main():
    torch.autograd.set_detect_anomaly(True)
    data = get_data(DATA_ID)[0]
    data_train, data_val, data_test = get_data(DATA_ID)[1]
    training = get_training(DISCRIMINATOR_ID, data_train, data_val)
    training.fit()
    evaluation = Evaluation(training, data_train, data_test, data.scaler, GENERATOR_ID, DISCRIMINATOR_ID, ACTIVATION_ID,
                            DATA_ID)
    evaluation.print_summary()
    evaluation.plot_paths(50)
    evaluation.save_paths()
    evaluation.save_best_generator()

if __name__ == '__main__':
    main()
