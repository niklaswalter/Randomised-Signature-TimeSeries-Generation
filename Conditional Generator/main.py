"""
Main file for model training
"""

from utils import *
from rsigcw1 import *
from sigcw1 import *
from generators import *
from evaluation import *
from data import *
from config import *
import config
import generators


def get_generator(id):
    if id == "CondNeuralSDE":
        return ConditionalNeuralSDEGenerator(INPUT_DIM, DIM, RESERVOIR_DIM, BROWNIAN_DIM, get_activation(ACTIVATION_ID))


def get_training(id, data_train, data_val):
    if id == "RSigCW1":
        return RSigCWGANTraining(data_train, data_val, BATCH_SIZE, get_generator(GENERATOR_ID), P, Q, RESERVOIR_DIM,
                                 MC_NUM, GRADIENT_STEPS, LEARNING_RATE, get_activation(ACTIVATION_ID))

    if id == "SigCW1":
        return SigCWGANTraining(data_train, data_val, BATCH_SIZE, get_generator(GENERATOR_ID), P, Q, MC_NUM,
                                GRADIENT_STEPS, LEARNING_RATE, TRUNC)

def main():
    # torch.autograd.set_detect_anomaly(True)
    data = get_data(DATA_ID)[0]
    data_train, data_val, data_test = get_data(DATA_ID)[1]
    training = get_training(DISCRIMINATOR_ID, data_train, data_val)
    training.fit()
    eval = Evaluation(training, data_train, data_test, data.scaler, GENERATOR_ID, DISCRIMINATOR_ID, ACTIVATION_ID,
                      DATA_ID)
    eval.print_summary()
    eval.plot_paths(50)
    eval.save_best_generator()
    eval.save_paths()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()