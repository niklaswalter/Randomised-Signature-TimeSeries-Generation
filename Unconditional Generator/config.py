"""
Global variables for training configuration
"""

# GPU Setup

import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Timeseries

N_LAGS = 10
DATA_DIM = 1

# Brownian motion

SAMPLES_BM = 100000
DRIFT_BM = 0.1
STD_BM = 0.2

# Geometric Brownian motion

SAMPLES_GBM = 50000
DRIFT_GBM = 0.0
STD_GBM = 1.0
INIT_GBM = 1.0

# AR process

SAMPLES_AR = 50000
PHI = -0.1
STD_AR = 1.0

# Hyperparameters training

LEARNING_RATE = 1e-4
GRADIENT_STEPS = 10
BATCH_SIZE = 10000

# R-SIG-W1

RESERVOIR_DIM_METRIC = 80

# SIG-W1

TRUNCATION_DEPTH = 4
NORMALISE_SIG = True

# NeuralSDE

INPUT_DIM_NSDE = 32
BROWNIAN_DIM = 1
RESERVOIR_DIM_GEN = 80
ACTIVATION_ID = "Sigmoid"

# LSTM

INPUT_DIM_LSTM = 5
HIDDEN_DIM_LSTM = 64
NUM_LAYERS_LSTM = 2

# Random matrices and biases

B1, B2 = (torch.randn(RESERVOIR_DIM_GEN, RESERVOIR_DIM_GEN, device = DEVICE),
                        torch.randn(BROWNIAN_DIM, RESERVOIR_DIM_GEN, RESERVOIR_DIM_GEN, device = DEVICE))

lambda1, lambda2 = (torch.randn(RESERVOIR_DIM_GEN, 1, device = DEVICE),
                             torch.randn(BROWNIAN_DIM, RESERVOIR_DIM_GEN, 1, device = DEVICE))


# Data

DATA_ID = "BM"

# Generator

GENERATOR_ID = "NeuralSDE"

# Discriminator

DISCRIMINATOR_ID = "RSigW1"

# Switch Trainable Variance Parameter

TRAINABLE_VARIANCE = True

# Switch Same Random Matrices for Generator and RSig-W1

SAME_MATRICES = False

# Switch for time (in)homogeneous readouts

TIME_HOMOGENEOUS_READOUT = False
