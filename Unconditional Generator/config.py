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

SAMPLES_BM = 50000
DRIFT_BM = 0.0
STD_BM = 1.0

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
GRADIENT_STEPS = 3500
BATCH_SIZE = 6000

# R-SIG-W1

RESERVOIR_DIM_METRIC = 80

# SIG-W1

TRUNCATION_DEPTH = 4
NORMALISE_SIG = True

# NeuralSDE

INPUT_DIM_NSDE = 5
BROWNIAN_DIM = 1
RESERVOIR_DIM_GEN = 80
ACTIVATION_ID = "Sigmoid"

# LSTM

INPUT_DIM_LSTM = 5
HIDDEN_DIM_LSTM = 64
NUM_LAYERS_LSTM = 2

# Data

DATA_ID = "FOREX"

# Generator

GENERATOR_ID = "NeuralSDE"

# Discriminator

DISCRIMINATOR_ID = "RSigW1"


