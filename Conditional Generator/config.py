# Setup for computation machine

import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global variables for training configuration

N_LAGS = 16
P = 6
Q = 10
DIM = 1
TRUNC = 4

# Brownian motion

SAMPLES_BM = 10000
DRIFT_BM = 0.0
STD_BM = 1.0

# Geometric Brownian motion

SAMPLES_GBM = 10000
DRIFT_GBM = 0.0
STD_GBM = 1.0
INIT_GBM = 1.0

# AR process

SAMPLES_AR = 10000
PHI = -0.1
STD_AR = 1.0

# Hyperparameters training

INPUT_DIM = 30
LEARNING_RATE = 1e-4
GRADIENT_STEPS = 800
BATCH_SIZE = 450
MC_NUM = 200

# NeuralSDE

BROWNIAN_DIM = 1
RESERVOIR_DIM = 80
ACTIVATION_ID = "Sigmoid"

# Data

DATA_ID = "SP500"

# Generator

GENERATOR_ID = "CondNeuralSDE"

# Discriminator

DISCRIMINATOR_ID = "SigCW1"


A1, A2 = torch.randn(RESERVOIR_DIM, RESERVOIR_DIM, device = DEVICE), torch.randn(RESERVOIR_DIM, RESERVOIR_DIM, device = DEVICE)

xi1, xi2 = torch.randn(RESERVOIR_DIM, 1, device = DEVICE), torch.randn(RESERVOIR_DIM, 1, device = DEVICE)
