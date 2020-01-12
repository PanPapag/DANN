import numpy
import torch
import random

"""
CONSTANTS & SEEDS INITIALIZATION
"""
LR = 1e-3
BATCH_SIZE = 128
IMAGE_STREAM = 3
IMAGE_SIZE = 28
N_EPOCHS = 100
GAMMA = 10

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
