# !/usr/bin/python
import os
import numpy as np

ROOT = '/content/drive/My Drive/Colab Notebooks/'  # Working directory path
BASE = ROOT + 'MURA-v1.1'  # MURA dataset directory path
TRAIN_DIR = ROOT + 'MURA-v1.1/train/XR_WRIST'  # Train data directory (only wrist radiographs)
SEED = 42  # Assign a seed number
IMAGE_SIZE = (224, 224)  # Width x Height ratio of images
NUM_CHANNELS = 1
INPUT_SHAPE = (*IMAGE_SIZE, NUM_CHANNELS)
INITIAL_LEARNING_RATE = 0.0001
BATCH_SIZE = 8  # Experiment between 4 and 256
NUM_EPOCHS = 5  # Experiment between 50 and 250
MEAN = 0
STD = 0

np.random.seed(SEED)

# Create a directory for saving model's outputs
project_dir = os.path.join(os.getcwd(), 'results')
if not os.path.isdir(project_dir):
    os.makedirs(project_dir)
    
# Create a directory for saving model's checkpoints
checkpoint_dir = os.path.join(project_dir, 'models/')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Create a directory for saving model's log data
csv_loggger_dir = os.path.join(project_dir, 'logs/')
if not os.path.isdir(csv_loggger_dir):
    os.makedirs(csv_loggger_dir)

# Create a directory for saving model's output plots.
figures_dir = os.path.join(project_dir, 'figures/')
if not os.path.isdir(figures_dir):
  os.makedirs(figures_dir)