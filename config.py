# config.py — single source of truth

# Dataset
NUM_COARSE_CLASSES = 2     # vehicle_1, reptiles
NUM_FINE_CLASSES = 10
INPUT_CHANNELS = 3 
INPUT_SIZE = 32
FEATURE_DIM = 512          # ResNet-18 penultimate layer
NUM_WORKERS = 1

# Human-readable names mapped to our remapped labels
COARSE_CLASSES = {
    'vehicles_1': 0,
    'reptiles':   1,
}

FINE_CLASSES = {
    'bicycle':      0,
    'bus':          1,
    'motorcycle':   2,
    'pickup_truck': 3,
    'train':        4,
    'crocodile':    5,
    'dinosaur':     6,
    'lizard':       7,
    'snake':        8,
    'turtle':       9,
}

# Original CIFAR-100 coarse labels: https://www.cs.toronto.edu/~kriz/cifar.html (they're in the order they appear in the dataset)
# Maps original CIFAR-100 coarse index to project's coarse label
COARSE_REMAP = {
    18: 0,  # vehicles_1 -> 0
    15: 1,  # reptiles   -> 1
}

# Used the helper script helpers/get_labels.py to get the original fine label ids for our target classes
# Maps original CIFAR-100 fine index -> our remapped fine label
FINE_REMAP = {
    # vehicles_1
    8:  0,  # bicycle
    13: 1,  # bus
    48: 2,  # motorcycle
    58: 3,  # pickup_truck
    90: 4,  # train
    # reptiles
    27: 5,  # crocodile
    29: 6,  # dinosaur
    44: 7,  # lizard
    78: 8,  # snake
    93: 9,  # turtle
}

# Maps our remapped fine label -> our remapped coarse label
FINE_TO_COARSE = {
    0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0,
    5 : 1, 6 : 1, 7 : 1, 8 : 1, 9 : 1,  
}

# Training defaults
DEFAULT_LR = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS_PER_TASK = 50

# Fisher
DEFAULT_FISHER_SAMPLES = 2000

# Paths
DATASETS_DIR = "datasets"
CHECKPOINT_DIR = "results/checkpoints"
LOG_DIR = "results/logs"
FIGURE_DIR = "results/figures"