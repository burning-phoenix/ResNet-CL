# config.py — single source of truth

# Dataset
NUM_COARSE_CLASSES = 2     # vehicle_1, reptiles
NUM_FINE_CLASSES = 10
INPUT_CHANNELS = 3 
INPUT_SIZE = 32
FEATURE_DIM = 512          # ResNet-18 penultimate layer


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

# Maps original CIFAR-100 coarse index to project's coarse label
COARSE_REMAP = {
    18: 0,  # vehicles_1 -> 0
    15: 1,  # reptiles   -> 1
}

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

# Maps our remapped coarse label -> list of our remapped fine labels
# Useful for init_coarse_from_fine() in the model
COARSE_TO_FINE = {
    0: [0, 1, 2, 3, 4],  # vehicles_1
    1: [5, 6, 7, 8, 9],  # reptiles
}

# Training defaults
DEFAULT_LR = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS_PER_TASK = 50

# Fisher
DEFAULT_FISHER_SAMPLES = 2000

# Paths
CHECKPOINT_DIR = "results/checkpoints"
LOG_DIR = "results/logs"
FIGURE_DIR = "results/figures"