import torch
import pytest
from data.cifar100 import get_cifar100_loaders

ROOT = './datasets'

@pytest.fixture(scope='module')
def loaders():
    train_loader, test_loader = get_cifar100_loaders(batch_size=128)
    return train_loader, test_loader

@pytest.fixture(scope='module')
def batch(loaders):
    train_loader, _ = loaders
    x, y_c, y_f = next(iter(train_loader))
    return x, y_c, y_f

def test_batch_shape(batch):
    x, y_c, y_f = batch
    assert x.shape == (128, 3, 32, 32)
    assert y_c.shape == (128,)
    assert y_f.shape == (128,)

def test_coarse_label_range(batch):
    _, y_c, _ = batch
    assert y_c.min() >= 0
    assert y_c.max() <= 1   # only 2 superclasses

def test_fine_label_range(batch):
    _, _, y_f = batch
    assert y_f.min() >= 0
    assert y_f.max() <= 9  # only 10 fine classes

def test_fine_to_coarse_consistency(loaders):
    """Every fine label maps to exactly one coarse label."""
    train_loader, _ = loaders
    fine_to_coarse = {}
    for _, y_c, y_f in train_loader:
        for c, f in zip(y_c.tolist(), y_f.tolist()):
            if f in fine_to_coarse:
                assert fine_to_coarse[f] == c, \
                    f"Fine label {f} maps to multiple coarse labels!"
            else:
                fine_to_coarse[f] = c

def test_sample_counts(loaders):
    train_loader, test_loader = loaders
    assert len(train_loader.dataset) == 5000
    assert len(test_loader.dataset) == 1000