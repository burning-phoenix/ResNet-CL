"""Trainer tests (training.trainer). Run with run_tests.py - only test_e_* are run."""
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import NUM_COARSE_CLASSES, NUM_FINE_CLASSES
from training.trainer import train_one_phase, save_checkpoint, load_checkpoint


class _MockModel(torch.nn.Module):
    """Minimal model for trainer: forward(x, task_id) -> logits."""

    def __init__(self, coarse_classes=2, fine_classes=10):
        super().__init__()
        self.coarse_classes = coarse_classes
        self.fine_classes = fine_classes
        self.linear = torch.nn.Linear(3 * 32 * 32, max(coarse_classes, fine_classes))

    def forward(self, x, task_id):
        flat = x.flatten(1)
        logits = self.linear(flat)
        n = self.coarse_classes if task_id == "coarse" else self.fine_classes
        return logits[:, :n]


@pytest.fixture
def tiny_train_loader():
    """Small dataset for a quick training step."""
    n = 16
    x = torch.randn(n, 3, 32, 32)
    y_coarse = torch.randint(0, NUM_COARSE_CLASSES, (n,))
    y_fine = torch.randint(0, NUM_FINE_CLASSES, (n,))
    ds = TensorDataset(x, y_coarse, y_fine)
    return DataLoader(ds, batch_size=4)


def test_e_train_one_phase_returns_log(tiny_train_loader):
    """train_one_phase returns a list of epoch dicts with epoch, loss, acc."""
    model = _MockModel(NUM_COARSE_CLASSES, NUM_FINE_CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    log = train_one_phase(model, tiny_train_loader, "coarse", optimizer, epochs=2)
    assert len(log) == 2
    for entry in log:
        assert "epoch" in entry
        assert "loss" in entry
        assert "acc" in entry
        assert entry["loss"] >= 0
        assert 0 <= entry["acc"] <= 1


def test_e_train_one_phase_fine_task(tiny_train_loader):
    """train_one_phase works for task_id 'fine'."""
    model = _MockModel(NUM_COARSE_CLASSES, NUM_FINE_CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    log = train_one_phase(model, tiny_train_loader, "fine", optimizer, epochs=1)
    assert len(log) == 1
    assert "acc" in log[0]


def test_e_train_one_phase_with_penalty(tiny_train_loader):
    """train_one_phase accepts penalty_fn and runs without error."""
    model = _MockModel(NUM_COARSE_CLASSES, NUM_FINE_CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    penalty_fn = lambda m: 0.0
    log = train_one_phase(
        model, tiny_train_loader, "coarse", optimizer, epochs=1, penalty_fn=penalty_fn
    )
    assert len(log) == 1


def test_e_save_checkpoint_creates_file():
    """save_checkpoint creates a .pth file in CHECKPOINT_DIR."""
    import training.trainer as trainer_mod
    with tempfile.TemporaryDirectory() as tmp:
        trainer_mod.CHECKPOINT_DIR = tmp
        model = _MockModel(2, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(model, optimizer, epoch=1, task_id="coarse")
        path = os.path.join(tmp, "checkpoint_coarse_1.pth")
        assert os.path.isfile(path)


def test_e_load_checkpoint_restores_state():
    """After save_checkpoint, load_checkpoint restores epoch and task_id."""
    import training.trainer as trainer_mod
    with tempfile.TemporaryDirectory() as tmp:
        trainer_mod.CHECKPOINT_DIR = tmp
        model = _MockModel(2, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(model, optimizer, epoch=1, task_id="coarse")
        model2 = _MockModel(2, 10)
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        loaded_epoch, loaded_task = load_checkpoint(model2, opt2, epoch=1, task_id="coarse")
        assert loaded_epoch == 1
        assert loaded_task == "coarse"


def test_e_load_checkpoint_missing_file_raises():
    """load_checkpoint raises FileNotFoundError when file does not exist."""
    import training.trainer as trainer_mod
    with tempfile.TemporaryDirectory() as tmp:
        trainer_mod.CHECKPOINT_DIR = tmp
        model = _MockModel(2, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with pytest.raises(FileNotFoundError, match="not found"):
            load_checkpoint(model, optimizer, epoch=99, task_id="coarse")
    
