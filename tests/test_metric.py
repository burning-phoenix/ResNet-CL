import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.metrics import compute_cl_metrics, evaluate
from config import NUM_COARSE_CLASSES, NUM_FINE_CLASSES


# ---- compute_cl_metrics (pure function) ----

def test_e_compute_cl_metrics_shape():
    """compute_cl_metrics returns dict with bwt, forgetting, fwt."""
    R = np.array([[0.8, 0.1], [0.7, 0.6]])
    out = compute_cl_metrics(R, task2_num_classes=10)
    assert "bwt" in out
    assert "forgetting" in out
    assert "fwt" in out


def test_e_compute_cl_metrics_bwt_forgetting():
    """bwt = R[1,0] - R[0,0], forgetting = R[0,0] - R[1,0]."""
    R = np.array([[0.8, 0.1], [0.7, 0.6]])
    out = compute_cl_metrics(R, task2_num_classes=10)
    assert out["bwt"] == pytest.approx(-0.1)
    assert out["forgetting"] == pytest.approx(0.1)


def test_e_compute_cl_metrics_fwt():
    """fwt = R[0,1] - 1/K (paper Table 5)."""
    R = np.array([[0.8, 0.15], [0.7, 0.6]])
    out = compute_cl_metrics(R, task2_num_classes=10)
    assert out["fwt"] == pytest.approx(0.15 - 0.1)


def test_e_compute_cl_metrics_numpy():
    """Accepts numpy array (as used in run_trajectory)."""
    R = np.array([[0.9, 0.02], [0.85, 0.8]])
    out = compute_cl_metrics(R, task2_num_classes=10)
    assert out["bwt"] == pytest.approx(-0.05)
    assert out["forgetting"] == pytest.approx(0.05)
    assert out["fwt"] == pytest.approx(0.02 - 0.1)


# ---- evaluate (needs model + loader) ----

class _MockModel(torch.nn.Module):

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
def tiny_eval_loader():
    """Small batch of (x, y_coarse, y_fine) for evaluation."""
    n = 8
    x = torch.randn(n, 3, 32, 32)
    y_coarse = torch.randint(0, NUM_COARSE_CLASSES, (n,))
    y_fine = torch.randint(0, NUM_FINE_CLASSES, (n,))
    ds = TensorDataset(x, y_coarse, y_fine)
    return DataLoader(ds, batch_size=4)


def test_e_evaluate_returns_dict(tiny_eval_loader):
    """evaluate returns accuracy, per_class_f1, macro_f1."""
    model = _MockModel(NUM_COARSE_CLASSES, NUM_FINE_CLASSES)
    result = evaluate(model, tiny_eval_loader, "coarse")
    assert "accuracy" in result
    assert "per_class_f1" in result
    assert "macro_f1" in result
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["macro_f1"] <= 1


def test_e_evaluate_fine_task(tiny_eval_loader):
    """evaluate with task_id 'fine' uses fine labels."""
    model = _MockModel(NUM_COARSE_CLASSES, NUM_FINE_CLASSES)
    result = evaluate(model, tiny_eval_loader, "fine")
    assert "accuracy" in result
    assert 1 <= len(result["per_class_f1"]) <= NUM_FINE_CLASSES
    assert all(0 <= v <= 1 for v in result["per_class_f1"].values())
