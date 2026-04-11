"""
Tests for training/ewc.py and training/l2_cl.py.

Run from project root: python -m pytest tests/test_ewc_l2.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.ewc import compute_fisher, ewc_penalty, snapshot_backbone
from training.l2_cl import l2_cl_penalty


class AnalyticBinaryModel(nn.Module):
    """
    Analytic model whose Fisher is known in closed form

    x -> backbone (1->1, no bias) -> h -> coarse_head [[+1],[-1]]

    Logits are [w*x, -w*x].  Exact diagonal Fisher for the backbone
    weight given a single input x is:

        F = 4 * x^2 * p * (1 - p),   p = sigmoid(2*w*x)
    """

    def __init__(self, backbone_weight: float = 0.7):
        super().__init__()
        self.backbone = nn.Linear(1, 1, bias=False)
        self.coarse_head = nn.Linear(1, 2, bias=False)
        self.fine_head = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.backbone.weight.fill_(backbone_weight)
            self.coarse_head.weight.copy_(torch.tensor([[1.0], [-1.0]]))
            self.fine_head.weight.copy_(torch.tensor([[0.5], [-0.5]]))

    def forward(self, x, task_id: str):
        h = self.backbone(x)
        if task_id == "coarse":
            return self.coarse_head(h)
        if task_id == "fine":
            return self.fine_head(h)
        raise ValueError(task_id)


def _expected_fisher(model, x_val):
    w = model.backbone.weight.detach().item()
    p = torch.sigmoid(torch.tensor(2.0 * w * x_val)).item()
    return 4.0 * (x_val ** 2) * p * (1.0 - p)


def _triplet_loader(x, batch_size):
    """compute_fisher expects (x, y_coarse, y_fine) tuples."""
    n = x.shape[0]
    dummy = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(x, dummy, dummy),
                      batch_size=batch_size, shuffle=False)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 3)       # weight (3,2), bias (3,)
        self.coarse_head = nn.Linear(3, 2)
        self.fine_head = nn.Linear(3, 4)

    def forward(self, x, task_id):
        h = self.backbone(x)
        if task_id == "coarse":
            return self.coarse_head(h)
        return self.fine_head(h)



def test_fisher_matches_closed_form_single_sample():
    """Verify against an analytic ground truth, catches wrong formula, missing probability weighting, or forgetting to square gradients."""
    model = AnalyticBinaryModel(backbone_weight=0.7)
    loader = _triplet_loader(torch.tensor([[2.0]]), batch_size=1)

    fisher = compute_fisher(model, loader, "coarse", n_samples=1)

    expected = torch.tensor([[_expected_fisher(model, 2.0)]])
    torch.testing.assert_close(fisher["backbone.weight"], expected,
                               atol=1e-6, rtol=1e-5)


def test_fisher_per_sample_not_per_batch():
    """Two samples in 1 batch.  Squaring the batch-aggregated gradient introduces cross-terms and gives incorrect values."""
    model = AnalyticBinaryModel(backbone_weight=0.4)
    x = torch.tensor([[1.0], [3.0]])
    loader = _triplet_loader(x, batch_size=2)

    fisher = compute_fisher(model, loader, "coarse", n_samples=2)

    per_sample_avg = (_expected_fisher(model, 1.0)
                      + _expected_fisher(model, 3.0)) / 2.0
    expected = torch.tensor([[per_sample_avg]])
    torch.testing.assert_close(fisher["backbone.weight"], expected,
                               atol=1e-6, rtol=1e-5)


def test_fisher_n_samples_is_hard_cutoff_mid_batch():
    model = AnalyticBinaryModel(backbone_weight=0.5)
    x = torch.tensor([[1.0], [2.0], [10.0]])
    loader = _triplet_loader(x, batch_size=3)

    fisher = compute_fisher(model, loader, "coarse", n_samples=2)

    # Only first two samples contribute
    per_sample_avg = (_expected_fisher(model, 1.0)
                      + _expected_fisher(model, 2.0)) / 2.0
    expected = torch.tensor([[per_sample_avg]])
    torch.testing.assert_close(fisher["backbone.weight"], expected,
                               atol=1e-6, rtol=1e-5)


def test_fisher_empty_loader_raises():
    model = AnalyticBinaryModel()
    loader = _triplet_loader(torch.empty((0, 1)), batch_size=1)
    with pytest.raises(RuntimeError, match="No samples processed"):
        compute_fisher(model, loader, "coarse", n_samples=1)



def test_ewc_penalty_hand_computed():
    """backbone.weight (3x2) shifted +1, Fisher=2 everywhere. backbone.bias (3,) shifted -2, Fisher=3 everywhere.

    unscaled = 2*(1^2)*6  +  3*(2^2)*3  = 12 + 36 = 48
    With lam=1.7:  (1.7/2)*48 = 40.8
    """
    torch.manual_seed(0)
    model = TinyModel()
    theta_star = snapshot_backbone(model)
    params = dict(model.named_parameters())

    fisher_dict = {
        "backbone.weight": torch.full_like(params["backbone.weight"], 2.0),
        "backbone.bias":   torch.full_like(params["backbone.bias"], 3.0),
    }

    with torch.no_grad():
        params["backbone.weight"].add_(1.0)
        params["backbone.bias"].sub_(2.0)
        params["coarse_head.weight"].add_(999.0)  # must be ignored

    penalty = ewc_penalty(model, fisher_dict, theta_star, lam=1.7)

    expected_unscaled = (
        (fisher_dict["backbone.weight"]
         * (params["backbone.weight"] - theta_star["backbone.weight"]) ** 2).sum()
        + (fisher_dict["backbone.bias"]
           * (params["backbone.bias"] - theta_star["backbone.bias"]) ** 2).sum()
    )
    expected = (1.7 / 2.0) * expected_unscaled
    torch.testing.assert_close(penalty, expected)



def test_l2_equals_ewc_with_uniform_fisher():
    torch.manual_seed(7)
    model = TinyModel()
    snap = snapshot_backbone(model)
    with torch.no_grad():
        model.backbone.weight.add_(torch.randn_like(model.backbone.weight))
        model.backbone.bias.add_(torch.randn_like(model.backbone.bias))

    fisher_ones = {n: torch.ones_like(snap[n]) for n in snap}
    lam = 3.0

    ewc_val = ewc_penalty(model, fisher_ones, snap, lam)
    l2_val = l2_cl_penalty(model, snap, lam)
    torch.testing.assert_close(ewc_val, l2_val)



def test_ewc_penalty_is_differentiable():
    model = TinyModel()
    snap = snapshot_backbone(model)
    fisher = {n: torch.ones_like(snap[n]) for n in snap}
    with torch.no_grad():
        model.backbone.weight.add_(0.1)

    penalty = ewc_penalty(model, fisher, snap, lam=1.0)
    penalty.backward()
    assert model.backbone.weight.grad is not None
    assert (model.backbone.weight.grad != 0).any()


def test_l2_penalty_is_differentiable():
    model = TinyModel()
    snap = snapshot_backbone(model)
    with torch.no_grad():
        model.backbone.weight.add_(0.1)

    penalty = l2_cl_penalty(model, snap, lam=1.0)
    penalty.backward()
    assert model.backbone.weight.grad is not None
    assert (model.backbone.weight.grad != 0).any()