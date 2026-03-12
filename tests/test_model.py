import torch
import torch.nn.functional as F

from config import FEATURE_DIM, NUM_COARSE_CLASSES, NUM_FINE_CLASSES
from data.cifar100 import FINE_TO_COARSE
from models import MultiHeadResNet18


def test_forward_shapes_for_both_heads():
    model = MultiHeadResNet18()
    x = torch.randn(4, 3, 32, 32)

    coarse_logits = model(x, "coarse")
    fine_logits = model(x, "fine")

    assert coarse_logits.shape == (4, NUM_COARSE_CLASSES)
    assert fine_logits.shape == (4, NUM_FINE_CLASSES)


def test_get_features_shape():
    model = MultiHeadResNet18()
    x = torch.randn(4, 3, 32, 32)
    features = model.get_features(x)
    assert features.shape == (4, FEATURE_DIM)


def test_get_backbone_params_excludes_head_params():
    model = MultiHeadResNet18()

    backbone_param_ids = {id(param) for param in model.get_backbone_params()}
    coarse_head_param_ids = {id(param) for param in model.coarse_head.parameters()}
    fine_head_param_ids = {id(param) for param in model.fine_head.parameters()}

    assert backbone_param_ids
    assert backbone_param_ids.isdisjoint(coarse_head_param_ids)
    assert backbone_param_ids.isdisjoint(fine_head_param_ids)


def test_freeze_head_coarse_sets_requires_grad_false():
    model = MultiHeadResNet18()
    model.freeze_head("coarse")

    assert all(not param.requires_grad for param in model.coarse_head.parameters())
    assert all(param.requires_grad for param in model.fine_head.parameters())


def test_init_coarse_from_fine_aggregates_weights_and_biases():
    model = MultiHeadResNet18()

    with torch.no_grad():
        # Make every fine-class row unique and easy to average exactly.
        weight_template = torch.arange(
            NUM_FINE_CLASSES * FEATURE_DIM, dtype=torch.float32
        ).reshape(NUM_FINE_CLASSES, FEATURE_DIM)
        bias_template = torch.arange(NUM_FINE_CLASSES, dtype=torch.float32)
        model.fine_head.weight.copy_(weight_template)
        model.fine_head.bias.copy_(bias_template)

        # Overwrite coarse head to ensure values are replaced by init method.
        model.coarse_head.weight.fill_(-999.0)
        model.coarse_head.bias.fill_(-999.0)

    model.init_coarse_from_fine()

    for coarse_id in range(NUM_COARSE_CLASSES):
        child_fine_ids = [f for f, c in FINE_TO_COARSE.items() if c == coarse_id]
        expected_weight = model.fine_head.weight[child_fine_ids].mean(dim=0)
        expected_bias = model.fine_head.bias[child_fine_ids].mean()

        assert torch.allclose(model.coarse_head.weight[coarse_id], expected_weight)
        assert torch.isclose(model.coarse_head.bias[coarse_id], expected_bias)


def test_init_coarse_from_fine_beats_random_after_training():
    """
    Trains the fine head on a deterministic synthetic dataset, then verifies
    coarse-head initialization from fine weights performs better than random chance.
    """
    torch.manual_seed(0)
    model = MultiHeadResNet18()

    n_per_class = 16
    y_fine = torch.arange(NUM_FINE_CLASSES, dtype=torch.long).repeat_interleave(n_per_class)
    features = torch.zeros(y_fine.numel(), FEATURE_DIM, dtype=torch.float32)
    features[torch.arange(y_fine.numel()), y_fine] = 5.0
    y_coarse = torch.tensor(
        [FINE_TO_COARSE[int(fine_id)] for fine_id in y_fine.tolist()], dtype=torch.long
    )

    optimizer = torch.optim.SGD(model.fine_head.parameters(), lr=0.2)
    model.fine_head.train()
    for _ in range(200):
        logits = model.fine_head(features)
        loss = F.cross_entropy(logits, y_fine)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        fine_acc = (model.fine_head(features).argmax(dim=1) == y_fine).float().mean().item()
    assert fine_acc > 0.95

    model.init_coarse_from_fine()
    with torch.no_grad():
        coarse_preds = model.coarse_head(features).argmax(dim=1)
        coarse_acc = (coarse_preds == y_coarse).float().mean().item()

    random_chance = 1.0 / NUM_COARSE_CLASSES
    assert coarse_acc > random_chance
