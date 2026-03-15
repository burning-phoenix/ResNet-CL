import torch

from config import FEATURE_DIM, NUM_COARSE_CLASSES, NUM_FINE_CLASSES
from models import MultiHeadResNet18
from models import MultiHeadLogReg


def test_resnet_forward_shapes_for_both_heads():
    model = MultiHeadResNet18()
    x = torch.randn(4, 3, 32, 32)

    coarse_logits = model(x, "coarse")
    fine_logits = model(x, "fine")

    assert coarse_logits.shape == (4, NUM_COARSE_CLASSES)
    assert fine_logits.shape == (4, NUM_FINE_CLASSES)


def test_resnet_get_features_shape():
    model = MultiHeadResNet18()
    x = torch.randn(4, 3, 32, 32)
    features = model.get_features(x)
    assert features.shape == (4, FEATURE_DIM)


def test_resnet_get_backbone_params_excludes_head_params():
    model = MultiHeadResNet18()

    backbone_param_ids = {id(param) for param in model.get_backbone_params()}
    coarse_head_param_ids = {id(param) for param in model.coarse_head.parameters()}
    fine_head_param_ids = {id(param) for param in model.fine_head.parameters()}

    assert backbone_param_ids
    assert backbone_param_ids.isdisjoint(coarse_head_param_ids)
    assert backbone_param_ids.isdisjoint(fine_head_param_ids)


def test_resnet_freeze_head_coarse_sets_requires_grad_false():
    model = MultiHeadResNet18()
    model.freeze_head("coarse")

    assert all(not param.requires_grad for param in model.coarse_head.parameters())
    assert all(param.requires_grad for param in model.fine_head.parameters())

def test_logreg_forward_shapes_for_both_heads():
    model = MultiHeadLogReg()
    x = torch.randn(4, 3, 32, 32)

    coarse_logits = model(x, "coarse")
    fine_logits = model(x, "fine")

    assert coarse_logits.shape == (4, NUM_COARSE_CLASSES)
    assert fine_logits.shape == (4, NUM_FINE_CLASSES)


def test_logreg_get_features_shape():
    model = MultiHeadLogReg()
    x = torch.randn(4, 3, 32, 32)
    features = model.get_features(x)
    assert features.shape == (4, FEATURE_DIM)


def test_logreg_freeze_head_coarse_sets_requires_grad_false():
    model = MultiHeadLogReg()
    model.freeze_head("coarse")

    assert all(not param.requires_grad for param in model.coarse_head.parameters())
    assert all(param.requires_grad for param in model.fine_head.parameters())
