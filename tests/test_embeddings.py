import torch

from data.cifar100 import get_cifar100_loaders
from evaluation.embeddings import extract_embeddings, compute_cluster_metrics
from models import MultiHeadResNet18


def test_extract_embeddings_shapes_and_lengths():
    model = MultiHeadResNet18()
    train_loader, test_loader = get_cifar100_loaders(batch_size=32, augment=False)

    embeddings, labels_coarse, labels_fine = extract_embeddings(model, test_loader)

    # Total number of samples should match the dataset length.
    assert embeddings.shape[0] == len(test_loader.dataset)
    assert labels_coarse.shape[0] == len(test_loader.dataset)
    assert labels_fine.shape[0] == len(test_loader.dataset)

    # Feature dimension should be 512 from the ResNet-18 backbone.
    assert embeddings.shape[1] == 512


def test_compute_cluster_metrics_returns_expected_keys():
    # Build two simple, well-separated clusters in 512 dimensions.
    cluster_a = torch.zeros((8, 512), dtype=torch.float32)
    cluster_b = torch.ones((8, 512), dtype=torch.float32) * 10.0

    embeddings = torch.cat([cluster_a, cluster_b], dim=0)
    labels = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)

    metrics = compute_cluster_metrics(embeddings, labels)

    assert "silhouette_score" in metrics
    assert "davies_bouldin_index" in metrics
    assert isinstance(metrics["silhouette_score"], float)
    assert isinstance(metrics["davies_bouldin_index"], float)


def test_compute_cluster_metrics_reflects_good_clusters():
    # Same simple setup: clusters are very far apart.
    cluster_a = torch.zeros((10, 512), dtype=torch.float32)
    cluster_b = torch.ones((10, 512), dtype=torch.float32) * 20.0

    embeddings = torch.cat([cluster_a, cluster_b], dim=0)
    labels = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)

    metrics = compute_cluster_metrics(embeddings, labels)

    # For clearly separated clusters, silhouette should be high
    # and Davies-Bouldin should be low.
    assert metrics["silhouette_score"] > 0.9
    assert metrics["davies_bouldin_index"] < 0.5

