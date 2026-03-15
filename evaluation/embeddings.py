import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score, davies_bouldin_score


def extract_embeddings(model, test_loader: DataLoader):
    """
    Run the model on the given test loader and collect backbone features
    together with both levels of labels.

    Returns:
        embeddings: Tensor of shape (N, 512)
        labels_coarse: Tensor of shape (N,)
        labels_fine: Tensor of shape (N,)
    """
    device = next(model.parameters()).device
    model.eval()

    all_embeddings = []
    all_coarse = []
    all_fine = []

    with torch.no_grad():
        for x, y_coarse, y_fine in test_loader:
            x = x.to(device)
            features = model.get_features(x)

            all_embeddings.append(features.cpu())
            all_coarse.append(y_coarse.cpu())
            all_fine.append(y_fine.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels_coarse = torch.cat(all_coarse, dim=0)
    labels_fine = torch.cat(all_fine, dim=0)

    return embeddings, labels_coarse, labels_fine


def compute_cluster_metrics(embeddings, labels):
    """
    Compute clustering quality metrics from embeddings and labels.

    Args:
        embeddings: Tensor or array of shape (N, D)
        labels: Tensor or array of shape (N,)

    Returns:
        dict: {silhouette_score, davies_bouldin_index}
    """
    embeddings_np = np.array(embeddings.detach().cpu().tolist())
    labels_np = np.array(labels.detach().cpu().tolist())

    silhouette = float(silhouette_score(embeddings_np, labels_np))
    davies_bouldin = float(davies_bouldin_score(embeddings_np, labels_np))

    metrics = {
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
    }

    return metrics