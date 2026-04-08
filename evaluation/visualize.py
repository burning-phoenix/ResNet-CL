import os

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


def plot_tsne(embeddings, labels, title, save_path):
    """Run t-SNE on embeddings and save a scatter plot image."""
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels

    if len(embeddings_np) != len(labels_np):
        raise ValueError("embeddings and labels must have the same number of samples.")

    if len(embeddings_np) <= 30:
        raise ValueError("t-SNE with perplexity=30 needs more than 30 samples.")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto",
        method="exact",
    )
    embedding_2d = tsne.fit_transform(embeddings_np)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels_np, s=8, cmap="tab20")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
