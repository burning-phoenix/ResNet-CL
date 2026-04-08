import argparse
import json
import os

import torch

from config import FIGURE_DIR
from data.cifar100 import get_cifar100_loaders
from evaluation.embeddings import compute_cluster_metrics, extract_embeddings
from evaluation.visualize import plot_tsne
from models.resnet18 import MultiHeadResNet18


def load_resnet_from_checkpoint(checkpoint_path, device):
    """Load MultiHeadResNet18 weights from a trainer checkpoint file."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = MultiHeadResNet18().to(device)
    try:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        saved = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model


def analyze_checkpoint(
    checkpoint_path,
    tag,
    batch_size=128,
    subset="all",
    figure_dir=None,
    device=None,
    ):
    """
    One checkpoint -> metrics JSON + two PNGs under figure_dir.

    Args:
        checkpoint_path: path to checkpoint_*.pth from save_checkpoint
        tag: short name used in filenames (e.g. coarse_to_fine_after_t1)
        batch_size: test loader batch size
        subset: same subset as training if you used A/B ('all', 'A', or 'B')
        figure_dir: where to write outputs (default: config FIGURE_DIR)
        device: torch device (default: cuda if available else cpu)
    """
    if figure_dir is None:
        figure_dir = FIGURE_DIR
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_resnet_from_checkpoint(checkpoint_path, device)
    _, test_loader = get_cifar100_loaders(
        batch_size=batch_size, augment=False, subset=subset
    )

    embeddings, labels_coarse, labels_fine = extract_embeddings(model, test_loader)
    metrics = compute_cluster_metrics(embeddings, labels_coarse)

    os.makedirs(figure_dir, exist_ok=True)

    coarse_png = os.path.join(figure_dir, f"{tag}_tsne_coarse.png")
    fine_png = os.path.join(figure_dir, f"{tag}_tsne_fine.png")
    metrics_json = os.path.join(figure_dir, f"{tag}_cluster_metrics.json")

    plot_tsne(
        embeddings,
        labels_coarse,
        title=f"{tag} — coarse labels",
        save_path=coarse_png,
    )
    plot_tsne(
        embeddings,
        labels_fine,
        title=f"{tag} — fine labels",
        save_path=fine_png,
    )

    summary = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "tag": tag,
        "num_samples": int(embeddings.shape[0]),
        "silhouette_score": metrics["silhouette_score"],
        "davies_bouldin_index": metrics["davies_bouldin_index"],
        "figures": {
            "tsne_coarse": os.path.abspath(coarse_png),
            "tsne_fine": os.path.abspath(fine_png),
        },
    }
    with open(metrics_json, "w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Y7: embeddings + cluster metrics + t-SNE from one checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="e.g. results/checkpoints/checkpoint_coarse_50.pth",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Prefix for output files, e.g. c2f_seed42_after_t1",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset", default="all", choices=["all", "A", "B"])
    parser.add_argument(
        "--figures-dir",
        default=FIGURE_DIR,
        help="Output folder for PNGs + JSON (default: FIGURE_DIR from config)",
    )
    args = parser.parse_args()

    analyze_checkpoint(
        checkpoint_path=args.checkpoint,
        tag=args.tag,
        batch_size=args.batch_size,
        subset=args.subset,
        figure_dir=args.figures_dir,
    )
    print("Done. Wrote figures and JSON under:", args.figures_dir)


if __name__ == "__main__":
    main()
