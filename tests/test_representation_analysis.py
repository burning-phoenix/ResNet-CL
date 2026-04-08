import json
import os

import pytest
import torch

from models.resnet18 import MultiHeadResNet18


def test_analyze_checkpoint_writes_json_and_figures(tmp_path, monkeypatch):
    import evaluation.representation_analysis as rep

    def fake_plot_tsne(embeddings, labels, title, save_path):
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(b"png")

    monkeypatch.setattr(rep, "plot_tsne", fake_plot_tsne)

    device = torch.device("cpu")
    model = MultiHeadResNet18().to(device)
    ckpt_path = tmp_path / "checkpoint_test.pth"
    torch.save(
        {
            "epoch": 1,
            "task_id": "coarse",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
        },
        ckpt_path,
    )

    fig_dir = tmp_path / "figures"
    summary = rep.analyze_checkpoint(
        checkpoint_path=str(ckpt_path),
        tag="y7_test",
        batch_size=32,
        subset="all",
        figure_dir=str(fig_dir),
        device=device,
    )

    metrics_path = fig_dir / "y7_test_cluster_metrics.json"
    assert metrics_path.is_file()
    with open(metrics_path, encoding="ascii") as f:
        loaded = json.load(f)
    assert loaded["tag"] == "y7_test"
    assert "silhouette_score" in loaded
    assert "davies_bouldin_index" in loaded
    assert summary["num_samples"] == loaded["num_samples"]

    assert (fig_dir / "y7_test_tsne_coarse.png").is_file()
    assert (fig_dir / "y7_test_tsne_fine.png").is_file()


def test_load_resnet_from_checkpoint_missing_file_raises():
    from evaluation.representation_analysis import load_resnet_from_checkpoint

    device = torch.device("cpu")
    with pytest.raises(FileNotFoundError):
        load_resnet_from_checkpoint("/nonexistent/path.pth", device)
