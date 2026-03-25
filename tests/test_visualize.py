import torch

from evaluation.visualize import plot_tsne


def test_plot_tsne_saves_png(tmp_path):
    embeddings = torch.randn(64, 512)
    labels = torch.randint(0, 2, (64,))
    output_path = tmp_path / "tsne_plot.png"

    plot_tsne(
        embeddings=embeddings,
        labels=labels,
        title="t-SNE Smoke Test",
        save_path=str(output_path),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
