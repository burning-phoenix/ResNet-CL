import torch
import torch.nn as nn
from torchvision.models import resnet18

from config import FEATURE_DIM, NUM_COARSE_CLASSES, NUM_FINE_CLASSES


class MultiHeadResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(pretrained=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.coarse_head = nn.Linear(FEATURE_DIM, NUM_COARSE_CLASSES)
        self.fine_head = nn.Linear(FEATURE_DIM, NUM_FINE_CLASSES)

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, 3, 32, 32)
            task_id: str, either "coarse" or "fine"
        Returns:
            logits: Tensor (B, 20) or (B, 100)
        """
        features = self.get_features(x)
        if task_id == "coarse":
            return self.coarse_head(features)
        if task_id == "fine":
            return self.fine_head(features)
        raise ValueError(f"Unsupported task_id '{task_id}'. Use 'coarse' or 'fine'.")

    def get_backbone_params(self):
        """Returns iterator over shared backbone parameters only."""
        return self.backbone.parameters()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 512-dim feature vectors (B, 512). For embedding extraction."""
        return self.backbone(x)

    def freeze_head(self, task_id: str) -> None:
        """Sets requires_grad=False for all parameters in the specified head."""
        if task_id == "coarse":
            head = self.coarse_head
        elif task_id == "fine":
            head = self.fine_head
        else:
            raise ValueError(f"Unsupported task_id '{task_id}'. Use 'coarse' or 'fine'.")

        for param in head.parameters():
            param.requires_grad = False
