import torch
import torch.nn as nn
from torchvision.models import resnet18

from config import FEATURE_DIM, NUM_COARSE_CLASSES, NUM_FINE_CLASSES
from data.cifar100 import FINE_TO_COARSE


class MultiHeadResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
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

    def init_coarse_from_fine(self) -> None:
        """Aggregates fine-head weights/biases into the coarse head."""
        children_by_coarse = {}
        for coarse_id in range(NUM_COARSE_CLASSES):
            children_by_coarse[coarse_id] = []

        for fine_id in range(NUM_FINE_CLASSES):
            coarse_id = FINE_TO_COARSE[fine_id]
            children_by_coarse[coarse_id].append(fine_id)

        with torch.no_grad():
            for coarse_id, fine_ids in children_by_coarse.items():
                self.coarse_head.weight[coarse_id] = self.fine_head.weight[fine_ids].mean(dim=0)
                self.coarse_head.bias[coarse_id] = self.fine_head.bias[fine_ids].mean()
