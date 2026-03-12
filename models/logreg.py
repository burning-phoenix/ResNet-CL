import torch
import torch.nn as nn
from config import FEATURE_DIM, NUM_COARSE_CLASSES, NUM_FINE_CLASSES, INPUT_CHANNELS, INPUT_SIZE

class MultiHeadLogReg(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        input_dim = INPUT_CHANNELS * INPUT_SIZE * INPUT_SIZE

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, FEATURE_DIM),
            nn.ReLU()
        )
        
        self.coarse_head = nn.Linear(FEATURE_DIM, NUM_COARSE_CLASSES)
        self.fine_head = nn.Linear(FEATURE_DIM, NUM_FINE_CLASSES)

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, 3, 32, 32)
            task_id: str, either "coarse" or "fine"
        Returns:
            logits: Tensor (B, 2) or (B, 10)
        """
        x = self.backbone(x)
        if task_id == "coarse":
            return self.coarse_head(x)
        if task_id == "fine":
            return self.fine_head(x)
        raise ValueError(f"Unsupported task_id '{task_id}'. Use 'coarse' or 'fine'.")

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
