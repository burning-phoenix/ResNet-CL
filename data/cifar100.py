from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2

from config import (
    DEFAULT_BATCH_SIZE,
    DATASETS_DIR,
    FINE_REMAP,
    FINE_TO_COARSE,
    NUM_WORKERS,
)


TARGET_FINE_INDICES = set(FINE_REMAP.keys())

class CIFAR100SubDataset(Dataset):
    def __init__(self, root, train=True, transform=None, subset = "all"):
        assert subset in ("A", "B", "all"), \
            f"subset must be 'A', 'B', or 'all', got '{subset}'"
        self.dataset = CIFAR100(root=root, train=train, download=True, transform=transform)
        # Filter the indices to only relevant classes

        if not train or subset == "all":
            self.indices = [i for i in range(len(self.dataset)) if self.dataset[i][1] in TARGET_FINE_INDICES]
        else:
            self.indices = []
            fine_to_indices = defaultdict(list)
            for i in range(len(self.dataset)):
                fine = self.dataset[i][1]
                if fine in TARGET_FINE_INDICES:
                    fine_to_indices[fine].append(i)
            
            for fine, indices in fine_to_indices.items():
                indices = sorted(indices)  # For deterministic splitting
                half = len(indices) // 2
                if subset == "A":
                    self.indices.extend(indices[:half])
                elif subset == "B":
                    self.indices.extend(indices[half:])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, fine = self.dataset[self.indices[idx]]
        fine = FINE_REMAP[fine]
        coarse = FINE_TO_COARSE[fine]
        y_coarse = torch.tensor(coarse, dtype=torch.long)
        y_fine = torch.tensor(fine, dtype=torch.long)
        return img, y_coarse, y_fine
    
def get_cifar100_loaders(batch_size=DEFAULT_BATCH_SIZE, augment=True, subset = "all"):
    """
    Returns:
        train_loader: yields (x, y_coarse, y_fine)
            x: Tensor (B, 3, 32, 32), normalized
            y_coarse: Tensor (B,), values in 0..19
            y_fine: Tensor (B,), values in 0..99
        test_loader: same format, no augmentation
    """
    # Augmentation code adapted from https://medium.com/@praburam_93885/a-custom-resnet-for-cifar-100-6f214c1ccae0
    # Also referred documentation for torchvision.transforms: https://pytorch.org/vision/stable/transforms.html
    if augment:

        train_transform = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32, padding=4),
            v2.ToTensor(),
            v2.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    else:

        train_transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_loader = DataLoader(
        CIFAR100SubDataset(root=DATASETS_DIR, train=True, transform=train_transform, subset=subset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        CIFAR100SubDataset(root=DATASETS_DIR, train=False, transform=test_transform, subset="all"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, test_loader