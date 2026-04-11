import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os

import numpy as np
import torch
import torch.optim as optim

from config import (
    DEFAULT_EPOCHS_PER_TASK,
    DEFAULT_FISHER_SAMPLES,
    DEFAULT_LR,
    DEFAULT_MOMENTUM,
    LOG_DIR,
    NUM_COARSE_CLASSES,
    NUM_FINE_CLASSES,
)
from data.cifar100 import get_cifar100_loaders
from evaluation.metrics import compute_cl_metrics, evaluate
from models.resnet18 import MultiHeadResNet18
from training.ewc import compute_fisher, ewc_penalty, snapshot_backbone
from training.l2_cl import l2_cl_penalty
from training.trainer import save_checkpoint, train_one_phase


def experiment(trajectory, condition, lambda_ewc, lambda_l2, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_a, test_loader = get_cifar100_loaders(subset="A")
    train_b, _ = get_cifar100_loaders(subset="B")

    model = MultiHeadResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=DEFAULT_LR, momentum=DEFAULT_MOMENTUM)

    t1, t2 = ("coarse", "fine") if trajectory == "coarse_to_fine" else ("fine", "coarse")

    R = np.zeros((2, 2))
    train_one_phase(model, train_a, t1, optimizer, DEFAULT_EPOCHS_PER_TASK)
    save_checkpoint(model, optimizer, DEFAULT_EPOCHS_PER_TASK, t1)

    R[0, 0] = evaluate(model, test_loader, t1)["accuracy"]
    R[0, 1] = evaluate(model, test_loader, t2)["accuracy"]

    theta_star = snapshot_backbone(model)
    if trajectory == "fine_to_coarse":
        model.init_coarse_from_fine()

    penalty_fn = None
    if condition == "ewc":
        fisher_dict = compute_fisher(
            model, train_a, task_id=t1, n_samples=DEFAULT_FISHER_SAMPLES
        )
        penalty_fn = lambda m: ewc_penalty(m, fisher_dict, theta_star, lambda_ewc)
    elif condition == "l2":
        penalty_fn = lambda m: l2_cl_penalty(m, theta_star, lambda_l2)

    model.freeze_head(t1)

    train_one_phase(
        model, train_b, t2, optimizer, DEFAULT_EPOCHS_PER_TASK, penalty_fn
    )
    save_checkpoint(model, optimizer, DEFAULT_EPOCHS_PER_TASK, t2)

    R[1, 0] = evaluate(model, test_loader, t1)["accuracy"]
    R[1, 1] = evaluate(model, test_loader, t2)["accuracy"]

    task2_k = NUM_FINE_CLASSES if t2 == "fine" else NUM_COARSE_CLASSES
    cl_metrics = compute_cl_metrics(R, task2_num_classes=task2_k)
    output = {
        "trajectory": trajectory,
        "condition": condition,
        "lambda_ewc": lambda_ewc,
        "lambda_l2": lambda_l2,
        "seed": seed,
        "R_matrix": R.tolist(),
        "bwt": cl_metrics["bwt"],
        "forgetting": cl_metrics["forgetting"],
        "fwt": cl_metrics["fwt"],
    }

    log_file = os.path.join(
        LOG_DIR,
        f"trajectory_{trajectory}_condition_{condition}_lambda_ewc_{lambda_ewc}_lambda_l2_{lambda_l2}_seed_{seed}.json",
    )
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(output)



