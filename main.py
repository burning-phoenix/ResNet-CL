"""
Experiment driver for hierarchical continual learning on CIFAR-100 (vehicles_1 vs reptiles).

 instance-level train splits A/B, three trajectories
(Coarse->Fine, Fine->Coarse, Flat), regularization conditions (none, L2, EWC), lambda grids,
seeds, and both ResNet-18 and multinomial logistic regression architectures.
"""


from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterator

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
from models.logreg import MultiHeadLogReg
from models.resnet18 import MultiHeadResNet18
from training.ewc import compute_fisher, ewc_penalty, snapshot_backbone
from training.l2_cl import l2_cl_penalty
from training.trainer import save_checkpoint, train_one_phase

# Table 2 / Table 3 settings from the PDF
LAMBDA_EWC_GRID = (100, 400, 1000)
LAMBDA_L2_GRID = (0.01, 0.1, 1.0)
SEEDS = (42, 123, 456)

TRAJECTORY_COARSE_TO_FINE = "coarse_to_fine"
TRAJECTORY_FINE_TO_COARSE = "fine_to_coarse"
TRAJECTORY_FLAT = "flat"

CONDITION_UNREG = "unreg"
CONDITION_L2 = "l2"
CONDITION_EWC = "ewc"

ARCH_RESNET = "resnet18"
ARCH_LOGREG = "logreg"
ALL_ARCHITECTURES = (ARCH_RESNET, ARCH_LOGREG)


@dataclass(frozen=True)
class ExperimentConfig:
    trajectory: str
    condition: str
    architecture: str
    seed: int
    lambda_ewc: float | None
    lambda_l2: float | None


def build_model(architecture: str) -> torch.nn.Module:
    if architecture == ARCH_RESNET:
        return MultiHeadResNet18()
    if architecture == ARCH_LOGREG:
        return MultiHeadLogReg()
    raise ValueError(
        f"Unknown architecture '{architecture}'. "
        f"Use '{ARCH_RESNET}' or '{ARCH_LOGREG}'."
    )


def _sanitize_run_tag(tag: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag)


def _arch_tuple(args: argparse.Namespace) -> tuple[str, ...]:
    if getattr(args, "paper_matrix", False):
        return ALL_ARCHITECTURES
    if args.architecture == "both":
        return ALL_ARCHITECTURES
    return (args.architecture,)


def iter_experiment_configs(
    architectures: tuple[str, ...] = ALL_ARCHITECTURES,
) -> Iterator[ExperimentConfig]:
    """
    Full paper matrix:
      - Coarse->Fine: unreg + 3 L2 + 3 EWC
      - Fine->Coarse: unreg + 3 L2 + 3 EWC
      - Flat baseline: unreg only
    repeated for each seed and architecture.
    """
    for arch in architectures:
        for seed in SEEDS:
            for traj in (TRAJECTORY_COARSE_TO_FINE, TRAJECTORY_FINE_TO_COARSE):
                yield ExperimentConfig(traj, CONDITION_UNREG, arch, seed, None, None)

                for lam in LAMBDA_L2_GRID:
                    yield ExperimentConfig(traj, CONDITION_L2, arch, seed, None, float(lam))

                for lam in LAMBDA_EWC_GRID:
                    yield ExperimentConfig(traj, CONDITION_EWC, arch, seed, float(lam), None)

            yield ExperimentConfig(TRAJECTORY_FLAT, CONDITION_UNREG, arch, seed, None, None)


def _checkpoint_task_metrics(
    model: torch.nn.Module,
    test_loader,
    task_names: tuple[str, str],
) -> dict[str, dict[str, Any]]:
    """
    Evaluate both task heads at a checkpoint so logs match the paper more closely.
    """
    out: dict[str, dict[str, Any]] = {}
    for task in task_names:
        metrics = evaluate(model, test_loader, task)
        out[task] = {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "per_class_f1": metrics.get("per_class_f1"),
        }
    return out


def run_sequential_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """
    Task 1 on subset A, Task 2 on subset B; shared test set.
    Matches the sequential setup described in the PDF.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.architecture).to(device)
    optimizer = optim.SGD(model.parameters(), lr=DEFAULT_LR, momentum=DEFAULT_MOMENTUM)

    train_a, test_loader = get_cifar100_loaders(subset="A")
    train_b, _ = get_cifar100_loaders(subset="B")

    t1, t2 = (
        ("coarse", "fine")
        if cfg.trajectory == TRAJECTORY_COARSE_TO_FINE
        else ("fine", "coarse")
    )

    run_tag = _sanitize_run_tag(
        f"{cfg.architecture}_{cfg.trajectory}_{cfg.condition}_s{cfg.seed}"
        f"_ew{cfg.lambda_ewc}_l2{cfg.lambda_l2}"
    )

    R = np.zeros((2, 2), dtype=float)

    # -------------------------
    # Phase 1
    # -------------------------
    train_one_phase(model, train_a, t1, optimizer, DEFAULT_EPOCHS_PER_TASK)
    save_checkpoint(model, optimizer, DEFAULT_EPOCHS_PER_TASK, t1, run_tag=run_tag)

    after_task1_metrics = _checkpoint_task_metrics(model, test_loader, (t1, t2))
    R[0, 0] = float(after_task1_metrics[t1]["accuracy"])
    R[0, 1] = float(after_task1_metrics[t2]["accuracy"])

    theta_star = snapshot_backbone(model)

    # Fine->Coarse initialization from fine head weights, per PDF
    if cfg.trajectory == TRAJECTORY_FINE_TO_COARSE:
        model.init_coarse_from_fine()

    # -------------------------
    # Task 2 penalty
    # -------------------------
    penalty_fn = None
    if cfg.condition == CONDITION_EWC:
        fisher = compute_fisher(
            model,
            train_a,
            task_id=t1,
            n_samples=DEFAULT_FISHER_SAMPLES,
        )
        penalty_fn = lambda m: ewc_penalty(m, fisher, theta_star, cfg.lambda_ewc)
    elif cfg.condition == CONDITION_L2:
        penalty_fn = lambda m: l2_cl_penalty(m, theta_star, cfg.lambda_l2)

    # Freeze previous head, train next task
    model.freeze_head(t1)

    # -------------------------
    # Phase 2
    # -------------------------
    train_one_phase(
        model,
        train_b,
        t2,
        optimizer,
        DEFAULT_EPOCHS_PER_TASK,
        penalty_fn=penalty_fn,
    )
    save_checkpoint(model, optimizer, DEFAULT_EPOCHS_PER_TASK, t2, run_tag=run_tag)

    after_task2_metrics = _checkpoint_task_metrics(model, test_loader, (t1, t2))
    R[1, 0] = float(after_task2_metrics[t1]["accuracy"])
    R[1, 1] = float(after_task2_metrics[t2]["accuracy"])

    task2_k = NUM_FINE_CLASSES if t2 == "fine" else NUM_COARSE_CLASSES
    cl_metrics = compute_cl_metrics(R, task2_num_classes=task2_k)

    return {
        "trajectory": cfg.trajectory,
        "condition": cfg.condition,
        "architecture": cfg.architecture,
        "seed": cfg.seed,
        "lambda_ewc": cfg.lambda_ewc,
        "lambda_l2": cfg.lambda_l2,
        "task1": t1,
        "task2": t2,
        "epochs_per_task": DEFAULT_EPOCHS_PER_TASK,
        "R_matrix": R.tolist(),
        "metrics_after_task1": after_task1_metrics,
        "metrics_after_task2": after_task2_metrics,
        **cl_metrics,
    }


def run_flat_baseline(cfg: ExperimentConfig) -> dict[str, Any]:
    """
    Non-sequential upper bound: 10-way fine task on A union B.
    """
    assert cfg.trajectory == TRAJECTORY_FLAT and cfg.condition == CONDITION_UNREG

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.architecture).to(device)
    optimizer = optim.SGD(model.parameters(), lr=DEFAULT_LR, momentum=DEFAULT_MOMENTUM)

    train_all, test_loader = get_cifar100_loaders(subset="all")
    run_tag = _sanitize_run_tag(f"{cfg.architecture}_flat_unreg_s{cfg.seed}")

    train_one_phase(model, train_all, "fine", optimizer, DEFAULT_EPOCHS_PER_TASK)
    save_checkpoint(model, optimizer, DEFAULT_EPOCHS_PER_TASK, "fine", run_tag=run_tag)

    fine_metrics = evaluate(model, test_loader, "fine")

    return {
        "trajectory": TRAJECTORY_FLAT,
        "condition": CONDITION_UNREG,
        "architecture": cfg.architecture,
        "seed": cfg.seed,
        "task": "fine",
        "epochs_per_task": DEFAULT_EPOCHS_PER_TASK,
        "accuracy": fine_metrics.get("accuracy"),
        "macro_f1": fine_metrics.get("macro_f1"),
        "precision": fine_metrics.get("precision"),
        "recall": fine_metrics.get("recall"),
        "per_class_f1": fine_metrics.get("per_class_f1"),
    }


def run_one(cfg: ExperimentConfig) -> dict[str, Any]:
    if cfg.trajectory == TRAJECTORY_FLAT:
        return run_flat_baseline(cfg)
    return run_sequential_experiment(cfg)


def _log_path(cfg: ExperimentConfig) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    base = _sanitize_run_tag(
        f"{cfg.architecture}_{cfg.trajectory}_{cfg.condition}_s{cfg.seed}"
        f"_ew{cfg.lambda_ewc}_l2{cfg.lambda_l2}"
    )
    return os.path.join(LOG_DIR, f"{base}.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical CL experiments (ACL paper)")

    p.add_argument(
        "--list",
        action="store_true",
        help="Print all experiment configs and exit (no training).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Combine with --list or use alone to show count only.",
    )
    p.add_argument(
        "--run-all",
        action="store_true",
        help="Run the full matrix for the selected architecture(s).",
    )
    p.add_argument(
        "--paper-matrix",
        action="store_true",
        help="Run the full PDF experiment setup across BOTH architectures automatically.",
    )
    p.add_argument(
        "--architecture",
        choices=(ARCH_RESNET, ARCH_LOGREG, "both"),
        default="both",
        help="Architecture to use. Default is 'both' so run-all matches the paper.",
    )
    p.add_argument(
        "--trajectory",
        choices=(
            TRAJECTORY_COARSE_TO_FINE,
            TRAJECTORY_FINE_TO_COARSE,
            TRAJECTORY_FLAT,
        ),
        default=TRAJECTORY_COARSE_TO_FINE,
    )
    p.add_argument(
        "--condition",
        choices=(CONDITION_UNREG, CONDITION_L2, CONDITION_EWC),
        default=CONDITION_UNREG,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--lambda-ewc",
        type=float,
        default=None,
        help="EWC lambda (required if condition=ewc)",
    )
    p.add_argument(
        "--lambda-l2",
        type=float,
        default=None,
        help="L2 lambda (required if condition=l2)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    selected_arches = _arch_tuple(args)

    if args.list:
        cfgs = list(iter_experiment_configs(architectures=selected_arches))
        if args.dry_run:
            print(f"Total configurations: {len(cfgs)}")
            return
        for c in cfgs:
            print(c)
        return

    if args.run_all or args.paper_matrix:
        cfgs = list(iter_experiment_configs(architectures=selected_arches))
        print(json.dumps({
            "total_runs": len(cfgs),
            "architectures": list(selected_arches),
            "seeds": list(SEEDS),
        }, indent=2))

        for i, cfg in enumerate(cfgs, start=1):
            out = run_one(cfg)
            path = _log_path(cfg)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

            summary = {
                "idx": i,
                "of": len(cfgs),
                "saved": path,
                "trajectory": cfg.trajectory,
                "condition": cfg.condition,
                "architecture": cfg.architecture,
                "seed": cfg.seed,
            }

            if cfg.trajectory == TRAJECTORY_FLAT:
                summary["accuracy"] = out.get("accuracy")
                summary["macro_f1"] = out.get("macro_f1")
            else:
                summary["R_matrix"] = out.get("R_matrix")
                summary["forgetting"] = out.get("forgetting")
                summary["bwt"] = out.get("bwt")
                summary["fwt"] = out.get("fwt")

            print(json.dumps(summary))
        return

    if args.condition == CONDITION_EWC and args.lambda_ewc is None:
        raise SystemExit("--lambda-ewc is required when --condition ewc")
    if args.condition == CONDITION_L2 and args.lambda_l2 is None:
        raise SystemExit("--lambda-l2 is required when --condition l2")

    cfg = ExperimentConfig(
        trajectory=args.trajectory,
        condition=args.condition,
        architecture=selected_arches[0],
        seed=args.seed,
        lambda_ewc=args.lambda_ewc if args.condition == CONDITION_EWC else None,
        lambda_l2=args.lambda_l2 if args.condition == CONDITION_L2 else None,
    )

    out = run_one(cfg)
    path = _log_path(cfg)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print("Saved:", path)

# instead of runtrejectory.py

if __name__ == "__main__":
    main()