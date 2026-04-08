import torch
import torch.nn.functional as F


def compute_fisher(
    model,
    data_loader,
    task_id: str,
    n_samples: int,
) -> dict[str, torch.Tensor]:
    """Compute the exact diagonal of the true Fisher over backbone parameters.

    For each sample x, iterates over every output class k, computes the gradient of -log p(k|x), squares it, and weights by p(k|x). 
    Processes samples one at a time as squaring batch-aggregated gradients gives an incorrect quantity.

    Args:
        model:       Multi-head model with forward(x, task_id).
        data_loader: Yields (x, y_coarse, y_fine) batches.
        task_id:     coarse or fine
        n_samples:   Number of samples to average over.

    Returns a dictionary mapping backbone parameter names to tensors of per-element Fisher values (same shape as the parameters).
    """
    device = next(model.parameters()).device
    backbone_names = _backbone_param_names(model)

    fisher = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if name in backbone_names
    }

    model.eval()
    samples_processed = 0

    for batch_x, batch_y_coarse, batch_y_fine in data_loader:
        if samples_processed >= n_samples:
            break

        batch_x = batch_x.to(device)

        for i in range(batch_x.size(0)):
            if samples_processed >= n_samples:
                break

            single_x = batch_x[i].unsqueeze(0)
            logits = model(single_x, task_id)
            probs = F.softmax(logits, dim=1).squeeze(0)
            log_probs = F.log_softmax(logits, dim=1).squeeze(0)
            num_classes = logits.size(1)

            for k in range(num_classes):
                nll_k = -log_probs[k]

                model.zero_grad()
                nll_k.backward(retain_graph=(k < num_classes - 1))

                prob_k = probs[k].item()
                for name, param in model.named_parameters():
                    if name in backbone_names and param.grad is not None:
                        fisher[name] += prob_k * (param.grad.detach() ** 2)

            samples_processed += 1

    if samples_processed == 0:
        raise RuntimeError("No samples processed, data_loader empty.")

    for name in fisher:
        fisher[name] /= samples_processed

    return fisher



def ewc_penalty(
    model,
    fisher_dict: dict[str, torch.Tensor],
    theta_star: dict[str, torch.Tensor],
    lam: float,
) -> torch.Tensor:
    """Quadratic penalty weighted by Fisher importance.

    (Lambda / 2) * Sum_over_i  F_i · (theta_i − theta*_i)^2

    High-Fisher parameters are anchored to their Task-1 values, low-Fisher parameters are free to adapt to the new task.

    Args:
        model:       Current model whose parameters have moved during T2.
        fisher_dict: Per-parameter Fisher values from compute_fisher().
        theta_star:  Backbone parameter snapshot from end of T1.
        lam:         Regularization strength.

    Returns a scalar tensor to add to the new-task loss.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if name in fisher_dict:
            importance = fisher_dict[name]
            old_value = theta_star[name]
            penalty += (importance * (param - old_value) ** 2).sum()

    return (lam / 2) * penalty



def snapshot_backbone(model) -> dict[str, torch.Tensor]:
    """Save a detached copy of backbone parameters (theta*) at end of T1.

    Returns a dict with the same keys as compute_fisher() output so they zip together cleanly in ewc_penalty().
    """
    backbone_names = _backbone_param_names(model)

    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name in backbone_names
    }



def diagnose_fisher(fisher_dict: dict[str, torch.Tensor]) -> dict:
    """Report Fisher value distribution for sanity checking.

    Returns a dictionary with mean, max, near_zero_fraction, total_params,has_nan, all_zero.
    """
    all_values = torch.cat([f.flatten() for f in fisher_dict.values()])

    total_params = all_values.numel()
    mean_val = all_values.mean().item()
    max_val = all_values.max().item()
    has_nan = torch.isnan(all_values).any().item()
    all_zero = (max_val == 0.0)
    near_zero_count = (all_values < 1e-8).sum().item()
    near_zero_fraction = near_zero_count / total_params

    return {
        "mean": mean_val,
        "max": max_val,
        "near_zero_fraction": near_zero_fraction,
        "total_params": total_params,
        "has_nan": has_nan,
        "all_zero": all_zero,
    }



def _backbone_param_names(model) -> set:
    """Identify shared backbone parameters by excluding classification heads.

    Head parameters contain 'head_coarse' or 'head_fine' in their name
    """
    return {
        name for name, _ in model.named_parameters()
        if "coarse_head" not in name and "fine_head" not in name
    }