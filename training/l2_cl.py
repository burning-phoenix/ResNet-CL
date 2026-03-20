import torch


def l2_cl_penalty(
    model,
    theta_star: dict[str, torch.Tensor],
    lam: float,
) -> torch.Tensor:
    """Uniform quadratic penalty anchoring parameters to their Task-1 values

    (lamda / 2) * Sum_over-i (theta_i − theta*_i)^2

    Args:
        model:       Current model whose parameters have moved during T2.
        theta_star:  Backbone parameter snapshot from end of T1. Use ewc.snapshot_backbone() to create this.
        lam:         Regularization strength.

    Returns scalar tensor to add to the new-task loss.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)

    for name, param in model.named_parameters():
        if name in theta_star:
            old_value = theta_star[name]
            penalty += ((param - old_value) ** 2).sum()

    return (lam / 2) * penalty