import torch
import numpy as np
import os
import json
import torch.optim as optim
from config import DEFAULT_LR, DEFAULT_MOMENTUM, DEFAULT_EPOCHS_PER_TASK, DEFAULT_FISHER_SAMPLES, LOG_DIR
from data.cifar100 import get_cifar100_loaders #
from models.resnet18 import MultiHeadResNet18 #
from training.trainer import train_one_phase, save_checkpoint #
from training.ewc import compute_fisher, ewc_penalty #
from training.l2_cl import l2_cl_penalty #
from evaluation.metrics import evaluate, compute_cl_metrics #

def experiment(trajectory, condition, lambda_ewc, lambda_l2, seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = get_cifar100_loaders()
    
    model = MultiHeadResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=DEFAULT_LR, momentum=DEFAULT_MOMENTUM)
    
    t1,t2 = ("coarse", "fine") if trajectory == "coarse_to_fine" else ("fine", "coarse")
    R = np.zeros((2,2))
    train_one_phase(model, train_loader, t1, optimizer, DEFAULT_EPOCHS_PER_TASK)
    save_checkpoint(model, optimizer, DEFAULT_EPOCHS_PER_TASK, t1)
    
    
    
    R[0,0] = evaluate(model, test_loader, t1)["accuracy"]
    R[0,1] = evaluate(model, test_loader, t2)["accuracy"]
    
    Tstar = {n: p.clone().detach() for n, p in model.get_backbone_params()}    
    if trajectory == "fine_to_coarse":
        model.init_coarse_from_fine()
    
    
    penalty_fn = None
    if condition == "ewc":
        fisher_dict = compute_fisher(model, train_loader, DEFAULT_FISHER_SAMPLES) # No variant since removed as per Teams message
        penalty_fn = lambda m: ewc_penalty(m, fisher_dict, Tstar, lambda_ewc)
    elif condition == "l2":
        penalty_fn = lambda m: l2_cl_penalty(m, Tstar, lambda_l2)
        
    model.freeze_head(t1)
    
    
    train_one_phase(model, train_loader, t2, optimizer, DEFAULT_EPOCHS_PER_TASK, penalty_fn)
    save_checkpoint(model, optimizer, DEFAULT_EPOCHS_PER_TASK, t2)

    R[1,0] = evaluate(model, test_loader, t1)["accuracy"]
    R[1,1] = evaluate(model, test_loader, t2)["accuracy"]
    
    cl_metrics = compute_cl_metrics(R)
    Output = {
        "trajectory": trajectory,
        "condition": condition,
        "lambda_ewc": lambda_ewc,
        "lambda_l2": lambda_l2,
        "seed": seed,
        "R_matrix": R.tolist(),
        "bwt": cl_metrics["bwt"],
        "forgetting": cl_metrics["forgetting"],
        "fwt": cl_metrics["fwt"]
    }
    
    logFile = os.path.join(LOG_DIR, f"trajectory_{trajectory}_condition_{condition}_lambda_ewc_{lambda_ewc}_lambda_l2_{lambda_l2}_seed_{seed}.json")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(logFile, "w") as f:
        json.dump(Output, f)
        