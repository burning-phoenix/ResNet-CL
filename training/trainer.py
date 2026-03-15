import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from config import CHECKPOINT_DIR


def train_one_phase(model, loader, task_id, optimizer, epochs, penalty_fn=None):
    """Trains one phase of the model.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): The data loader.
        task_id (str): The task id.
        optimizer (torch.optim.Optimizer): The optimizer.
        epochs (int): The number of epochs to train for.
        penalty_fn (callable, optional): The penalty function. Defaults to None.
    """
    model.train()
    device = next(model.parameters()).device
    training_log = []
    for epoch in tqdm(range(epochs), desc=f"Training {task_id} phase"):
        total_correct = 0
        total_samples = 0
        running_loss = 0.0
        
        for x, y_coarse, y_fine in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            y_coarse = y_coarse.to(device)
            y_fine = y_fine.to(device)
            y = y_coarse if task_id == "coarse" else y_fine
            
            
            logits = model(x, task_id)
            loss = F.cross_entropy(logits, y)
            if penalty_fn is not None:
                loss += penalty_fn(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            total_samples += y.size(0)
            total_correct += (predicted == y).sum().item()
            
        epoch_loss = running_loss / len(loader)
        epoch_acc = total_correct / total_samples
        training_log.append({
            "epoch": epoch+1,
            "loss": epoch_loss,
            "acc": epoch_acc
        })
    return training_log


def save_checkpoint(model, optimizer, epoch, task_id):
    """Saves a checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The epoch number.
        task_id (str): The task id.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "task_id": task_id,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_{task_id}_{epoch}.pth"))
    
    
def load_checkpoint(model, optimizer, epoch, task_id):
    """Loads a checkpoint.

    Args:
        model (torch.nn.Module): The model to load.
        optimizer (torch.optim.Optimizer): The optimizer to load.
        epoch (int): The epoch number.
        task_id (str): The task id.
    """
    
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{task_id}_{epoch}.pth")
    if not os.path.exists(path) or not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file {path} not found")
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    return checkpoint["epoch"], checkpoint["task_id"]


