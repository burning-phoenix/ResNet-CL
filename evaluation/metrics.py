import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

def evaluate(model, test_loader, task_id):
    """Evaluates the model on the test set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        task_id (str): The task id.
    """
    model.eval()
    device = next(model.parameters()).device
    preds = []
    labels = []
    with torch.no_grad():
        for x, y_coarse, y_fine in tqdm(test_loader, desc="Evaluating"):
            x = x.to(device)
            y_coarse = y_coarse.to(device)
            y_fine = y_fine.to(device)
            y = y_coarse if task_id == "coarse" else y_fine
            logits = model(x, task_id)
            predicted = torch.argmax(logits, dim=1)
            
            
            preds.append(predicted)
            labels.append(y)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = (preds == labels).mean()
    
    PclassF1arr = f1_score(labels, preds, average=None)
    PclassF1 = {i: score for i, score in enumerate(PclassF1arr)}
    
    return {
        "accuracy": float(acc),
        "per_class_f1": PclassF1,
        "macro_f1": float(f1_score(labels, preds, average="macro"))
    }
    
def compute_cl_metrics(R):
    """Computes the CL metrics.

    Args:
        R (numpy.ndarray): The accuracy matrix.
    """
    bwt = R[1,0] - R[0,0]
    forgetting = R[0,0] - R[1,0]
    
    random_chance = 0.01
    fwt = R[0, 1] - random_chance
    
    return {
        "bwt": float(bwt),
        "forgetting": float(forgetting),
        "fwt": float(fwt)
    }