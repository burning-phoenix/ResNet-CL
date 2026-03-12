# Get labels for fine classes
from config import FINE_CLASSES
from torchvision.datasets import CIFAR100

# Get the y label ids for the fine classes
def get_labels():
    fine_labels = list(FINE_CLASSES.keys())
    cifar100 = CIFAR100(root='./datasets', train=True, download=True)
    
    # Map class name to id
    fine_label_ids = {label: cifar100.class_to_idx[label] for label in fine_labels}
    
    return fine_label_ids

if __name__ == "__main__":
    fine_label_ids = get_labels()
    print("Fine label ids:", fine_label_ids)