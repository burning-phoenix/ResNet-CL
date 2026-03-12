from data.cifar100 import get_cifar100_loaders
from evaluation.embeddings import extract_embeddings
from models import MultiHeadResNet18


def test_extract_embeddings_shapes_and_lengths():
    model = MultiHeadResNet18()
    train_loader, test_loader = get_cifar100_loaders(batch_size=32, augment=False)

    embeddings, labels_coarse, labels_fine = extract_embeddings(model, test_loader)

    # Total number of samples should match the dataset length.
    assert embeddings.shape[0] == len(test_loader.dataset)
    assert labels_coarse.shape[0] == len(test_loader.dataset)
    assert labels_fine.shape[0] == len(test_loader.dataset)

    # Feature dimension should be 512 from the ResNet-18 backbone.
    assert embeddings.shape[1] == 512

