import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    dataset_name="MNIST",
    batch_size=128,
    data_dir="./data"
):
    """
    Возвращает train/test DataLoader.
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        input_dim = 28 * 28
        num_classes = 10

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        input_dim = 32 * 32 * 3
        num_classes = 10

    else:
        raise ValueError("Неизвестный датасет")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, input_dim, num_classes
