import torch
import torchvision
import torchvision.transforms as transforms


def load_train(batch_size, save_path="./data"):
    train_dataset = torchvision.datasets.MNIST(
        root=save_path, train=True, transform=transforms.ToTensor(), download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    return train_loader


def load_test(batch_size, save_path="./data"):
    test_dataset = torchvision.datasets.MNIST(
        root=save_path, train=False, transform=transforms.ToTensor()
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return test_loader
