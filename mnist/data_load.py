import subprocess

import torch
import torchvision
import torchvision.transforms as transforms


def load_train(cfg, save_path="./data"):
    subprocess.run(
        [
            "dvc",
            "pull",
            f"{cfg.data.train_data_file}.dvc",
            f"{cfg.data.train_labels_file}.dvc",
        ]
    )
    subprocess.run(
        [
            "dvc",
            "pull",
            f"{cfg.data.train_data_file}.gz.dvc",
            f"{cfg.data.train_labels_file}.gz.dvc",
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root=save_path, train=True, transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )

    return train_loader


def load_test(cfg, save_path="./data"):
    subprocess.run(
        [
            "dvc",
            "pull",
            f"{cfg.data.test_data_file}.dvc",
            f"{cfg.data.test_labels_file}.dvc",
        ]
    )
    subprocess.run(
        [
            "dvc",
            "pull",
            f"{cfg.data.test_data_file}.gz.dvc",
            f"{cfg.data.test_labels_file}.gz.dvc",
        ]
    )
    test_dataset = torchvision.datasets.MNIST(
        root=save_path, train=False, transform=transforms.ToTensor()
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=cfg.train.batch_size, shuffle=False
    )

    return test_loader
