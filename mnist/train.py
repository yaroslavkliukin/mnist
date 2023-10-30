import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from data_load import load_train
from functions import set_seed, train
from model import NeuralNet


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader = load_train(cfg.train.batch_size)

    # Initiating model
    model = NeuralNet(
        cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_classes
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, device, cfg.train.num_epochs)

    torch.save(model.state_dict(), cfg.data.model_path)


if __name__ == "__main__":
    main()
