import torch
import torch.nn as nn

from data_load import load_train
from functions import train
from model import NeuralNet


def main(
    num_epochs=2, batch_size=100, learning_rate=1e-3, save_path="./models/model.pth"
):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper-parameters
    input_size = 784  # 28x28
    hidden_size = 500
    num_classes = 10

    # Load data
    train_loader = load_train(batch_size)

    # Initiating model
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, device, num_epochs)

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
