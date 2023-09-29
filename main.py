import torch
import torch.nn as nn

from model import NeuralNet
from data_load import data_load
from functions import train, test

def main(num_epochs=2, batch_size=100, learning_rate=1e-3):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters 
    input_size = 784 # 28x28
    hidden_size = 500 
    num_classes = 10

    # Load data
    train_loader, test_loader = data_load(batch_size)

    # Initiating model
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    train(model, train_loader, criterion, optimizer, device, num_epochs)

    # Test the model
    test(model, test_loader, device)

if __name__=="__main__":
    main()
