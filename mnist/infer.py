import torch

from data_load import load_test
from functions import test
from model import NeuralNet


def main(
    batch_size=100,
    model_path="./models/model.pth",
    answers_path="./predictions/answers.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = 784
    hidden_size = 500
    num_classes = 10

    test_loader = load_test(batch_size)

    model = NeuralNet(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    test(model, test_loader, device, answers_path)


if __name__ == "__main__":
    main()
