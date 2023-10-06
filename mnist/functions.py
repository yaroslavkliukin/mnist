import pandas as pd
import torch


def train(model, train_loader, criterion, optimizer, device, num_epochs=2):
    model.train()

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
                )


@torch.no_grad()
def test(model, test_loader, device, answers_path):
    model.eval()

    answers = list()
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        answer = [x.item() for x in predicted]
        answers.extend(answer)

        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    answers_frame = pd.DataFrame(data={"labels": answers})
    answers_frame.to_csv(answers_path)

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network on the 10000 test images: {acc} %")
