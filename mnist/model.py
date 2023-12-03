import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out


class MNISTNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = NeuralNet(input_size, hidden_size, num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.acc_metric = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        features, labels = batch
        features = features.reshape(-1, 28 * 28)
        y_preds = self.model(features)
        loss = self.ce_loss(y_preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch):
        features, labels = batch
        features = features.reshape(-1, 28 * 28)
        y_preds = self.model(features)
        loss = self.ce_loss(y_preds, labels)
        val_acc = self.acc_metric(y_preds, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_acc", val_acc, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss, "val_acc": val_acc}

    def test_step(self, batch):
        features, labels = batch
        features = features.reshape(-1, 28 * 28)
        y_preds = self.model(features)
        loss = self.ce_loss(y_preds, labels)
        val_acc = self.acc_metric(y_preds, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_acc", val_acc, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss, "val_acc": val_acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return [optimizer]
