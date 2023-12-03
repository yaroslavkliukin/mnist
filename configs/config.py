from dataclasses import dataclass


@dataclass
class Model:
    name: str
    input_size: int
    hidden_size: int
    num_classes: int


@dataclass
class Data:
    model_path: str
    preds_path: str


@dataclass
class Train:
    num_epochs: int
    batch_size: int
    learning_rate: float
    seed: int


@dataclass
class Infer:
    batch_size: int


@dataclass
class Params:
    model: Model
    data: Data
    train: Train
    infer: Infer
