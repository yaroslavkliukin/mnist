from dataclasses import dataclass


@dataclass
class OnnxParameters:
    export_to_onnx: bool
    onnx_path: str
    input_shape: list
    mlflow_onnx_export_path: str


@dataclass
class Model:
    name: str
    input_size: int
    hidden_size: int
    num_classes: int
    onnx_parameters: OnnxParameters


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
    inference_port: int
    inference_addr: str


@dataclass
class MLflow:
    experiment_name: str
    tracking_uri: str


@dataclass
class Loggers:
    mlflow: MLflow


@dataclass
class Params:
    model: Model
    data: Data
    train: Train
    infer: Infer
    loggers: Loggers
