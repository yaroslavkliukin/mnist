import fire
import lightning.pytorch as pl
import torch
from configs.config import Params
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore

from data_load import load_test, load_train
from functions import convert_to_onnx, set_seed
from model import MNISTNet


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


def main(tracking_uri: str):
    with initialize(version_base="1.3", config_path="./../configs"):
        cfg = compose(
            config_name="config",
            overrides=[f"loggers.mlflow.tracking_uri={tracking_uri}"],
        )
    set_seed(cfg.train.seed)

    # Load data
    train_loader = load_train(cfg)
    val_loader = load_test(cfg)

    # Initiating model
    model = MNISTNet(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_classes)

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.loggers.mlflow.experiment_name,
            tracking_uri=cfg.loggers.mlflow.tracking_uri,
        )
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        logger=loggers,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    torch.save(model.state_dict(), cfg.data.model_path)

    if cfg.model.onnx_parameters.export_to_onnx:
        convert_to_onnx(model=model, conf=cfg.model.onnx_parameters)


if __name__ == "__main__":
    fire.Fire(main)
