import hydra
import lightning.pytorch as pl
import torch
from configs.config import Params
from hydra.core.config_store import ConfigStore

from data_load import load_test, load_train
from functions import set_seed
from model import MNISTNet


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="./../configs", config_name="config", version_base="1.3")
def main(cfg: Params):
    set_seed(cfg.train.seed)

    # Load data
    train_loader = load_train(cfg.train.batch_size)
    val_loader = load_test(cfg.infer.batch_size)

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


if __name__ == "__main__":
    main()
