import hydra
import lightning.pytorch as pl
import torch
from configs.config import Params
from hydra.core.config_store import ConfigStore

from data_load import load_test
from model import MNISTNet


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="./../configs", config_name="config", version_base="1.3")
def main(cfg: Params):
    test_loader = load_test(cfg.infer.batch_size)

    model = MNISTNet(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_classes)
    model.load_state_dict(torch.load(cfg.data.model_path))

    trainer = pl.Trainer()
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
