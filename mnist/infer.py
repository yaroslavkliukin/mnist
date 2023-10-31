import hydra
import torch
from hydra.core.config_store import ConfigStore

from conf.config import Params
from data_load import load_test
from functions import test
from model import NeuralNet


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = load_test(cfg.infer.batch_size)

    model = NeuralNet(cfg.model.input_size, cfg.model.hidden_size, cfg.model.num_classes)
    model.load_state_dict(torch.load(cfg.data.model_path, map_location=device))
    model.to(device)

    test(model, test_loader, device, cfg.data.preds_path)


if __name__ == "__main__":
    main()
