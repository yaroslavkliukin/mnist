import subprocess
from pathlib import Path

import fire
from configs.config import Params
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore


def run_server(port):
    command = (
        f"mlflow models serve -p {port} -m ./models/onnx_model --env-manager=local".split(
            " "
        )
    )
    proc = subprocess.run(command)
    return proc


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


def main(port):
    # get model from gdrive
    with initialize(version_base="1.3", config_path="./../configs"):
        cfg = compose(config_name="config", overrides=[f"infer.inference_port={port}"])
    file = Path(cfg.model.onnx_parameters.mlflow_onnx_export_path)
    if not file.is_dir():
        subprocess.run(
            ["dvc", "pull", f"{cfg.model.onnx_parameters.mlflow_onnx_export_path}.dvc"]
        )
    run_server(cfg.infer.inference_port)


if __name__ == "__main__":
    fire.Fire(main)
