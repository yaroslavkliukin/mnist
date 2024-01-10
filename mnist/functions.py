import os
import random

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
from mlflow.models import infer_signature


def set_seed(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def convert_to_onnx(model, conf):
    model.eval()
    input_tensor = torch.randn(1, *conf.input_shape)
    torch.onnx.export(
        model,
        input_tensor,
        conf.onnx_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={"IMAGES": {0: "BATCH_SIZE"}, "CLASS_PROBS": {0: "BATCH_SIZE"}},
    )

    # check that all is good
    original_emb = model(input_tensor).detach().numpy()
    ort_input = {
        "IMAGES": input_tensor.numpy(),
    }

    ort_session = ort.InferenceSession(conf.onnx_path)
    onnx_embedding = ort_session.run(None, ort_input)[0]

    assert np.allclose(
        original_emb, onnx_embedding, atol=1e-5
    ), "something wrond with onnx model"

    # register model in mlflow
    onnx_model = onnx.load(conf.onnx_path)
    with mlflow.start_run():
        signature = infer_signature(input_tensor.numpy(), original_emb)
        mlflow.onnx.save_model(
            onnx_model, conf.mlflow_onnx_export_path, signature=signature
        )
