import mxnet as mx
import numpy as np

# Path to your model
symbol_file = "model_checkpoints/models/target/i1_webface_soft/model_s3-symbol.json"
params_file = "model_checkpoints/models/target/i1_webface_soft/model_s3-0021.params"

# Load symbol and params
sym = mx.sym.load(symbol_file)
params = mx.nd.load(params_file)
arg_params = {k: v for k, v in params.items() if k.startswith('arg:')}
aux_params = {k: v for k, v in params.items() if k.startswith('aux:')}

from mxnet.contrib import onnx as onnx_mxnet

input_shape = (1, 3, 112, 112)  # adjust to your model

onnx_file = "model.onnx"

onnx_mxnet.export_model(
    symbol_file,
    params_file,
    [input_shape],
    np.float32,
    onnx_file
)

print("ONNX model exported to:", onnx_file)