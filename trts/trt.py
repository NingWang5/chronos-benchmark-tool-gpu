import os
import sys
import numpy as np

# This import causes pycuda to automatically manage CUDA context creation and cleanup.

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

import tensorrt as trt

sys.path.insert(1, os.path.join(sys.path[0], "."))
from . import common

class ModelData(object):
    MODEL_PATH = "ResNet50.onnx"
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The Onnx path is used for Onnx models.
def build_engine_onnx(args, model_file, train_loader, test_loader):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if args.quantize:
        from . import calibrator
        calib = calibrator.Calibrator(test_loader)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib

    config.max_workspace_size = common.GiB(1)

    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    return builder.build_engine(network, config)


def load_normalized_test_case(test_case, pagelocked_buffer):

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, test_case)
    return test_case


def main(args, onnx_model_file, train_loader, test_loader):

    # Build a TensorRT engine.
    engine = build_engine_onnx(args, onnx_model_file, train_loader, test_loader)
    # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
    # Allocate buffers and create a CUDA stream.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Contexts are used to perform inference.
    context = engine.create_execution_context()

    latency = []
    import time
    for x, y in test_loader:
        test_case = x.numpy()
        # Copy to pagelocked memory.
        inputs[0].host = test_case
        # Run the engine.
        st = time.time()
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        latency.append(time.time()-st)

    from scipy import stats
    avg_latnecy = stats.trim_mean(latency, 0.1)
    percentile_latency = np.percentile(latency, [50, 90, 95, 99])
    return avg_latnecy, percentile_latency
