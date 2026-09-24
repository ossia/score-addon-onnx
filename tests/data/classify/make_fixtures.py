# Generates the fixture of test_node_model_swap.cpp: an image classifier that
# declares a dynamic H/W but only runs at 8x8 (it reshapes to a fixed length),
# so another resolution fails on that frame only.
import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

here = os.path.dirname(os.path.abspath(__file__))
w = numpy_helper.from_array(
    (np.arange(192 * 4, dtype=np.float32).reshape(192, 4) % 7 - 3) / 100, "w")
shape = numpy_helper.from_array(np.array([1, 192], np.int64), "shape")
g = helper.make_graph(
    [helper.make_node("Reshape", ["x", "shape"], ["flat"]),
     helper.make_node("MatMul", ["flat", "w"], ["y"])],
    "dyn8",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "H", "W"])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    [w, shape])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, os.path.join(here, "dyn8.onnx"))
