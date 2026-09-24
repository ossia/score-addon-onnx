# Generates the fixtures of test_sequence_batch.cpp.
import os
import onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper

here = os.path.dirname(os.path.abspath(__file__))

def save(name, nodes, ins, outs, inits=[]):
    g = helper.make_graph(nodes, name, ins, outs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(here, name + ".onnx"))

def const(name, arr):
    return numpy_helper.from_array(np.array(arr), name)

# The batch is declared dynamic but a Reshape bakes 2, as in Informer:
# y = 2 * x, runs only when fed [2,3].
save("batch2",
     [helper.make_node("Reshape", ["x", "shape"], ["r"]),
      helper.make_node("Add", ["r", "r"], ["y"])],
     [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", 3])],
     [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["B", 3])],
     [const("shape", np.array([2, 3], np.int64))])

# Loads (the Reshape target is only known at run time) but never runs:
# 4 values cannot be reshaped to [3, ?].
save("broken",
     [helper.make_node("ReduceSum", ["x"], ["s"], keepdims=0),
      helper.make_node("Constant", [], ["zero"], value_float=0.0),
      helper.make_node("Mul", ["s", "zero"], ["z"]),
      helper.make_node("Cast", ["z"], ["zi"], to=TensorProto.INT64),
      helper.make_node("Add", ["target", "zi"], ["shape"]),
      helper.make_node("Reshape", ["x", "shape"], ["y"])],
     [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, "N"])],
     [const("target", np.array([3, -1], np.int64))])

# A recurrent counter: y = x + state, state_out = state + 1. The payload's
# length is free, so a large one takes the worker path.
save("counter",
     [helper.make_node("Add", ["x", "state"], ["y"]),
      helper.make_node("Add", ["state", "one"], ["state_out"])],
     [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "N"]),
      helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 1])],
     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "N"]),
      helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [1, 1])],
     [const("one", np.array([[1.0]], np.float32))])

