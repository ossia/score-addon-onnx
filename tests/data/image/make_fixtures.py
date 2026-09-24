# Generates the fixtures of test_node_failures.cpp.
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

# An image model that returns its input, but fails on bright images: it also
# reshapes the planes into groups of 2 + round(mean) values (and adds 0 x
# their sum), and a 32 x 32 plane splits into pairs (dark image) but not
# triples (white one).
save("bright_fails",
     [helper.make_node("ReduceMean", ["image"], ["m"], keepdims=0),
      helper.make_node("Round", ["m"], ["r"]),
      helper.make_node("Cast", ["r"], ["ri"], to=TensorProto.INT64),
      helper.make_node("Add", ["ri", "two"], ["k"]),
      helper.make_node("Unsqueeze", ["k", "axes"], ["k1"]),
      helper.make_node("Concat", ["head", "k1"], ["groups"], axis=0),
      helper.make_node("Reshape", ["image", "groups"], ["p"]),
      helper.make_node("ReduceSum", ["p"], ["sum"], keepdims=0),
      helper.make_node("Mul", ["sum", "zero"], ["nothing"]),
      helper.make_node("Add", ["image", "nothing"], ["out"])],
     [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 32, 32])],
     [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 32, 32])],
     [numpy_helper.from_array(np.array(2, np.int64), "two"),
      numpy_helper.from_array(np.array(0, np.float32), "zero"),
      numpy_helper.from_array(np.array([0], np.int64), "axes"),
      numpy_helper.from_array(np.array([1, 3, -1], np.int64), "head")])
