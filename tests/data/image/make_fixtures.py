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

# Two generators with a latent z [1,4] and a fixed [1,3,4,4] image: one in
# tanh range (Auto maps it with Denormalize), one in 0..255 (raw values).
ramp = np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(1, 3, 4, 4)
for name, image in (("gen_tanh", ramp * 1.8 - 0.9), ("gen_bytes", ramp * 210 + 20)):
    save(name,
         [helper.make_node("ReduceSum", ["z"], ["s"], keepdims=0),
          helper.make_node("Mul", ["s", "zero"], ["nothing"]),
          helper.make_node("Add", ["image", "nothing"], ["out"])],
         [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])],
         [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, 4, 4])],
         [numpy_helper.from_array(image.astype(np.float32), "image"),
          numpy_helper.from_array(np.array(0, np.float32), "zero")])

# Two image models of any size: one returns its input, one returns black.
for name, k in (("identity_dyn", 1.0), ("black_dyn", 0.0)):
    save(name,
         [helper.make_node("Mul", ["image", "k"], ["out"])],
         [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "H", "W"])],
         [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, "H", "W"])],
         [numpy_helper.from_array(np.array(k, np.float32), "k")])

# Two point-cloud models of any size: points + 1, and points * 0.
for name, op, v in (("cloud_plus1", "Add", 1.0), ("cloud_zero", "Mul", 0.0)):
    save(name,
         [helper.make_node(op, ["points", "v"], ["out"])],
         [helper.make_tensor_value_info("points", TensorProto.FLOAT, [1, "N", 3])],
         [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, "N", 3])],
         [numpy_helper.from_array(np.array(v, np.float32), "v")])
