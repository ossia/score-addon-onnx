# Generates the fixtures of test_aux_inputs.cpp.
import os
import onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper

here = os.path.dirname(os.path.abspath(__file__))

def save(name, nodes, ins, outs):
    g = helper.make_graph(nodes, name, ins, outs)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(here, name + ".onnx"))

# Sequence: y = x * t, t a scalar control (Param 1).
save("scale",
     [helper.make_node("Mul", ["x", "t"], ["y"])],
     [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4]),
      helper.make_tensor_value_info("t", TensorProto.FLOAT, [1])],
     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])

# Geometry: out = points + t, the cloud first or second.
pts = helper.make_tensor_value_info("points", TensorProto.FLOAT, [1, "N", 3])
t = helper.make_tensor_value_info("t", TensorProto.FLOAT, [1])
out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, "N", 3])
save("geom_2in", [helper.make_node("Add", ["points", "t"], ["out"])], [pts, t], [out])
save("geom_cloud_second", [helper.make_node("Add", ["points", "t"], ["out"])], [t, pts], [out])

# Audio Analyzer: y = mean(audio) + 100 * flag, flag a bool (CLAP `longer`).
save("audio_flag",
     [helper.make_node("ReduceMean", ["audio"], ["m"], keepdims=1),
      helper.make_node("Cast", ["flag"], ["ff"], to=TensorProto.FLOAT),
      helper.make_node("Constant", [], ["hundred"], value_float=100.0),
      helper.make_node("Mul", ["ff", "hundred"], ["f100"]),
      helper.make_node("Add", ["m", "f100"], ["y"])],
     [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "N"]),
      helper.make_tensor_value_info("flag", TensorProto.BOOL, [1])],
     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])])

# Audio Processor: a 49152-sample block (above 48000, so it runs async), y = x/2.
save("audio_big_block",
     [helper.make_node("Constant", [], ["half"], value_float=0.5),
      helper.make_node("Mul", ["audio", "half"], ["out"])],
     [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, 49152])],
     [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 49152])])

# Audio Processor: 4 stereo stems [1,4,2,N], stem k = mix * (k+1) (Demucs layout).
mix = helper.make_tensor_value_info("mix", TensorProto.FLOAT, [1, 2, "N"])
stems = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2, "N"])
nodes = [helper.make_node("Unsqueeze", ["mix", "axes"], ["u"])]
for k in range(4):
    nodes.append(helper.make_node("Constant", [], [f"g{k}"], value_float=float(k + 1)))
    nodes.append(helper.make_node("Mul", ["u", f"g{k}"], [f"s{k}"]))
nodes.append(helper.make_node("Concat", [f"s{k}" for k in range(4)], ["x"], axis=1))
g = helper.make_graph(nodes, "stems", [mix], [stems],
                      [numpy_helper.from_array(np.array([1], np.int64), "axes")])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, os.path.join(here, "audio_stems.onnx"))

# Audio Processor: out = audio * gain, gain a scalar control (Param 2).
save("audio_gain",
     [helper.make_node("Mul", ["audio", "gain"], ["out"])],
     [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "N"]),
      helper.make_tensor_value_info("gain", TensorProto.FLOAT, [1])],
     [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, "N"])])
