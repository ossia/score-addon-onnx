# Generates the fixtures of test_audio_overlap.cpp.
import os
import onnx
from onnx import TensorProto, helper

here = os.path.dirname(os.path.abspath(__file__))

def save(name, nodes, ins, outs):
    g = helper.make_graph(nodes, name, ins, outs)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(here, name + ".onnx"))

# A frame-based model that returns its 512-sample frame unchanged.
save("identity_512",
     [helper.make_node("Identity", ["audio"], ["out"])],
     [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, 512])],
     [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 512])])
