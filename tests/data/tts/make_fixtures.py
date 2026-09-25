# Generates the fixtures of test_texttoken_rate.cpp. TTS: int64 token ids
# [1,L] in, a constant 64-sample waveform [1,1,1,64] out (the Piper rank-4
# layout), in three flavours of sample-rate information.
import json, os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

here = os.path.dirname(os.path.abspath(__file__))

def model(meta_rate=None):
    x = helper.make_tensor_value_info("input", TensorProto.INT64, [1, "L"])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 1, 64])
    wave = numpy_helper.from_array(np.full((1, 1, 1, 64), 0.5, np.float32), "wave")
    zero = numpy_helper.from_array(np.zeros((), np.float32), "zero")
    nodes = [
        helper.make_node("Cast", ["input"], ["xf"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["xf"], ["s"], keepdims=0),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Add", ["wave", "z"], ["output"]),
    ]
    g = helper.make_graph(nodes, "tts", [x], [y], [wave, zero])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    if meta_rate:
        helper.set_model_props(m, {"sample_rate": str(meta_rate)})
    onnx.checker.check_model(m)
    return m

def write(sub, meta_rate=None, sidecar_rate=None):
    d = os.path.join(here, sub)
    os.makedirs(d, exist_ok=True)
    onnx.save(model(meta_rate), os.path.join(d, "tts.onnx"))
    if sidecar_rate:
        with open(os.path.join(d, "tts.onnx.json"), "w") as f:
            json.dump({"audio": {"sample_rate": sidecar_rate}}, f)

write("sidecar", sidecar_rate=16000)              # upstream Piper voice
write("metadata", meta_rate=24000, sidecar_rate=16000)  # metadata wins
write("none")                                     # name guess: 22050

# int32 / bool inputs (BUG-LEDGER T5): data = 2 * tokens + text_lengths + 100 * flag
def int_model():
    t = helper.make_tensor_value_info("inputs", TensorProto.INT32, [1, "L"])
    n = helper.make_tensor_value_info("text_lengths", TensorProto.INT32, [1])
    b = helper.make_tensor_value_info("flag", TensorProto.BOOL, [1])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "L"])
    two = numpy_helper.from_array(np.array(2, np.float32), "two")
    hundred = numpy_helper.from_array(np.array(100, np.float32), "hundred")
    nodes = [
        helper.make_node("Cast", ["inputs"], ["tf"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["text_lengths"], ["nf"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["flag"], ["bf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["tf", "two"], ["t2"]),
        helper.make_node("Mul", ["bf", "hundred"], ["b100"]),
        helper.make_node("Add", ["t2", "nf"], ["s"]),
        helper.make_node("Add", ["s", "b100"], ["logits"]),
    ]
    g = helper.make_graph(nodes, "int32", [t, n, b], [y], [two, hundred])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m

os.makedirs(os.path.join(here, "int32"), exist_ok=True)
onnx.save(int_model(), os.path.join(here, "int32", "encoder.onnx"))
