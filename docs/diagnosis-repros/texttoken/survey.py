import onnx, sys, glob
from onnx import TensorProto
paths = sys.argv[1:]
for p in paths:
    try:
        m = onnx.load(p, load_external_data=False)
    except Exception as e:
        print("ERR", p, e); continue
    print("=== ", p)
    print("  metadata:", {x.key: (x.value[:80]) for x in m.metadata_props})
    init = {i.name for i in m.graph.initializer}
    def sh(t):
        return [d.dim_value if d.HasField('dim_value') else (d.dim_param or '?') for d in t.type.tensor_type.shape.dim]
    for i in m.graph.input:
        if i.name in init: continue
        print("  IN ", i.name, TensorProto.DataType.Name(i.type.tensor_type.elem_type), sh(i))
    for o in m.graph.output:
        print("  OUT", o.name, TensorProto.DataType.Name(o.type.tensor_type.elem_type), sh(o))
