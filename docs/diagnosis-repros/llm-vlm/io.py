import onnx, sys, glob, os
def summarize(p):
    m = onnx.load(p, load_external_data=False)
    g=m.graph
    init={i.name for i in g.initializer}
    ins=[i for i in g.input if i.name not in init]
    def s(v):
        t=v.type.tensor_type
        dims=[d.dim_param or d.dim_value for d in t.shape.dim]
        return f"{v.name}:{onnx.TensorProto.DataType.Name(t.elem_type)}{dims}"
    def compress(lst):
        out=[];
        for x in lst:
            out.append(s(x))
        # compress repetitive
        if len(out)>8: out=out[:6]+[f"...({len(out)} total)"]+out[-2:]
        return out
    print("==",p)
    print(" IN:", compress(ins))
    print(" OUT:", compress(g.output))
for p in sys.argv[1:]:
    try: summarize(p)
    except Exception as e: print("ERR",p,e)
