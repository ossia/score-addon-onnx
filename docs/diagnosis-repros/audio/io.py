import onnx,sys
for p in sys.argv[1:]:
    m=onnx.load(p,load_external_data=False)
    init={i.name for i in m.graph.initializer}
    def sh(v):
        t=v.type.tensor_type
        return [d.dim_value if d.HasField('dim_value') else (d.dim_param or '?') for d in t.shape.dim], onnx.TensorProto.DataType.Name(t.elem_type)
    print('==',p.split('/')[-1], 'meta:',{x.key:x.value[:60] for x in m.metadata_props})
    for v in m.graph.input:
        if v.name in init: continue
        print('  IN ',v.name,*sh(v))
    for v in m.graph.output: print('  OUT',v.name,*sh(v))
