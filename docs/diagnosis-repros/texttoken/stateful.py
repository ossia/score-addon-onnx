import onnx,sys
from onnx import TensorProto as TP
m={TP.FLOAT:'Float',TP.INT64:'Int64',TP.INT32:'Int32',TP.BOOL:'Bool',TP.FLOAT16:'Float16'}
for name,p in [(a.split('=')[0],a.split('=')[1]) for a in sys.argv[1:]]:
    g=onnx.load(p,load_external_data=False).graph; init={i.name for i in g.initializer}
    def sh(t): return ','.join(str(d.dim_value) if d.HasField('dim_value') else '-1' for d in t.type.tensor_type.shape.dim)
    ins=','.join('{"%s",{%s},T::%s}'%(i.name,sh(i),m.get(i.type.tensor_type.elem_type,'Float')) for i in g.input if i.name not in init)
    outs=','.join('{"%s",{%s},T::%s}'%(i.name,sh(i),m.get(i.type.tensor_type.elem_type,'Float')) for i in g.output)
    print('  st("%s",{%s},{%s});'%(name,ins,outs))
