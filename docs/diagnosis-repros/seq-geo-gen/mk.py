import onnx, sys
T={1:'float',10:'float16',16:'bfloat16',11:'double',2:'uint8',3:'int8',4:'uint16',5:'int16',12:'uint32',6:'int32',13:'uint64',7:'int64',9:'bool'}
def dims(t):
    d=t.type.tensor_type.shape.dim
    if len(d)==0: return 'scalar'
    return ','.join(str(x.dim_value if x.HasField('dim_value') else -1) for x in d)
for p in sys.argv[1:]:
    m=onnx.load(p, load_external_data=False)
    inits={i.name for i in m.graph.initializer}
    print('MODEL',p.split('/')[-1])
    for i in m.graph.input:
        if i.name in inits: continue
        print('IN',i.name,T.get(i.type.tensor_type.elem_type,'float'),dims(i))
    for o in m.graph.output: print('OUT',o.name,T.get(o.type.tensor_type.elem_type,'float'),dims(o))
    print('END')
