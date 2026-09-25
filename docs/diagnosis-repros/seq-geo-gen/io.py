import onnxruntime as ort, sys
for p in sys.argv[1:]:
    try:
        s=ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(p, 'LOAD FAIL', str(e)[:200]); continue
    print('==', p.split('/')[-1])
    for i in s.get_inputs(): print('  in ', i.name, i.shape, i.type)
    for o in s.get_outputs(): print('  out', o.name, o.shape, o.type)
