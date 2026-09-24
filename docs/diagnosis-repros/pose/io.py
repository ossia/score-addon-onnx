import onnxruntime as ort, sys
for p in sys.argv[1:]:
    try:
        s=ort.InferenceSession(p,providers=['CPUExecutionProvider'])
        print(p.split('/')[-1], [(i.name,i.shape) for i in s.get_inputs()], '->', [(o.name,o.shape) for o in s.get_outputs()])
    except Exception as e: print(p, 'ERR', str(e)[:100])
