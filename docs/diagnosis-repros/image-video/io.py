import sys, onnxruntime as ort
so=ort.SessionOptions(); so.log_severity_level=3
for p in sys.argv[1:]:
    try:
        s=ort.InferenceSession(p,so,providers=['CPUExecutionProvider'])
    except Exception as e:
        print(p,"ERR",e); continue
    print("==",p)
    for i in s.get_inputs(): print("  in ",i.name,i.shape,i.type)
    for o in s.get_outputs(): print("  out",o.name,o.shape,o.type)
