import onnxruntime as ort, numpy as np, time
s=ort.InferenceSession('/mnt/sdd1/models/wild/demucs__htdemucs_ft_vocals.onnx',providers=['CPUExecutionProvider'])
for n in [1024, 44100, 343980]:
    x=(np.random.randn(1,2,n)*0.1).astype(np.float32)
    try:
        t=time.time(); y=s.run(None,{'mix':x})[0]; print(n,'->',y.shape, 'time %.2fs'%(time.time()-t))
    except Exception as e: print(n,'ERR',str(e)[:200])
