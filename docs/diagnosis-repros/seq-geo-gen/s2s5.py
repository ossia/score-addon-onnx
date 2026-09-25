import onnxruntime as ort, numpy as np
ort.set_default_logger_severity(3)
s=ort.InferenceSession('/mnt/sdd1/models/wild/silero-vad__silero_vad.onnx',providers=['CPUExecutionProvider'])
t=np.arange(16000*2)/16000.
rng=np.random.default_rng(0)
speechlike=(0.3*np.sin(2*np.pi*220*t)*(1+np.sin(2*np.pi*3*t))+0.02*rng.standard_normal(t.size)).astype(np.float32)
for sr in [0,8000,16000]:
    h=np.zeros((2,1,64),np.float32); c=h.copy(); outs=[]
    try:
        for k in range(0, 16000, 512):
            o,h,c=s.run(None,{'input':speechlike[None,k:k+512],'sr':np.array(sr,np.int64),'h':h,'c':c})
            outs.append(float(o[0,0]))
        print('silero sr=',sr,'mean prob',np.mean(outs[5:]),'last',outs[-1])
    except Exception as e: print('silero sr=',sr,'ERR',str(e)[:200])
s=ort.InferenceSession('/mnt/sdd1/models/wild/informer2020__informer_ETTh1.onnx',providers=['CPUExecutionProvider'])
for B in [1,2]:
    feed={i.name:np.zeros([B]+i.shape[1:],np.float32) for i in s.get_inputs()}
    try: print('informer B=',B,'ok', s.run(None,feed)[0].shape)
    except Exception as e: print('informer B=',B,'ERR',str(e)[:300])
