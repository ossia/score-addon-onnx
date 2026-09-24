import onnxruntime as ort, numpy as np, wave, glob
from scipy.signal import resample_poly
def load(p):
    w=wave.open(p); sr=w.getframerate(); ch=w.getnchannels(); x=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
    x=x.reshape(-1,ch).mean(1); return resample_poly(x,16000,sr).astype(np.float32) if sr!=16000 else x
sp=np.concatenate([load(p) for p in sorted(glob.glob('/usr/share/sounds/alsa/Front_*.wav'))])
rng=np.random.default_rng(0); noise=(rng.standard_normal(16000*3)*0.05).astype(np.float32)
for path,blk in [('/mnt/win2/models/models-presets/models/audio-analyzer/silero-vad-v4-16k.onnx',512),('/mnt/win2/models/models-presets/models/audio-analyzer/silero-vad-512.onnx',512)]:
    s=ort.InferenceSession(path,providers=['CPUExecutionProvider']); names=[i.name for i in s.get_inputs()]; outs=[o.name for o in s.get_outputs()]
    for label,pair in [('correct',{2:1,3:2}) if 'sr' in names else ('correct',{1:1,2:2}), ('AA/AP (c<-h_out)',{2:1,3:1}) if 'sr' in names else ('AA/AP (c<-h_out)',{1:1,2:1})]:
        for sname,sig in [('speech',sp),('noise',noise)]:
            st={i:np.zeros((2,1,64),np.float32) for i in pair}; probs=[]
            for k in range(0,len(sig)-blk+1,blk):
                feed={names[0]:sig[None,k:k+blk]}
                if 'sr' in names: feed['sr']=np.array(16000,dtype=np.int64)
                for i in pair: feed[names[i]]=st[i]
                r=s.run(None,feed); probs.append(float(r[0].ravel()[0]))
                for i,o in pair.items(): st[i]=r[o]
            p=np.array(probs); print(path.split('/')[-1],label,sname,'mean %.3f  p90 %.3f  frac>0.5 %.2f'%(p.mean(),np.percentile(p,90),(p>0.5).mean()))
