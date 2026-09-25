import onnxruntime as ort, numpy as np, subprocess, json
W='/mnt/win2/models/models-presets/models/text-token'; S='/mnt/sdd1/models/sherpa'
TEXT="Hello world. This is a longer test sentence for the text to speech node, to check how many seconds of audio come out."
ipa=subprocess.run(['espeak-ng','-q','--ipa=3','-v','en-us',TEXT],capture_output=True,text=True).stdout.replace('\n',' ').strip()
ipa=ipa.replace('_','')
def table(p):
    t={}
    for l in open(p,encoding='utf-8'):
        l=l.rstrip('\n'); 
        if not l: continue
        s,i=l.rsplit(' ',1) if not l.startswith(' ') else (' ',l.split()[-1]); t[s]=int(i)
    return t
so=ort.SessionOptions(); so.log_severity_level=3
# --- Piper
t=table(f'{W}/piper-en_US-amy-low/tokens.txt')
ids=[t['^']]
for ch in ipa:
    if ch in t: ids+= [t[ch], 0]
ids.append(t['$'])
s=ort.InferenceSession(f'{W}/piper-en_US-amy-low/en_US-amy-low.onnx',so,providers=['CPUExecutionProvider'])
o=s.run(None,{'input':np.array([ids],np.int64),'input_lengths':np.array([len(ids)],np.int64),'scales':np.array([0.667,1.0,0.8],np.float32)})[0]
print('PIPER tokens',len(ids),'out shape',o.shape,'secs@16k',o.shape[-1]/16000, 'secs@22050', o.shape[-1]/22050)
import wave
def wr(fn,a,sr):
    a=np.clip(a.reshape(-1),-1,1); w=wave.open(fn,'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((a*32767).astype(np.int16).tobytes()); w.close()
wr('piper.wav',o,16000)
# --- Kitten / Kokoro, style: constant 0.667 (Param1 default) vs voices.bin
for name,mp,tokp,vb,nvoice in [('KITTEN',f'{S}/tts/kitten-nano-en-v0_1-fp16/model.fp16.onnx',f'{S}/tts/kitten-nano-en-v0_1-fp16/tokens.txt',f'{S}/tts/kitten-nano-en-v0_1-fp16/voices.bin',8),
                               ('KOKORO',f'{S}/tts/kokoro-en-v0_19/model.onnx',f'{S}/tts/kokoro-en-v0_19/tokens.txt',f'{S}/tts/kokoro-en-v0_19/voices.bin',11)]:
    t=table(tokp); ids=[0]+[t[c] for c in ipa if c in t]+[0]
    s=ort.InferenceSession(mp,so,providers=['CPUExecutionProvider'])
    inn=[i.name for i in s.get_inputs()]
    v=np.fromfile(vb,np.float32); print(name,'voices.bin floats',v.size,'-> per voice',v.size//nvoice, 'rows of 256:', v.size//nvoice//256)
    per=v.size//nvoice
    real=v[:per].reshape(-1,256); row=real[min(len(ids),real.shape[0]-1)] if real.shape[0]>1 else real[0]
    for lab,st in [('style=voice0',row),('style=const0.667',np.full(256,0.667,np.float32)),('style=0',np.zeros(256,np.float32))]:
        o=s.run(None,{inn[0]:np.array([ids],np.int64),inn[1]:st.reshape(1,256).astype(np.float32),inn[2]:np.array([1.0],np.float32)})[0]
        print(f'  {lab:18s} out {o.shape} secs@24k {o.size/24000:.2f} rms {np.sqrt((o**2).mean()):.4f} peak {np.abs(o).max():.3f}')
        wr(f'{name.lower()}_{lab.split("=")[1]}.wav',o,24000)
