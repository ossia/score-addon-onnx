import numpy as np, onnxruntime as ort, time, sys
from PIL import Image
M="/mnt/win2/models/models-presets/models/video-processor/rvm_mobilenetv3_fp32.onnx"
def mk(threads=0):
    so=ort.SessionOptions(); so.log_severity_level=3
    if threads: so.intra_op_num_threads=threads
    return ort.InferenceSession(M,so,providers=['CPUExecutionProvider'])
def crop(im,w,h):
    iw,ih=im.size; sc=max(w/iw,h/ih); im=im.resize((round(iw*sc),round(ih*sc)),Image.BILINEAR)
    x0=(im.size[0]-w)//2; y0=(im.size[1]-h)//2; return im.crop((x0,y0,x0+w,y0+h))
B="/home/jcelerier/projets/oss/ailia-models/"
person=Image.open(B+"background_removal/modnet/input.jpg").convert('RGB')
empty=Image.open(B+"background_removal/modnet/input.jpg").convert('RGB').crop((0,0,1400,800))  # top-left corner: wall/window, no person
def run(s,img,w,h,dr,n=6):
    x=(np.asarray(crop(img,w,h)).astype(np.float32)/255).transpose(2,0,1)[None].copy()
    r=[np.zeros((1,1,1,1),np.float32)]*4; ts=[]
    for i in range(n):
        t=time.perf_counter(); o=s.run(None,{'src':x,'r1i':r[0],'r2i':r[1],'r3i':r[2],'r4i':r[3],'downsample_ratio':np.array([dr],np.float32)}); ts.append(time.perf_counter()-t)
        r=o[2:]
    return o, ts
s=mk()
for nm,img in (('person',person),('empty',empty)):
    o,_=run(s,img,512,288,0.25)
    pha=o[1][0,0]; print(f"{nm:7s} pha range [{pha.min():.4f},{pha.max():.4f}] mean {pha.mean():.4f}; frac>0.5 {(pha>.5).mean():.3f}")
    mm=(pha-pha.min())/(pha.max()-pha.min()+1e-12)
    Image.fromarray(np.concatenate([(np.clip(pha,0,1)*255).astype(np.uint8),(mm*255).astype(np.uint8)],1)).save(f"rvm_{nm}.png")
    print("  fgr range",o[0].min(),o[0].max(),"state shapes",[a.shape for a in o[2:]])
for th in (0,4,1):
    s=mk(th)
    for (w,h,dr) in ((512,288,1.0),(512,288,0.25),(1920,1080,0.25),(3840,2160,0.125)):
        _,ts=run(s,person,w,h,dr,n=5)
        print(f"threads={th or 'default'} {w}x{h} dr={dr}: median {np.median(ts[1:])*1000:.1f} ms")
