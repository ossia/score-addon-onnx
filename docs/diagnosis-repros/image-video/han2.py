import numpy as np, onnxruntime as ort
from PIL import Image
so=ort.SessionOptions(); so.log_severity_level=3
s=ort.InferenceSession("/mnt/win2/models/models-presets/models/image-processor/han-x3.onnx",so,providers=['CPUExecutionProvider'])
B="/home/jcelerier/projets/oss/ailia-models/"
imgs=[B+"background_removal/modnet/input.jpg",B+"background_removal/cascade_psp/aeroplane.jpg",B+"face_detection/mtcnn/input.jpg",B+"pose_estimation/animalpose/input.jpg",B+"depth_estimation/depth_anything/demo1.png"]
def psnr(a,b,sh=6):
    a=a[sh:-sh,sh:-sh].astype(np.float64); b=b[sh:-sh,sh:-sh].astype(np.float64); return 10*np.log10(255**2/((a-b)**2).mean())
cfg={'None/Passthrough':(0,1,lambda v:v),'Centered/Denormalize':(127.5,127.5,lambda v:(v+1)*127.5),'DivBy255/DirectClamp':(0,255,lambda v:v*255)}
rows=[]
for p in imgs:
    im=Image.open(p).convert('RGB'); w,h=im.size
    sc=max(1,min(w,h)//480); im=im.resize((w//sc,h//sc),Image.LANCZOS) if sc>1 else im
    w,h=im.size; S=min(w,h)//3*3; S=min(S,480)
    hr=im.crop(((w-S)//2,(h-S)//2,(w-S)//2+S,(h-S)//2+S)); H=np.asarray(hr)
    lr=hr.resize((S//3,S//3),Image.BICUBIC); a=np.asarray(lr).astype(np.float32)
    bic=np.asarray(lr.resize((S,S),Image.BICUBIC))
    line=f"{p.split('/')[-2]:12s} S={S} bicubic {psnr(bic,H):.2f}"
    outs={}
    for k,(m,sd,f) in cfg.items():
        for order in ('RGB','BGR'):
            aa=a[:,:,::-1] if order=='BGR' else a
            x=((aa-m)/sd).transpose(2,0,1)[None].copy()
            y=s.run(None,{'lr':x})[0][0].transpose(1,2,0)
            if order=='BGR': y=y[:,:,::-1]
            o=np.clip(np.round(f(y.astype(np.float64))),0,255).astype(np.uint8)
            outs[(k,order)]=o
            line+=f" | {k}/{order} {psnr(o,H):.2f}"
    print(line)
    c=slice(S//2-80,S//2+80)
    rows.append(np.concatenate([H[c,c],bic[c,c],outs[('None/Passthrough','RGB')][c,c],outs[('Centered/Denormalize','RGB')][c,c]],1))
Image.fromarray(np.concatenate(rows,0)).resize((640*2,160*len(rows)*2),Image.NEAREST).save('han_cmp.png')
