import numpy as np, onnxruntime as ort
from PIL import Image
so=ort.SessionOptions(); so.log_severity_level=3
s=ort.InferenceSession("/mnt/win2/models/models-presets/models/image-processor/han-x3.onnx",so,providers=['CPUExecutionProvider'])
imgs={'portrait':"/home/jcelerier/projets/oss/ailia-models/background_removal/modnet/input.jpg",
 'scene':"/home/jcelerier/projets/oss/ailia-models/depth_estimation/depth_anything/demo1.png",
 'sr':"/home/jcelerier/projets/oss/ailia-models/super_resolution/real-esrgan/input.jpg"}
def psnr(a,b): m=((a.astype(np.float64)-b.astype(np.float64))**2).mean(); return 10*np.log10(255**2/m)
rows=[]
for k,p in imgs.items():
    im=Image.open(p).convert('RGB'); w,h=im.size; S=min(w,h,768)
    hr=im.crop(((w-S)//2,(h-S)//2,(w-S)//2+S,(h-S)//2+S)).resize((768,768),Image.BICUBIC)
    lr=hr.resize((256,256),Image.BICUBIC); a=np.asarray(lr).astype(np.float32)
    H=np.asarray(hr)
    res={}
    for norm,(m,sd) in {'None':(0,1),'DivBy255':(0,255),'Centered':(127.5,127.5)}.items():
        x=((a-m)/sd).transpose(2,0,1)[None].copy()
        y=s.run(None,{'lr':x})[0][0].transpose(1,2,0).astype(np.float64)
        print(k,norm,'out range',round(y.min(),3),round(y.max(),3),'mean',round(y.mean(),3))
        for mode,f in {'Passthrough':lambda v:np.clip(v,0,255),'Denormalize':lambda v:np.clip((v+1)*127.5,0,255),'DirectClamp':lambda v:np.clip(v,0,1)*255}.items():
            o=np.round(f(y)).astype(np.uint8); res[(norm,mode)]=o
            print(f"  {mode:12s} PSNR {psnr(o,H):.2f}")
    bic=np.asarray(lr.resize((768,768),Image.BICUBIC)); print(k,'bicubic PSNR',round(psnr(bic,H),2))
    # crop a 256 region for visual
    c=slice(256,512)
    tiles=[H[c,c],bic[c,c],res[('None','Passthrough')][c,c],res[('Centered','Denormalize')][c,c]]
    rows.append(np.concatenate(tiles,1))
Image.fromarray(np.concatenate(rows,0)).save('han_cmp.png')
