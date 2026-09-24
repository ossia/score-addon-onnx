import numpy as np, onnxruntime as ort
from PIL import Image
so=ort.SessionOptions(); so.log_severity_level=3
P=Image.open("/home/jcelerier/projets/oss/ailia-models/background_removal/modnet/input.jpg").convert('RGB')
def crop(im,w,h):
    iw,ih=im.size; sc=max(w/iw,h/ih); im=im.resize((round(iw*sc),round(ih*sc)),Image.BILINEAR)
    x0=(im.size[0]-w)//2; y0=(im.size[1]-h)//2; return im.crop((x0,y0,x0+w,y0+h))
def sess(p): return ort.InferenceSession(p,so,providers=['CPUExecutionProvider'])
tiles=[]
# --- PP-HumanSeg (NCHW 2ch, preset-less) ---
s=sess("/mnt/sdd1/models/pinto/196__human_segmentation_pphumanseg_2021oct.onnx")
im=crop(P,192,192); a=np.asarray(im).astype(np.float32)
for nm,(m,sd) in {'Centered':(127.5,127.5)}.items():
    y=s.run(None,{'x':((a-m)/sd).transpose(2,0,1)[None]})[0]
    print('pphumanseg out',y.shape,'ch0 range',y[0,0].min(),y[0,0].max(),'ch1 range',y[0,1].min(),y[0,1].max())
    ctr=(slice(60,130),slice(70,120)); corner=(slice(0,20),slice(0,20))
    print('  center(person) ch0 mean %.3f ch1 mean %.3f | corner(bg) ch0 %.3f ch1 %.3f'%(y[0,0][ctr].mean(),y[0,1][ctr].mean(),y[0,0][corner].mean(),y[0,1][corner].mean()))
    mm=lambda v:((v-v.min())/(v.max()-v.min())*255).astype(np.uint8)
    tiles.append(np.asarray(im.resize((256,256)))); 
    for c in (0,1): tiles.append(np.repeat(np.asarray(Image.fromarray(mm(y[0,c])).resize((256,256)))[:,:,None],3,2))
# --- hand seg (dynamic) ---
s=sess("/mnt/sdd1/models/wild2/hand_recognition__hands_segmentation_pytorch__hands_segmentation_pytorch.onnx")
im=crop(P,256,256); a=np.asarray(im).astype(np.float32)
m=np.array([255*.485,255*.456,255*.406],np.float32); sd=np.array([255*.229,255*.224,255*.225],np.float32)
y=s.run(None,{'input':((a-m)/sd).transpose(2,0,1)[None]})[0]
print('handseg out',y.shape,'ch0 mean %.3f ch1 mean %.3f  frac(ch1>ch0)=%.3f'%(y[0,0].mean(),y[0,1].mean(),(y[0,1]>y[0,0]).mean()))
# --- hair 060 ---
s=sess("/mnt/sdd1/models/pinto/060__hair_segmenter.onnx")
im=crop(P,512,512); a=np.asarray(im).astype(np.float32)/255
try:
    s.run(None,{'input':a.transpose(2,0,1)[None].copy()})
except Exception as e: print('hair060 with 3ch tensor (what C++ builds):',str(e).splitlines()[0][:200])
x=np.concatenate([a,np.zeros_like(a[:,:,:1])],2).transpose(2,0,1)[None].copy()
y=s.run(None,{'input':x})[0]
print('hair060 4ch (alpha=0) out',y.shape,'ch0 mean %.3f ch1 mean %.3f frac(hair)=%.3f'%(y[0,:,:,0].mean(),y[0,:,:,1].mean(),(y[0,:,:,1]>y[0,:,:,0]).mean()))
tiles.append(np.asarray(im.resize((256,256))))
hair=(np.exp(y[0,:,:,1])/(np.exp(y[0,:,:,0])+np.exp(y[0,:,:,1])))
tiles.append(np.repeat(np.asarray(Image.fromarray((hair*255).astype(np.uint8)).resize((256,256)))[:,:,None],3,2))
# second frame with prior = previous mask
x2=np.concatenate([a,hair[:,:,None].astype(np.float32)],2).transpose(2,0,1)[None].copy()
y2=s.run(None,{'input':x2})[0]; hair2=(np.exp(y2[0,:,:,1])/(np.exp(y2[0,:,:,0])+np.exp(y2[0,:,:,1])))
print('hair060 frame2 (alpha=prev mask) frac(hair)=%.3f  meanabs diff vs frame1 %.3f'%((hair2>.5).mean(),np.abs(hair2-hair).mean()))
Image.fromarray(np.concatenate(tiles,1)).save('seg_cmp.png')
