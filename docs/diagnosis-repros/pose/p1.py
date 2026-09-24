import sys; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from rep import *
def multiclass(p, mw, mh, norm, img='body', thr=0.4, keep=0):
    s=sess(p); bgr=load(img)
    std = [1,1,1] if norm=='raw' else [255,255,255]
    t,(sc,px,py)=letterbox(bgr,mw,mh,False,[0,0,0],std)
    out=s.run(None,{s.get_inputs()[0].name:t})[0]
    sbb='score_x' in s.get_outputs()[0].name.lower()
    cs,cx=(2,3) if sbb else (6,2)
    rows=[r for r in out if r[cs]>=thr]
    kept=[r for r in rows if keep<0 or int(r[1]+.5)==keep]
    ih,iw=bgr.shape[:2]
    desc=[(int(r[1]+.5),round(float(r[cs]),3),[round(float((r[cx+i]-(px if i%2==0 else py))/sc),0) for i in range(4)]) for r in kept[:5]]
    return len(out),len(rows),len(kept),desc
import glob
for p,mw,mh in [('/mnt/win2/models/libreonnx/423_6DRepNet360/gold_yolo_n_head_post_0277_0.5071_1x3x480x640.onnx',640,480),
                ('/mnt/win2/models/libreonnx/423_6DRepNet360/gold_yolo_m_head_post_0248_0.5327_1x3x384x640.onnx',640,384),
                ('/mnt/sdd1/models/pinto/426__yolox_n_body_head_hand_post_0461_0.4428_1x3x128x160.onnx',160,128),
                ('/mnt/win2/models/models-presets/models/pose-detector/detectors/det-bhh-yolox-n-480x640.onnx',640,480),
                ('/home/jcelerier/Documents/ossia/score/packages/pose-detector/detectors/det-body-yolox-bhh-320.onnx',320,320)]:
  for img in ['body','demo1']:
    for norm in ['raw','unit']:
        print(p.split('/')[-1],img,norm,'total/above_thr/kept(class0 or all for head):',multiclass(p,mw,mh,norm,img,keep=-1))
