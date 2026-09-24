import sys; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from rep import *
def grid(out,mw,mh,thr=0.3):
    d=out[0]; A,F=d.shape; dets=[]; idx=0
    for st in (8,16,32):
        gh,gw=mh//st,mw//st
        for gy in range(gh):
            for gx in range(gw):
                if idx>=A: break
                r=d[idx]; idx+=1
                bc=int(np.argmax(r[5:])); sc=r[4]*r[5+bc]
                if sc>=thr and bc==0:
                    dets.append((float(sc),(r[0]+gx)*st,(r[1]+gy)*st,np.exp(r[2])*st,np.exp(r[3])*st))
    return dets
for p in ['/mnt/sdd1/models/wild2/pose2__yolox__yolox_s.opt.onnx','/mnt/sdd1/models/wild2/pose2__yolox__yolox_tiny.opt.onnx',
          '/home/jcelerier/Documents/ossia/score/packages/pose-detector/detectors/det-coco-yolox-416.onnx',
          '/mnt/win2/models/models-presets/models/pose-detector/detectors/det-coco-yolox-nano-480x640.onnx']:
    s=sess(p); sh=s.get_inputs()[0].shape; mh,mw=sh[2],sh[3]
    for norm in ['raw','unit']:
        t,(sc,px,py)=letterbox(load('body'),mw,mh,False,[0,0,0],[1,1,1] if norm=='raw' else [255]*3)
        out=s.run(None,{s.get_inputs()[0].name:t})[0]
        d=grid(out,mw,mh)
        print(p.split('/')[-1],norm,'maxobj',round(float(out[0,:,4].max()),3),'persons>=0.3 (pre-NMS):',len(d), 'top', [ (round(x[0],2), int((x[1]-px)/sc), int((x[2]-py)/sc)) for x in sorted(d,reverse=True)[:3]])
