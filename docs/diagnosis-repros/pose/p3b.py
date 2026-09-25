import sys; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from rep import *
for p in ['/mnt/sdd1/models/wild2/pose2__yolox__yolox_s.opt.onnx','/mnt/sdd1/models/wild2/pose2__yolox__yolox_tiny.opt.onnx',
          '/home/jcelerier/Documents/ossia/score/packages/pose-detector/detectors/det-coco-yolox-416.onnx',
          '/mnt/win2/models/models-presets/models/pose-detector/detectors/det-coco-yolox-nano-480x640.onnx']:
    s=sess(p); sh=s.get_inputs()[0].shape; mh,mw=sh[2],sh[3]
    for img in ['body','face','hand']:
      r=[]
      for norm in ['raw','unit']:
        t,_=letterbox(load(img),mw,mh,False,[0,0,0],[1,1,1] if norm=='raw' else [255]*3)
        o=s.run(None,{s.get_inputs()[0].name:t})[0][0]
        sc=o[:,4]*o[:,5:].max(1); r.append((norm,int((sc>0.3).sum()),round(float(sc.max()),3)))
      print(p.split('/')[-1],img,r)
