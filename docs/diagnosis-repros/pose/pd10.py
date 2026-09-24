import sys; sys.path.insert(0,'.')
from rep import *
PK='/home/jcelerier/Documents/ossia/score/packages/pose-detector/detectors/'
def ssd(p,a,b,imgname):
  s=sess(PK+p); sh=s.get_inputs()[0].shape; nhwc=sh[3]==3; m=sh[1] if nhwc else sh[2]
  t,_=letterbox(load(imgname),m,m,True,[-b*255/a]*3,[255/a]*3,order='rgb',layout='nhwc' if nhwc else 'nchw')
  o=s.run(None,{s.get_inputs()[0].name:t}); sc=np.concatenate([x[0,:,0] for x in o if x.shape[-1]==1])
  return sorted((1/(1+np.exp(-np.clip(sc,-80,80)))).tolist())[-3:]
def e2e(p,imgname):
  s=sess(PK+p); m=s.get_inputs()[0].shape[2]
  mean,std=([103.53,116.28,123.675],[57.375,57.12,58.395]) if m<=320 else ([0,0,0],[1,1,1])
  t,_=letterbox(load(imgname),m,m,False,mean,std)
  o=s.run(None,{s.get_inputs()[0].name:t}); d=o[0][0]; l=o[1][0]
  return sorted(d[l==0][:,4].tolist())[-4:]
for img in ['body','hand','face']:
  print(img,'blazepose128',[round(x,3) for x in ssd('det-body-blazepose-128.onnx',2,-1,img)])
  print(img,'palm128',[round(x,3) for x in ssd('det-hand-palm-128.onnx',1,0,img)])
  print(img,'blazeface128',[round(x,3) for x in ssd('det-face-blazeface-128.onnx',2,-1,img)])
  print(img,'yolox416 person',[round(x,3) for x in e2e('det-body-yolox-416.onnx',img)])
  print(img,'rtmdet320 hand',[round(x,3) for x in e2e('det-hand-rtmdet-320.onnx',img)])
