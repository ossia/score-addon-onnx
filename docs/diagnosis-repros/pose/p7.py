import sys, math; sys.path.insert(0,'.')
from rep import *
PK='/home/jcelerier/Documents/ossia/score/packages/pose-detector/detectors/'
def gen(size,strides,apc=2):
  out=[];i=0
  while i<len(strides):
    s=strides[i];j=i
    while j<len(strides) and strides[j]==s: j+=1
    fm=(size+s-1)//s
    for y in range(fm):
      for x in range(fm):
        for _ in range(apc*(j-i)): out.append(((x+.5)/fm,(y+.5)/fm))
    i=j
  return np.array(out)
def blazeface(bgr):
  s=sess(PK+'det-face-blazeface-128.onnx'); ih,iw=bgr.shape[:2]
  t,(sc,px,py)=letterbox(bgr,128,128,True,[127.5]*3,[127.5]*3,order='rgb',layout='nhwc')
  o=s.run(None,{s.get_inputs()[0].name:t}); names=[x.name for x in s.get_outputs()]
  scores=np.concatenate([x[0,:,0] for x in o if x.shape[-1]==1]); boxes=np.concatenate([x[0] for x in o if x.shape[-1]==16])
  a=gen(128,[8,16,16,16]); i=int(np.argmax(scores)); b=boxes[i]
  f=lambda nx,ny:(((nx*128-px)/sc)/iw, ((ny*128-py)/sc)/ih)
  xc,yc=f(a[i,0]+b[0]/128,a[i,1]+b[1]/128); w=(b[2]/sc)/iw; h=(b[3]/sc)/ih
  kps=[f(a[i,0]+b[4+2*k]/128,a[i,1]+b[5+2*k]/128) for k in range(6)]
  return dict(score=float(1/(1+np.exp(-scores[i]))),xc=xc,yc=yc,w=w,h=h,kps=kps)
def priors(W,H):
  L=[([16,32],8),([64,128],16),([256,512],32)]; out=[]
  for ms,st in L:
    fh=(H+st-1)//st; fw=(W+st-1)//st
    for i in range(fh):
      for j in range(fw):
        for m in ms: out.append(((j+.5)*st/W,(i+.5)*st/H,m/W,m/H))
  return np.array(out)
def retina(bgr):
  s=sess(PK+'det-face-retinaface-mobile.onnx'); ih,iw=bgr.shape[:2]; mw=mh=320
  t,(sc,px,py)=letterbox(bgr,mw,mh,False,[104,117,123],[1,1,1])
  o=s.run(None,{s.get_inputs()[0].name:t})
  loc=[x for x in o if x.shape[-1]==4][0][0]; conf=[x for x in o if x.shape[-1]==2][0][0]; lm=[x for x in o if x.shape[-1]==10][0][0]
  e=np.exp(conf-conf.max(1,keepdims=True)); sm=e[:,1]/e.sum(1)
  pr=priors(mw,mh); i=int(np.argmax(sm)); p=pr[i]; l=loc[i]
  f=lambda nx,ny:(((nx*mw-px)/sc)/iw, ((ny*mh-py)/sc)/ih)
  xc,yc=f(p[0]+l[0]*.1*p[2],p[1]+l[1]*.1*p[3]); w=(p[2]*np.exp(l[2]*.2))*mw/sc/iw; h=(p[3]*np.exp(l[3]*.2))*mh/sc/ih
  kps=[f(p[0]+lm[i,2*k]*.1*p[2],p[1]+lm[i,2*k+1]*.1*p[3]) for k in range(5)]
  return dict(score=float(sm[i]),xc=xc,yc=yc,w=w,h=h,kps=kps)
def mprect(d,W,H,scale=1.5,ks=0,ke=1,target=0):
  x0,y0=d['kps'][ks][0]*W,d['kps'][ks][1]*H; x1,y1=d['kps'][ke][0]*W,d['kps'][ke][1]*H
  ang=math.radians(target)-math.atan2(-(y1-y0),x1-x0)
  size=max(d['w']*W,d['h']*H)
  return (round(d['xc']*W,1),round(d['yc']*H,1),round(size*scale,1),round(math.degrees(ang),1))
img=load('face'); H,W=img.shape[:2]
for rot in [0,30,-45,180]:
  M=cv2.getRotationMatrix2D((W/2,H/2),rot,1.0); im=cv2.warpAffine(img,M,(W,H))
  bf=blazeface(im); rf=retina(im)
  print('rot',rot,'BlazeFace',round(bf['score'],2),'rect(cx,cy,size,angle)',mprect(bf,W,H),'eyes',[(round(x*W),round(y*H)) for x,y in bf['kps'][:2]])
  print('rot',rot,'RetinaFc ',round(rf['score'],2),'rect(cx,cy,size,angle)',mprect(rf,W,H),'eyes',[(round(x*W),round(y*H)) for x,y in rf['kps'][:2]], 'box wh',round(rf['w']*W),round(rf['h']*H),'vs bf',round(bf['w']*W),round(bf['h']*H))
