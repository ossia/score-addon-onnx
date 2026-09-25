import sys; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from rep import *
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
p='/mnt/sdd1/models/wild2/pose2__blazepalm__blazepalm.onnx'; s=sess(p)
for img in ['hand','body','handcrop','bodyhandcrop']:
  bgr=load(img); ih,iw=bgr.shape[:2]
  t,(sc,px,py)=letterbox(bgr,256,256,True,[0,0,0],[255]*3,order='rgb')
  outs=s.run(None,{s.get_inputs()[0].name:t})
  reg=[o for o in outs if o.shape[-1]==18][0][0]; cls=[o for o in outs if o.shape[-1]==1][0][0,:,0]
  for name,strides in [('current palmParams(256) {8,16,16,16}',[8,16,16,16]),('fixed {8,16,32,32,32}',[8,16,32,32,32])]:
    a=gen(256,strides); N=min(len(a),len(reg))
    scr=1/(1+np.exp(-np.clip(cls[:N],-100,100)))
    idx=np.where(scr>=0.5)[0]
    res=[]
    for i in idx[np.argsort(-scr[idx])][:3]:
      xc=a[i,0]+reg[i,0]/256; yc=a[i,1]+reg[i,1]/256; w=reg[i,2]/256
      res.append((round(float(scr[i]),3), int((xc*256-px)/sc), int((yc*256-py)/sc), int(w*256/sc)))
    print(img,name,'anchors',len(a),'model',len(reg),'hits',len(idx),'top(score,cx,cy,w px)',res)
