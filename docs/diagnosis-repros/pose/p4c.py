import sys; sys.path.insert(0,'.')
from rep import *
exec(open('p4.py').read().split("p='")[0].split('from rep import *')[1])
s=sess('/mnt/sdd1/models/wild2/pose2__blazepalm__blazepalm.onnx')
im=cv2.imread('/home/jcelerier/projets/oss/ailia-models/hand_recognition/blazehand/person_hand.jpg'); print(im.shape)
cur=gen(256,[8,16,16,16]); fix=gen(256,[8,16,32,32,32])
H,W=im.shape[:2]
for frac in [1.0,0.7,0.5,0.35]:
 for (cy,cx) in [(H//2,W//2)]:
  h2=int(H*frac/2); w2=int(W*frac/2)
  c=im[cy-h2:cy+h2, cx-w2:cx+w2]
  t,(sc,px,py)=letterbox(c,256,256,True,[0,0,0],[255]*3,order='rgb')
  outs=s.run(None,{s.get_inputs()[0].name:t})
  reg=[o for o in outs if o.shape[-1]==18][0][0]; cls=[o for o in outs if o.shape[-1]==1][0][0,:,0]
  scr=1/(1+np.exp(-np.clip(cls,-80,80)))
  hits=np.where(scr>=.5)[0]
  for i in hits[np.argsort(-scr[hits])][:3]:
    f=lambda a:(round(float((a[i,0]+reg[i,0]/256)*256),1), round(float((a[i,1]+reg[i,1]/256)*256),1))
    print('frac',frac,'idx',i,'score',round(float(scr[i]),3),'w',round(float(reg[i,2]),1),'center(model px) current',f(cur),'fixed',f(fix))
