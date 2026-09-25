import sys; sys.path.insert(0,'.')
from rep import *
exec(open('p4.py').read().split("p='")[0].split('from rep import *')[1])
s=sess('/mnt/sdd1/models/wild2/pose2__blazepalm__blazepalm.onnx')
im=cv2.imread('/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/hand.jpg')
cur=gen(256,[8,16,16,16]); fix=gen(256,[8,16,32,32,32])
for half in [55,45,35,28,22,18]:
  c=im[max(0,203-half):203+half, max(0,83-half):83+half]
  c=cv2.resize(c,(256,256))
  t,(sc,px,py)=letterbox(c,256,256,True,[0,0,0],[255]*3,order='rgb')
  outs=s.run(None,{s.get_inputs()[0].name:t})
  reg=[o for o in outs if o.shape[-1]==18][0][0]; cls=[o for o in outs if o.shape[-1]==1][0][0,:,0]
  scr=1/(1+np.exp(-np.clip(cls,-80,80))); i=int(np.argmax(scr))
  f=lambda a:(round(float((a[i,0]+reg[i,0]/256)*256),1), round(float((a[i,1]+reg[i,1]/256)*256),1))
  print('crop half',half,'best idx',i,'score',round(float(scr[i]),3),'hits>=.5',int((scr>=.5).sum()),'hits idx>=2048',int((scr[2048:]>=.5).sum()),'center current',f(cur),'fixed',f(fix))
