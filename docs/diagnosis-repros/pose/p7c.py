exec(open('p7.py').read().split('img=load')[0])
mf=sess('/home/jcelerier/Documents/ossia/score/packages/pose-detector/landmarks/lm-face-mobilefacenet-68.onnx')
def run_mf(im,d,W,H):
  x0,y0=d['kps'][0][0]*W,d['kps'][0][1]*H; x1,y1=d['kps'][1][0]*W,d['kps'][1][1]*H
  ang=-math.atan2(-(y1-y0),x1-x0); size=max(d['w']*W,d['h']*H)*1.1; cx,cy=d['xc']*W,d['yc']*H
  c,s=math.cos(ang),math.sin(ang); ax=size/112
  A=np.array([[c*ax,-s*ax,cx-.5*c*size+.5*s*size],[s*ax,c*ax,cy-.5*s*size-.5*c*size]],np.float32)
  crop=cv2.warpAffine(im,A,(112,112),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  t=((crop[:,:,::-1].astype(np.float32)/255.-[0.485,0.456,0.406])/[0.229,0.224,0.225]).astype(np.float32).transpose(2,0,1)[None]
  lm=mf.run(None,{'input':t})[0].reshape(68,2)*112
  return (A[:,:2]@lm.T).T+A[:,2]
img=load('face'); H,W=img.shape[:2]
fm=None
for rot in [0,30,-45]:
  M=cv2.getRotationMatrix2D((W/2,H/2),rot,1.0); im=cv2.warpAffine(img,M,(W,H))
  pb=run_mf(im,blazeface(im),W,H); pr=run_mf(im,retina(im),W,H)
  iod=np.linalg.norm(pb[36]-pb[45])
  print('rot',rot,'MobileFaceNet mean |diff| px',round(float(np.linalg.norm(pb-pr,axis=1).mean()),1),'interocular(outer)',round(float(iod),1))
