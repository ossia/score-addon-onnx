exec(open('p7.py').read().split('img=load')[0])
fm=sess('/home/jcelerier/Documents/ossia/score/packages/pose-detector/landmarks/lm-face-facemesh-468.onnx')
def run_fm(im,d,W,H):
  x0,y0=d['kps'][0][0]*W,d['kps'][0][1]*H; x1,y1=d['kps'][1][0]*W,d['kps'][1][1]*H
  ang=-math.atan2(-(y1-y0),x1-x0); size=max(d['w']*W,d['h']*H)*1.5; cx,cy=d['xc']*W,d['yc']*H
  c,s=math.cos(ang),math.sin(ang); ax=size/192
  A=np.array([[c*ax,-s*ax,cx-.5*c*size+.5*s*size],[s*ax,c*ax,cy-.5*s*size-.5*c*size]],np.float32)
  crop=cv2.warpAffine(im,A,(192,192),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  t=(crop[:,:,::-1].astype(np.float32)/255.)[None]
  o=fm.run(None,{fm.get_inputs()[0].name:t})
  lm=[x for x in o if x.size==1404][0].reshape(468,3); flag=[x for x in o if x.size==1][0].ravel()[0]
  pts=(A[:,:2]@lm[:,:2].T).T+A[:,2]
  return pts, float(1/(1+np.exp(-flag)))
img=load('face'); H,W=img.shape[:2]
for rot in [0,30,-45]:
  M=cv2.getRotationMatrix2D((W/2,H/2),rot,1.0); im=cv2.warpAffine(img,M,(W,H))
  pb,fb=run_fm(im,blazeface(im),W,H); pr,fr=run_fm(im,retina(im),W,H)
  iod=np.linalg.norm(pb[33]-pb[263])
  print('rot',rot,'flag bf/rf',round(fb,3),round(fr,3),'mean |diff| px',round(float(np.linalg.norm(pb-pr,axis=1).mean()),1),'interocular',round(float(iod),1))
