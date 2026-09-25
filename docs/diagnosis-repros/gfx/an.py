import sys,glob,numpy as np; from PIL import Image
for d in sys.argv[1:]:
  for f in sorted(glob.glob(d+'/g*.png')):
    a=np.asarray(Image.open(f).convert('RGB')).astype(float); print(f[len(d)-1:], a.shape, 'mean', a.reshape(-1,3).mean(0).round(1), 'redfrac', ((a[...,0]>200)&(a[...,1]<50)).mean().round(3))
