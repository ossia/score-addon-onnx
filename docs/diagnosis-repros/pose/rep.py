import onnxruntime as ort, numpy as np, cv2, sys
ort.set_default_logger_severity(3)
IMGS={'handcrop':'/tmp/claude-1000/-mnt-sdb2-home-jcelerier-projets-ossia-score-master-src-addons-score-addon-onnx/a3ff42b7-59b7-418e-8dc2-baba3a54c9a3/scratchpad/diag/pose/handcrop.png','bodyhandcrop':'/tmp/claude-1000/-mnt-sdb2-home-jcelerier-projets-ossia-score-master-src-addons-score-addon-onnx/a3ff42b7-59b7-418e-8dc2-baba3a54c9a3/scratchpad/diag/pose/bodyhandcrop.png','body':'/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/body.jpg',
      'demo1':'/home/jcelerier/projets/oss/ailia-models/depth_estimation/depth_anything/demo1.png',
      'hand':'/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/hand.jpg',
      'face':'/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/face.png'}
def load(name):
    bgr=cv2.imread(IMGS[name]); return bgr
def letterbox(bgr, mw, mh, center, mean, std, order='bgr', layout='nchw'):
    ih,iw=bgr.shape[:2]; sc=min(mw/iw, mh/ih)
    nw=max(1,round(iw*sc)); nh=max(1,round(ih*sc))
    px=(mw-nw)//2 if center else 0; py=(mh-nh)//2 if center else 0
    img=bgr if order=='bgr' else bgr[:,:,::-1]
    canvas=np.zeros((mh,mw,3),np.float32)
    canvas[py:py+nh,px:px+nw]=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    canvas=(canvas-np.array(mean,np.float32))/np.array(std,np.float32)
    t=canvas.transpose(2,0,1)[None] if layout=='nchw' else canvas[None]
    return np.ascontiguousarray(t), (sc,px,py)
def sess(p):
    return ort.InferenceSession(p,providers=['CPUExecutionProvider'])
