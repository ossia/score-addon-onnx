import onnxruntime as ort, numpy as np, time
from PIL import Image
ort.set_default_logger_severity(3)
P='/mnt/win2/models/models-presets/models/'
def stats(name,x):
    print(f'{name}: shape={x.shape} min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f} p1={np.percentile(x,1):.3f} p99={np.percentile(x,99):.3f} frac_outside[-1,1]={np.mean(np.abs(x)>1):.4f}')
def save(name,x,chw=True):
    im=x[0].transpose(1,2,0) if chw else x[0]
    Image.fromarray(np.clip((im+1)*127.5,0,255).astype(np.uint8)).save(name+'_denorm.png')
    mn,mx=im.min(),im.max(); Image.fromarray(((im-mn)/(mx-mn)*255).astype(np.uint8)).save(name+'_minmax.png')
rng=np.random.default_rng(42)
# IG1 EigenGAN
e=ort.InferenceSession('/mnt/win2/PINTO_model_zoo/161_EigenGAN-Tensorflow/saved_model_Anime/model_float32.onnx',providers=['CPUExecutionProvider'])
try: e.run(None,{'eps':rng.standard_normal((1,512)).astype(np.float32)})
except Exception as ex: print('EigenGAN input0-only:',str(ex)[:160])
feed={i.name:rng.standard_normal(i.shape).astype(np.float32) for i in e.get_inputs()}
x=e.run(None,feed)[0]; stats('EigenGAN all-normal',x); save('eigengan',x,chw=False)
feed0={k:(v if k=='eps' else np.zeros_like(v)) for k,v in feed.items()}
x0=e.run(None,feed0)[0]; stats('EigenGAN z_*=0',x0); print('  diff vs all-normal mean abs', np.abs(x-x0).mean())
# StyleGAN2
s=ort.InferenceSession(P+'image-generator/stylegan2-ffhq-1024.onnx',providers=['CPUExecutionProvider'])
t=time.time(); x=s.run(None,{'input0':rng.standard_normal((1,512)).astype(np.float32)})[0]; print('sg2 time',time.time()-t); stats('StyleGAN2',x); save('stylegan2',x)
# IG2 mapping -> e4e
m=ort.InferenceSession(P+'sequence-processor/mobilestylegan-ffhq-mapping.onnx',providers=['CPUExecutionProvider'])
w=m.run(None,{'var':rng.standard_normal((1,512)).astype(np.float32)})[0]; stats('mapping w',w)
d=ort.InferenceSession(P+'image-generator/e4e-ffhq-decoder-wplus.onnx',providers=['CPUExecutionProvider'])
try: d.run(None,{'latent':w})
except Exception as ex: print('e4e fed [1,512]:',str(ex)[:160])
wp=np.repeat(w[:,None,:],18,axis=1)
x=d.run(None,{'latent':wp})[0]; stats('e4e(broadcast w)',x); save('e4e_broadcast',x)
x=d.run(None,{'latent':rng.standard_normal((1,18,512)).astype(np.float32)})[0]; stats('e4e(iid N(0,1) w+ = preset path)',x); save('e4e_iid',x)
