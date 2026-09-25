import sys; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from rep import *
p='/mnt/sdd1/models/pinto/459__yolov9_n_wholebody25_post_0100_1x3x128x160.onnx'
s=sess(p)
for img in ['body','demo1']:
  bgr=load(img)
  for (mw,mh) in [(128,128),(160,128),(320,320),(640,480),(640,640)]:
    for norm in ['raw','unit']:
      t,(sc,px,py)=letterbox(bgr,mw,mh,False,[0,0,0],[1,1,1] if norm=='raw' else [255]*3)
      try: out=s.run(None,{s.get_inputs()[0].name:t})[0]
      except Exception as e: print(mw,mh,norm,'ERR',str(e)[:80]); continue
      c0=[r for r in out if r[2]>=0.4 and int(r[1]+.5)==0]
      print(img,mw,mh,norm,'rows',len(out),'class0>=0.4:',len(c0),[(round(float(r[2]),2),[int((r[3+i]-(px if i%2==0 else py))/sc) for i in range(4)]) for r in c0[:4]])
