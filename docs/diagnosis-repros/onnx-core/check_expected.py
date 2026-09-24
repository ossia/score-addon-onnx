import sys
sys.path.insert(0, "/mnt/sdb2/home/jcelerier/projets/ossia/score-master/src/addons/score-addon-onnx/tests")
from classify_real_models import EXPECTED
for fn in sys.argv[1:]:
    fails=0; n=0
    for line in open(fn):
        name=line.split()[0]; exp=EXPECTED.get(name)
        if not exp: continue
        n+=1
        kind=line.split("kind=")[1].split()[0]; st=line.split("stateful=")[1].split()[0]=="1"; pose=line.split("pose=")[1].split()[0]
        ek,es=exp[0],exp[1]; ep=exp[2] if len(exp)>2 else None
        if not ((ek is None or kind==ek) and (es is None or st==es) and (ep is None or pose==ep)):
            fails+=1; print(fn, "FAIL", name, kind, st, pose, exp)
    print(fn, n, "checked", fails, "fail")
