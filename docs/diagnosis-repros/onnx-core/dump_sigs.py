import sys, glob, os
sys.path.insert(0, "/mnt/sdb2/home/jcelerier/projets/ossia/score-master/src/addons/score-addon-onnx/tests")
import classify_real_models as c
files = sorted(f for d in ["/mnt/sdd1/models/wild","/mnt/sdd1/models/wild2","/mnt/sdd1/models/pinto"] for f in glob.glob(d+"/*.onnx"))
with open("/tmp/claude-1000/-mnt-sdb2-home-jcelerier-projets-ossia-score-master-src-addons-score-addon-onnx/a3ff42b7-59b7-418e-8dc2-baba3a54c9a3/scratchpad/diag/onnx-core/sigs_all.txt","w") as o:
    for f in files:
        try:
            o.write(c.sig_of(f)); o.flush()
        except Exception as e:
            print("SKIP", f, e, file=sys.stderr)
