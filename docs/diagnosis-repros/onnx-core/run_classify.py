import sys, subprocess
sys.path.insert(0, "/mnt/sdb2/home/jcelerier/projets/ossia/score-master/src/addons/score-addon-onnx/tests")
import classify_real_models as c
files = sys.argv[1:]
sig = "".join(c.sig_of(f) for f in files)
if "-v" in sys.argv: pass
open("/tmp/claude-1000/-mnt-sdb2-home-jcelerier-projets-ossia-score-master-src-addons-score-addon-onnx/a3ff42b7-59b7-418e-8dc2-baba3a54c9a3/scratchpad/diag/onnx-core/sigs.txt","w").write(sig)
print(subprocess.run(["/tmp/claude-1000/-mnt-sdb2-home-jcelerier-projets-ossia-score-master-src-addons-score-addon-onnx/a3ff42b7-59b7-418e-8dc2-baba3a54c9a3/scratchpad/diag/onnx-core/classify_cli"], input=sig, capture_output=True, text=True).stdout)
