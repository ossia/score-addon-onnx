var UUID_ISF = "74ca45ff-92c9-44a0-8f1a-754dea05ee1b";
var UUID_WINDOW = "5a181207-7d40-4ad8-814e-879fcdf8cc31";
var FLICKS_PER_MS = 705600;
function llog(m) { console.log("[probe] " + m); }
Score.createDevice("Window", UUID_WINDOW, {});
var s = Score.find("Scenario.1"); if (s) Score.remove(s);
var g_root = Score.rootInterval();
Score.setIntervalDuration(g_root, 600000 * FLICKS_PER_MS);
Score.setIntervalMaxDuration(g_root, 600000 * FLICKS_PER_MS);
function finalizeRun() { Score.saveAs(OUT_DIR + "/final.score"); }
var g1 = Score.createProcess(g_root, UUID_ISF, "/tmp/claude-1000/-mnt-sdb2-home-jcelerier-projets-ossia-score-master-src-addons-score-addon-onnx/a3ff42b7-59b7-418e-8dc2-baba3a54c9a3/scratchpad/diag/gfx/two.fs");
var g2 = Score.createProcess(g_root, UUID_ISF, "/tmp/claude-1000/-mnt-sdb2-home-jcelerier-projets-ossia-score-master-src-addons-score-addon-onnx/a3ff42b7-59b7-418e-8dc2-baba3a54c9a3/scratchpad/diag/gfx/scale.fs");
Score.setAddress(Score.outlet(g2, 0), "Window:/");
var inl = Score.inlet(g2, 0);
llog("fmt before " + inl.textureFormat);

if(!Score.createCable(Score.outlet(g1, 0), inl)) llog("cable fail");
Score.saveAs(OUT_DIR + "/ready.score");
