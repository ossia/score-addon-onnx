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
