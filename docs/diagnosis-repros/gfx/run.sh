#!/usr/bin/env bash
# run.sh <scene.js> <outdir> [extra env...]
set -u; AP=--autoplay; [ -n "${NOAUTOPLAY:-}" ] && AP=
SCENE="$1"; OUT="$2"; shift 2
BIN=$HOME/ossia/score-workshop/build-developer/ossia-score
rm -rf "$OUT"; mkdir -p "$OUT"
for n in $(seq 130 160); do [ -e /tmp/.X11-unix/X$n ] && continue; Xvfb :$n -screen 0 1280x720x24 +extension GLX >/dev/null 2>&1 & XP=$!; sleep 1; DISP=:$n; break; done
CFG="$OUT/cfg"; mkdir -p $CFG/ossia; cp /tmp/mo-probe/config-home/ossia/score.conf $CFG/ossia/ 2>/dev/null || printf '[score_plugin_gfx]\nGraphicsApi=OpenGL\n' > $CFG/ossia/score.conf
{ echo "var OUT_DIR = \"$OUT\";"; cat "$SCENE"; } > $OUT/scene.js
(
flock -w 900 9 || exit 1
env -u DISPLAY XDG_CONFIG_HOME="$CFG" SCORE_AUDIO_BACKEND=dummy SCORE_DISABLE_AUDIOPLUGINS=1 SCORE_FORCE_OFFSCREEN_WINDOW=Window \
  DISPLAY=$DISP QT_QPA_PLATFORM=xcb __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe "$@" \
  timeout 60 $BIN --no-gui --no-restore --script $OUT/scene.js --wait 1 $AP > $OUT/run.log 2>&1 &
APP=$!
for _ in $(seq 1 60); do [ -s $OUT/ready.score ] && break; sleep 1; done
[ -n "${NOAUTOPLAY:-}" ] && { sleep 1; oscsend 127.0.0.1 6666 /play T; }
sleep 4
for i in 1 2 3; do oscsend 127.0.0.1 6666 /script s "Score.device('Window').grabTo('$OUT/g$i.png')"; sleep 1.5; done
oscsend 127.0.0.1 6666 /script s "finalizeRun()"; sleep 1
oscsend 127.0.0.1 6666 /stop; sleep 0.5; oscsend 127.0.0.1 6666 /exit s force
wait $APP; echo rc=$?
) 9>/tmp/score-harness.lock
kill $XP 2>/dev/null
python3 -c "
import sys,numpy as np; from PIL import Image
import glob
for f in sorted(glob.glob('$OUT/g*.png')):
  a=np.asarray(Image.open(f).convert('RGB')).astype(float); print(f.split('/')[-1], a.shape, 'mean', a.reshape(-1,3).mean(0).round(1), 'redfrac', ((a[...,0]>200)&(a[...,1]<50)).mean().round(3))
"
