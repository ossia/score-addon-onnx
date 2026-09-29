#!/usr/bin/env bash
# Multi-outlet texture validation.
#
#   multi-outlet.sh [ossia-score]
#
# A headless llvmpipe app run plays multi-outlet.js: an Images process feeds an
# Image Processor with a 2-output model (output 0 depth, output 1 sky), and a
# "Grid 2x2" ISF tiles the source and the processor's Image / Mask / Depth
# outlets onto Window:/. Frames are grabbed over OSC /script until the model
# has run, and analyze.py asserts that each outlet shows its own texture.
#
# It runs twice: with the processor -> grid cables in the document from the
# start ("init"), and with them -- and the Grid preset -- applied over OSC while
# playing ("live"), since those two cases go through different code paths.
#
# Environment:
#   OSSIA_SCORE          binary, when not given as the argument
#   MULTI_OUTLET_IMAGE   test photo with sky        (default: fixture/scene.png)
#   MULTI_OUTLET_MODEL   a model with the I/O of Depth Anything 3 metric:
#                        input 1x3x280x504, outputs depth + sky
#                        (default: fixture/two_outputs.onnx, see make_fixture.py)
#   OUT                  output directory            (default: a fresh temp dir)
#   SCORE_TESTS_COMMON   score's tests/integration/common (default: found from
#                        this add-on's place in score's src/addons)
#
# Rendering: our own Xvfb, xcb, Mesa llvmpipe forced. ONNX Runtime is pinned
# to the CPU provider.
#
# PASS = both runs exit 0, no JS SCENARIO-ERROR, analyze.py green on both.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${1:-${OSSIA_SCORE:-}}"
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/multi-outlet.XXXXXX")}"
IMAGE="${MULTI_OUTLET_IMAGE:-$HERE/fixture/scene.png}"
MODEL="${MULTI_OUTLET_MODEL:-$HERE/fixture/two_outputs.onnx}"
echo "model: $MODEL"
echo "image: $IMAGE"
TIMEOUT="${TIMEOUT:-420}"
GRABS="${GRABS:-40}"

command -v oscsend >/dev/null || { echo "SKIP: oscsend not found"; exit 77; }
PORTS_SH="${SCORE_TESTS_COMMON:-$HERE/../../../../../tests/integration/common}/control-ports.sh"
[ -f "$PORTS_SH" ] || { echo "SKIP: $PORTS_SH not found"; exit 77; }
. "$PORTS_SH"
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "SKIP: ossia-score not built (${BIN:-unset})"; exit 77; }
[ -f "$IMAGE" ] || { echo "SKIP: test image $IMAGE missing"; exit 77; }
[ -f "$MODEL" ] || { echo "SKIP: model $MODEL missing"; exit 77; }
python3 -c "import numpy, PIL, onnxruntime" 2>/dev/null \
  || { echo "SKIP: python numpy/PIL/onnxruntime missing"; exit 77; }

rm -rf "$OUT"; mkdir -p "$OUT"

# ---- a real X server, ours if we can have one -------------------------------
XVFB_PID=""
DISP=""
if command -v Xvfb >/dev/null && command -v xdpyinfo >/dev/null; then
  for n in $(seq 90 120); do
    [ -e "/tmp/.X11-unix/X$n" ] && continue
    Xvfb ":$n" -screen 0 1920x1080x24 +extension GLX >"$OUT/xvfb.log" 2>&1 &
    XVFB_PID=$!
    for _ in $(seq 1 20); do
      DISPLAY=":$n" xdpyinfo >/dev/null 2>&1 && { DISP=":$n"; break; }
      kill -0 "$XVFB_PID" 2>/dev/null || break
      sleep 0.25
    done
    [ -n "$DISP" ] && break
    kill "$XVFB_PID" 2>/dev/null; wait "$XVFB_PID" 2>/dev/null; XVFB_PID=""
  done
fi
if [ -z "$DISP" ] && [ -n "${DISPLAY:-}" ]; then DISP="$DISPLAY"; fi
[ -n "$DISP" ] || { echo "SKIP: no X display"; exit 77; }
cleanup_xvfb() { [ -n "$XVFB_PID" ] && { kill "$XVFB_PID" 2>/dev/null; wait "$XVFB_PID" 2>/dev/null; }; }
trap cleanup_xvfb EXIT
echo "display=$DISP$([ -n "$XVFB_PID" ] && echo ' (own Xvfb)' || echo ' (inherited)')"

# Hermetic config home: a fresh one with only GraphicsApi pinned to OpenGL.
# Nothing is read from or written to the user's own configuration.
CFG="$OUT/config-home"; rm -rf "$CFG"; mkdir -p "$CFG/ossia"
printf '[score_plugin_gfx]\nGraphicsApi=OpenGL\n' > "$CFG/ossia/score.conf"

jsstr() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"; }

send() { oscsend 127.0.0.1 "$OSC" "$@" 2>/dev/null; }

grab() { # png -> 0 iff file written
  local png="$1"
  rm -f "$png"
  for _ in $(seq 1 12); do
    send /script s "Score.device('Window').grabTo('$png')"
    sleep 0.6; [ -s "$png" ] && return 0
  done
  return 1
}

# The model runs on the worker thread: grab until the Mask and Depth tiles have
# content (the thresholds analyze.py asserts).
outlets_ready() {
  python3 - "$1" <<'PYEOF'
import sys, numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(np.float32) / 255
h, w, _ = a.shape
my, mx = h // 20, w // 20
bl = a[h // 2 + my: h - my, mx: w // 2 - mx, 0]
br = a[h // 2 + my: h - my, w // 2 + mx: w - mx, 0]
sys.exit(0 if bl.std() > 0.05 and br.mean() > 0.1 else 1)
PYEOF
}

# run_mode <init|live>: one app run, grabs to $OUT/<mode>/grid.png.
run_mode() {
  local mode="$1" dir="$OUT/$1"
  mkdir -p "$dir"
  {
    printf 'var OUT_DIR = %s;\n' "$(jsstr "$dir")"
    printf 'var IMAGE = %s;\n' "$(jsstr "$IMAGE")"
    printf 'var MODEL = %s;\n' "$(jsstr "$MODEL")"
    printf 'var GRID_PRESET = '; cat "$HERE/Grid 2x2.scp"; printf ';\n'
    printf 'var WIRING = %s;\n' "$(jsstr "$mode")"
    cat "$HERE/multi-outlet.js"
  } > "$dir/scene.js"

  rm -f "$CFG/ossia/failsafe.bit"
  (
    # A previous run still shutting down may still write into $OUT.
    for _ in $(seq 1 60); do
      pgrep -f -- "--script $OUT/" >/dev/null 2>&1 || break
      sleep 1
    done
    pick_control_ports || { echo 97 > "$dir/run.rc"; exit 0; }
    env -u DISPLAY XDG_CONFIG_HOME="$CFG" \
        SCORE_AUDIO_BACKEND=dummy SCORE_DISABLE_AUDIOPLUGINS=1 SCORE_ONNX_FORCE_PROVIDER=cpu \
        SCORE_FORCE_OFFSCREEN_WINDOW=Window \
        SCORE_LOCAL_OSC_PORT="$OSC" SCORE_LOCAL_WS_PORT="$WS" \
        DISPLAY="$DISP" QT_QPA_PLATFORM=xcb \
        __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
        QT_LOGGING_RULES='qt.rhi.general=true' \
      timeout --foreground "$TIMEOUT" "$BIN" --no-gui --no-restore \
        --script "$dir/scene.js" --wait 1 --autoplay >"$dir/run.log" 2>&1 &
    local APP=$!

    local ok=0
    for _ in $(seq 1 120); do [ -s "$dir/multi-outlet-init.score" ] && { ok=1; break; }; sleep 1; done
    if [ "$ok" = 0 ]; then
      echo "[$mode] no readiness marker -- startup failed (see $dir/run.log)" >&2
      kill "$APP" 2>/dev/null; wait "$APP" 2>/dev/null; echo 97 > "$dir/run.rc"; exit 0
    fi
    if [ "$mode" = live ]; then
      # The cables must reach a running graph: wait until the Window renders.
      grab "$dir/grid.png" || echo "GRAB-FAIL before wiring" >> "$dir/run.log"
      send /script s "wireOutlets()"
    fi

    local i
    for i in $(seq 1 "$GRABS"); do
      grab "$dir/grid.png" || { echo "GRAB-FAIL $i" >> "$dir/run.log"; continue; }
      if outlets_ready "$dir/grid.png"; then
        echo "[$mode] grid ready after $i grabs"
        break
      fi
      sleep 1
    done

    send /script s "finalizeRun()"
    for _ in $(seq 1 20); do [ -s "$dir/multi-outlet-final.score" ] && break; sleep 0.5; done
    send /stop; sleep 0.5
    send /exit s force
    wait "$APP"; echo $? > "$dir/run.rc"
  )
}

check_mode() { # mode -> appends to $FAILS
  local mode="$1" dir="$OUT/$1" rc renderer
  echo "=== $mode wiring"
  rc=$(cat "$dir/run.rc" 2>/dev/null || echo 97)
  [ "$rc" = 0 ] || FAILS+=" $mode:exit=$rc"
  if grep -q "NULL RHI BACKEND" "$dir/run.log" 2>/dev/null; then
    FAILS+=" $mode:NULL-RHI"
  else
    # Some builds never print the qt.rhi.general line (text-render has the same
    # NO-RENDERER-LINE): only a renderer that IS reported and is not llvmpipe
    # fails. The Null backend is caught above, and would also fail analyze.py
    # (it fills every texture with yellow).
    renderer=$(grep -m1 'qt\.rhi\.general: OpenGL VENDOR' "$dir/run.log" 2>/dev/null | sed 's/^.*qt\.rhi\.general: //')
    echo "backend: ${renderer:-not reported}"
    if [ -n "$renderer" ] && ! printf '%s' "$renderer" | grep -q llvmpipe; then
      FAILS+=" $mode:WRONG-BACKEND"
    fi
  fi
  grep -q "SCENARIO-ERROR" "$dir/run.log" 2>/dev/null && FAILS+=" $mode:JSERR"
  if [ -s "$dir/grid.png" ]; then
    python3 "$HERE/analyze.py" "$dir/grid.png" "$IMAGE" "$MODEL" || FAILS+=" $mode:ANALYZE"
  else
    FAILS+=" $mode:NO-GRAB"
  fi
}

FAILS=""
for mode in init live; do
  run_mode "$mode"
  check_mode "$mode"
done

if [ -z "$FAILS" ]; then
  echo "multi-outlet PASS (grabs: $OUT/init/grid.png $OUT/live/grid.png)"
else
  echo "multi-outlet FAIL:$FAILS  (out=$OUT)"; exit 1
fi
