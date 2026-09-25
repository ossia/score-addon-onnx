#!/bin/bash
# usage: run.sh <name> <autoplay:0|1> <play_osc_delay or -> <timeout>
D=$(dirname "$0"); name=$1; ap=$2; pdelay=$3; to=$4
BIN=$HOME/ossia/score-workshop/build-developer/ossia-score
CFG=$D/cfg-$name; mkdir -p $CFG
args=(--no-gui --no-restore --script $D/${SCRIPT:-x7.js} --wait 1)
[ "$ap" = 1 ] && args+=(--autoplay)
(
 flock -w 1800 9 || exit 98
 t0=$(date +%s.%N)
 env XDG_CONFIG_HOME=$CFG SCORE_AUDIO_BACKEND=dummy SCORE_DISABLE_AUDIOPLUGINS=1 QT_QPA_PLATFORM=offscreen \
   timeout --foreground $to $BIN "${args[@]}" > $D/$name.log 2>&1 &
 APP=$!
 if [ "$pdelay" != - ]; then sleep $pdelay; tp=$(date +%s.%N); oscsend 127.0.0.1 6666 /play T; echo "sent /play at $(echo "$tp - $t0"|bc)"; fi
 wait $APP; rc=$?
 t1=$(date +%s.%N)
 echo "$name rc=$rc elapsed=$(echo "$t1 - $t0"|bc)"
 if [ $rc = 124 ]; then :; fi
 oscsend 127.0.0.1 6666 /exit s force 2>/dev/null
) 9>/tmp/score-harness.lock
