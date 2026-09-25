#!/bin/bash
D=$(dirname "$0"); name=G
BIN=$HOME/ossia/score-workshop/build-developer/ossia-score
CFG=$D/cfg-$name; mkdir -p $CFG
(
 flock -w 1800 9 || exit 98
 t0=$(date +%s.%N)
 env XDG_CONFIG_HOME=$CFG SCORE_AUDIO_BACKEND=dummy SCORE_DISABLE_AUDIOPLUGINS=1 QT_QPA_PLATFORM=offscreen \
   timeout --foreground 90 $BIN --no-gui --no-restore --script $D/x7.js --wait 1 > $D/$name.log 2>&1 &
 APP=$!
 sleep 15; oscsend 127.0.0.1 6666 /play T; echo "play at $(echo "$(date +%s.%N) - $t0"|bc)"
 sleep 25; oscsend 127.0.0.1 6666 /script s "Score.setIntervalDuration(Score.rootInterval(), 700000*705600)"; echo "resize at $(echo "$(date +%s.%N) - $t0"|bc)"
 wait $APP; rc=$?
 echo "$name rc=$rc elapsed=$(echo "$(date +%s.%N) - $t0"|bc)"
) 9>/tmp/score-harness.lock
