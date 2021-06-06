#!/bin/bash
rm -rf release
mkdir -p release

cp -rf MyProcess *.{hpp,cpp,txt,json} LICENSE release/

mv release score-addon-my-process
7z a score-addon-my-process.zip score-addon-my-process
