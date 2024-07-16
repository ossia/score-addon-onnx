#!/bin/bash
rm -rf release
mkdir -p release

cp -rf Onnx *.{hpp,cpp,txt,json} LICENSE release/

mv release score-addon-onnx
7z a score-addon-onnx.zip score-addon-onnx
