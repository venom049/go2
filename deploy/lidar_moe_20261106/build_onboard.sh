#!/usr/bin/env bash
# Build only. No DDS initialization or robot command.
set -euo pipefail
ROOT=/home/unitree/go2-lidar-moe-20261106
cd "$ROOT"
sha256sum --check --quiet SHA256SUMS
mkdir -p controller/thirdparty run logs
chmod 700 run
ORT=controller/thirdparty/onnxruntime-linux-aarch64-1.22.0
if [[ ! -e $ORT ]];then
 ln -s /home/unitree/go2-original-bounded-20261106/controller/thirdparty/onnxruntime-linux-aarch64-1.22.0 "$ORT"
fi
[[ -f $ORT/lib/libonnxruntime.so.1.22.0 ]] || { echo 'Missing pinned ONNX Runtime 1.22.0';exit 1; }
cmake -S controller/robots/go2 -B controller/robots/go2/build -DUNITREE_SDK2_DIR=/home/unitree/go2_sdk2_new -DUNITREE_DDS_DIR=/usr/local -DCMAKE_BUILD_TYPE=Release
cmake --build controller/robots/go2/build -j2
sha256sum controller/robots/go2/build/go2_ctrl > BINARY_SHA256SUMS
echo 'Build complete. No motion controller has been started.'
