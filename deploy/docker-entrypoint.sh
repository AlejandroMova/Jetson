#!/bin/bash
# Compiles libcustom_softmax_parser.so for the current architecture (arm64 on Jetson).
# The pre-existing .so in the repo is x86_64 and will segfault on Jetson.
set -e

PARSER_SRC="/nx_tech/models/resnet_age_gender_FB2/custom_softmax_parser.cpp"
PARSER_OUT="/nx_tech/models/resnet_age_gender_FB2/libcustom_softmax_parser.so"
DS_INCLUDES="/opt/nvidia/deepstream/deepstream/sources/includes"

# Find TensorRT headers (NvCaffeParser.h) — location varies by JetPack version
TRT_INCLUDE=$(find /usr/include /usr/local/cuda/include -name "NvCaffeParser.h" \
    2>/dev/null -exec dirname {} \; | head -1 || echo "")

echo "[entrypoint] Compiling custom softmax parser for $(uname -m)..."
g++ -shared -fPIC \
    -o "$PARSER_OUT" \
    "$PARSER_SRC" \
    -I"$DS_INCLUDES" \
    ${TRT_INCLUDE:+-I"$TRT_INCLUDE"} \
    -std=c++14 -O2
echo "[entrypoint] Parser compiled: $PARSER_OUT"

exec "$@"
