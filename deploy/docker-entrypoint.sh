#!/bin/bash
# Compiles libcustom_softmax_parser.so for the current architecture (arm64 on Jetson).
# The pre-existing .so in the repo is x86_64 and will segfault on Jetson.
set -e

PARSER_DIR="/nx_tech/models/resnet_age_gender_FB2"
PARSER_SRC="$PARSER_DIR/custom_softmax_parser.cpp"
PARSER_OUT="$PARSER_DIR/libcustom_softmax_parser.so"
DS_INCLUDES="/opt/nvidia/deepstream/deepstream/sources/includes"

if [[ -f "$PARSER_SRC" ]]; then
    echo "[entrypoint] Compiling custom softmax parser for $(uname -m)..."
    g++ -shared -fPIC \
        -o "$PARSER_OUT" \
        "$PARSER_SRC" \
        -I"$DS_INCLUDES" \
        -std=c++14 -O2
    echo "[entrypoint] Parser compiled: $PARSER_OUT"
else
    echo "[entrypoint] Parser source not found — skipping compilation (no models volume mounted)"
fi

exec "$@"
