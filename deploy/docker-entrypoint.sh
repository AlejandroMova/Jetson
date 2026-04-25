#!/bin/bash
# 1) Compiles libcustom_softmax_parser.so for current arch (arm64 on Jetson).
# 2) Patches the age-gender ONNX to use dynamic batch in the Flatten reshape.
#    The ONNX was exported with a hardcoded batch=16 in the Reshape/Flatten layer;
#    TRT fails when fewer than 16 crops are queued. Patching [16,512] → [-1,512]
#    lets TRT handle any batch size from 1 to 16.
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

ONNX_PATH="$PARSER_DIR/resnet18_finetuned_int8_fixed.onnx"
if [[ -f "$ONNX_PATH" ]]; then
    python3 - <<'PYEOF'
import onnx, os, glob
from onnx import numpy_helper
import numpy as np

path = "/nx_tech/models/resnet_age_gender_FB2/resnet18_finetuned_int8_fixed.onnx"
model = onnx.load(path)
fixed = False
for init in list(model.graph.initializer):
    arr = numpy_helper.to_array(init)
    if arr.dtype == np.int64 and arr.shape == (2,) and arr[0] == 16 and arr[1] == 512:
        print(f"[entrypoint] Patching static reshape {arr.tolist()} → [-1, 512]")
        new_init = numpy_helper.from_array(np.array([-1, 512], dtype=np.int64), name=init.name)
        model.graph.initializer.remove(init)
        model.graph.initializer.append(new_init)
        fixed = True
        break

if fixed:
    onnx.save(model, path)
    for eng in glob.glob("/nx_tech/models/resnet_age_gender_FB2/*.engine"):
        os.remove(eng)
        print(f"[entrypoint] Removed stale engine: {eng}")
    print("[entrypoint] ONNX patched — engine will rebuild on first run.")
else:
    print("[entrypoint] ONNX already has dynamic batch — no patch needed.")
PYEOF
fi

exec "$@"
