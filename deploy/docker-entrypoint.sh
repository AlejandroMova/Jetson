#!/bin/bash
# 1) Compiles libcustom_softmax_parser.so for current arch (arm64 on Jetson).
# 2) Patches the age-gender ONNX to use dynamic batch in the Flatten reshape.
#    The model was exported with a hardcoded batch size in the Reshape node.
#    The shape can appear as a Constant node (not just an initializer) — both
#    are patched here. Any value != -1 in position 0 of a [?, 512] reshape
#    that feeds a Reshape op is replaced with -1 (dynamic batch).
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

# Collect shape inputs of all Reshape nodes (second input = shape tensor)
reshape_inputs = set()
for node in model.graph.node:
    if node.op_type == "Reshape" and len(node.input) > 1:
        reshape_inputs.add(node.input[1])

fixed = 0

# Fix Constant nodes whose output feeds a Reshape shape input
for node in model.graph.node:
    if node.op_type == "Constant" and node.output[0] in reshape_inputs:
        for attr in node.attribute:
            if attr.name == "value":
                arr = numpy_helper.to_array(attr.t)
                if arr.dtype == np.int64 and arr.shape == (2,) and arr[1] == 512 and arr[0] != -1:
                    print(f"[entrypoint] Patching Constant node reshape {arr.tolist()} → [-1, 512]")
                    attr.t.CopyFrom(numpy_helper.from_array(np.array([-1, 512], dtype=np.int64)))
                    fixed += 1

# Fix named initializers whose name feeds a Reshape shape input
for init in list(model.graph.initializer):
    if init.name in reshape_inputs:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.int64 and arr.shape == (2,) and arr[1] == 512 and arr[0] != -1:
            print(f"[entrypoint] Patching initializer reshape {arr.tolist()} → [-1, 512]")
            new_init = numpy_helper.from_array(np.array([-1, 512], dtype=np.int64), name=init.name)
            model.graph.initializer.remove(init)
            model.graph.initializer.append(new_init)
            fixed += 1

if fixed:
    onnx.save(model, path)
    for eng in glob.glob("/nx_tech/models/resnet_age_gender_FB2/*.engine"):
        os.remove(eng)
        print(f"[entrypoint] Removed stale engine: {eng}")
    print(f"[entrypoint] Patched {fixed} reshape(s) — engine will rebuild on first run.")
else:
    print("[entrypoint] ONNX reshape already dynamic — no patch needed.")
PYEOF
fi

exec "$@"
