#!/bin/bash
# 1) Compiles libcustom_softmax_parser.so for current arch (arm64 on Jetson).
# 2) Patches the age-gender ONNX to use dynamic batch.
#    Root cause: the model was exported from PyTorch with a static batch=16
#    in the ONNX input/output tensor shapes. TRT cannot run with any other
#    batch size. Fix: set the first dim of input and output to symbolic
#    (dim_param="batch") so TRT treats it as dynamic.
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

path = "/nx_tech/models/resnet_age_gender_FB2/resnet18_finetuned_int8_fixed.onnx"
model = onnx.load(path)
fixed = 0

# The model was exported with input [16,3,224,224] and output [16,6].
# Change the batch dimension (dim[0]) from static 16 to symbolic "batch".
for tensor in list(model.graph.input) + list(model.graph.output):
    if tensor.type.tensor_type.HasField("shape") and tensor.type.tensor_type.shape.dim:
        dim = tensor.type.tensor_type.shape.dim[0]
        if dim.dim_value == 16:
            dim.ClearField("dim_value")
            dim.dim_param = "batch"
            print(f"[entrypoint] Fixed {tensor.name}: batch dim 16 → dynamic")
            fixed += 1

if fixed:
    onnx.save(model, path)
    for eng in glob.glob(os.path.dirname(path) + "/*.engine"):
        os.remove(eng)
        print(f"[entrypoint] Removed stale engine: {eng}")
    print(f"[entrypoint] Patched {fixed} tensor(s) — engine will rebuild on first run.")
else:
    print("[entrypoint] ONNX already has dynamic batch — no patch needed.")
PYEOF
fi

exec "$@"
