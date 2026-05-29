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

# Pre-download InsightFace buffalo_l if face_recognition is active
NX_PIPELINE_VAL="${NX_PIPELINE:-$(cat /etc/nx_pipeline 2>/dev/null || echo '')}"
if echo "$NX_PIPELINE_VAL" | grep -q "face_recognition"; then
    INSIGHTFACE_ROOT="/nx_tech/models/insightface"
    MODEL_MARKER="${INSIGHTFACE_ROOT}/models/buffalo_l/w600k_r50.onnx"
    if [[ ! -f "$MODEL_MARKER" ]]; then
        echo "[entrypoint] Pre-descargando InsightFace buffalo_l..."
        python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', root='${INSIGHTFACE_ROOT}',
                   providers=['CUDAExecutionProvider','CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print('[entrypoint] InsightFace buffalo_l listo.')
"
    else
        echo "[entrypoint] InsightFace buffalo_l ya descargado — skip"
    fi
fi

# ── Main loop: live (app.py) o playback (app_video_testing.py) ───────────────
# Solo en QA mode. En producción (NX_QA_ENABLED no seteado) ejecuta el CMD directo.
if [ "${NX_QA_ENABLED:-false}" != "true" ]; then
    exec "$@"
fi

echo "[entrypoint] QA mode — loop live/playback activo"
while true; do
    # Verificar si Streamlit solicitó modo playback via Redis
    PLAYBACK_VIDEO=$(python3 - <<'PYEOF' 2>/dev/null
import os, sys
try:
    import redis
    r = redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), socket_timeout=2)
    v = r.get("nx:qa:playback_video")
    print(v.decode() if v else "")
except Exception:
    print("")
PYEOF
)

    if [ -n "$PLAYBACK_VIDEO" ]; then
        echo "[entrypoint] Modo playback: $PLAYBACK_VIDEO"

        # Read active capabilities and client from the last known pipeline status so
        # the playback pipeline runs the same models the live pipeline was using.
        # socket_timeout=2: conservative for container-local Redis; 0.5 s would also work.
        read -r PLAYBACK_CAPS PLAYBACK_CLIENT <<< "$(python3 - <<'PYEOF' 2>/dev/null
import os, json
try:
    import redis
    r = redis.Redis(
        host=os.environ.get("REDIS_HOST", "redis"),
        socket_timeout=2,
        decode_responses=True,
    )
    status = json.loads(r.get("nx:qa:status") or "{}")
    caps   = ",".join(status.get("capabilities", ["people_counting", "age_gender"]))
    client = status.get("client", "demo")
    print(caps, client)
except Exception:
    print("people_counting,age_gender demo")
PYEOF
)"
        echo "[entrypoint] Playback caps: $PLAYBACK_CAPS  client: $PLAYBACK_CLIENT"
        python3 /nx_tech/pipelines/app_video_testing.py \
            --input "$PLAYBACK_VIDEO" \
            --capabilities "$PLAYBACK_CAPS" \
            --client "$PLAYBACK_CLIENT" \
            --no-loop || true

        # Clear the key so the next loop iteration restarts in live mode.
        python3 - <<'PYEOF' 2>/dev/null
import os
try:
    import redis
    redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), socket_timeout=2).delete("nx:qa:playback_video")
except Exception:
    pass
PYEOF
        echo "[entrypoint] Playback terminado — volviendo a modo live"
    else
        # Modo live: arrancar pipeline principal
        "$@"
        EXIT_CODE=$?
        # Código 42 = solicitud de cambio a playback mode (app.py lo emite y loop continúa)
        # Cualquier otro código de error = salir
        if [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 42 ]; then
            echo "[entrypoint] Pipeline terminó con error (código $EXIT_CODE) — saliendo"
            exit $EXIT_CODE
        fi
    fi
done
