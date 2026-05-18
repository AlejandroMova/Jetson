"""
streamlit_app.py — NX Computing AI | QA Visual Dashboard

Dashboard de inspección remota para Jetsons desplegados.
Accesible vía Tailscale en http://<jetson-tailscale-ip>:8501

Panels:
  - Video en vivo con bboxes (MJPEG leído por Python desde deepstream:8080)
  - Detecciones en tiempo real (Redis pub/sub nx:qa:detections)
  - API Calls log (Redis pub/sub nx:qa:apicalls)
  - Toggles de capacidades (Redis hash nx:qa:capabilities)
  - Selector de cámara (todas / individual)

Arrancar con: ./qa.sh  (desde deploy/)
"""

import json
import os
import threading
import time
from collections import deque

import redis
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST  = os.getenv("REDIS_HOST", "redis")
MJPEG_PORT  = int(os.getenv("MJPEG_PORT", "8080"))

# Hostname del Jetson tal como lo ve el browser (Tailscale IP o LAN IP).
# st.context.headers["host"] = "100.67.192.58:8501" → extraemos solo la IP.
try:
    _host_hdr = st.context.headers.get("host", "")
    MJPEG_HOST = _host_hdr.split(":")[0] if _host_hdr else "localhost"
except Exception:
    MJPEG_HOST = "localhost"
MAX_DETECTIONS = 200
MAX_APICALLS   = 50

st.set_page_config(
    page_title="NX QA Visual",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Autorefresh solo para detecciones y API calls — el video lo maneja el browser
st_autorefresh(interval=500, limit=None, key="qa_tick")


# ── Redis helpers ─────────────────────────────────────────────────────────────
@st.cache_resource
def _get_redis():
    try:
        r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True,
                        socket_connect_timeout=2, socket_timeout=2)
        r.ping()
        return r
    except Exception:
        return None


def _redis_ok() -> bool:
    r = _get_redis()
    if r is None:
        return False
    try:
        r.ping()
        return True
    except Exception:
        return False


# ── Buffers de proceso — compartidos entre rerenders, sin ScriptRunContext ────
# cache_resource crea un singleton a nivel de proceso: el subscriber escribe
# aquí y el script lo lee en cada rerender sin problemas de thread safety.
@st.cache_resource
def _get_buffers():
    return {
        "detections": deque(maxlen=MAX_DETECTIONS),
        "apicalls":   deque(maxlen=MAX_APICALLS),
    }

@st.cache_resource
def _ensure_subscriber(_host=REDIS_HOST):
    """Arranca el subscriber UNA VEZ por proceso. Auto-reconecta si Redis cae."""
    bufs = _get_buffers()

    def _loop():
        while True:
            try:
                r = redis.Redis(host=_host, port=6379, decode_responses=True)
                p = r.pubsub()
                p.subscribe("nx:qa:detections", "nx:qa:apicalls")
                for msg in p.listen():
                    if msg["type"] != "message":
                        continue
                    try:
                        data = json.loads(msg["data"])
                    except Exception:
                        continue
                    ch = msg["channel"]
                    if "detections" in ch:
                        bufs["detections"].appendleft(data)
                    elif "apicalls" in ch:
                        bufs["apicalls"].appendleft(data)
            except Exception:
                time.sleep(2)   # reconectar tras error

    threading.Thread(target=_loop, daemon=True, name="qa-subscriber").start()
    return True


_ensure_subscriber()
_bufs = _get_buffers()


# ── Leer status del pipeline desde Redis ──────────────────────────────────────
_r = _get_redis()
status: dict = {}
if _r:
    try:
        raw = _r.get("nx:qa:status")
        if raw:
            status = json.loads(raw)
    except Exception:
        pass

active_caps: list = status.get("capabilities", [])
channels: list    = status.get("channels", [])


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📹 NX QA Visual")
    st.markdown("---")

    if status:
        st.markdown(f"**Cliente:** {status.get('client', '—')}")
        st.markdown(f"**Paquete:** {status.get('package', '—')}")
        st.markdown(f"**Sector:** {status.get('sector', '—')}")
        st.markdown(f"**Canales activos:** {len(channels)}")
        st.markdown(f"**Tracker:** {status.get('tracker', '—')}")
    else:
        st.warning("Pipeline no disponible o NX_QA_ENABLED no activo.")

    st.markdown("---")

    # Selector de cámara
    cam_label_map: dict = {"Todas las cámaras": "all"}
    jetson_id = status.get("jetson_id", "")
    for ch in channels:
        cam_id = f"{jetson_id}-ch{ch:02d}" if jetson_id else f"ch{ch:02d}"
        cam_label_map[f"Cámara {ch}"] = cam_id

    selected_label = st.selectbox(
        "📹 Vista de cámara",
        options=list(cam_label_map.keys()),
        index=0,
    )
    stream_key = cam_label_map[selected_label]

    st.markdown("---")

    # Toggles de capacidades
    IMPLEMENTED = [
        "people_counting", "age_gender", "fall_detection", "face_recognition",
    ]
    UNIMPLEMENTED = ["epp_detection", "fire_smoke", "license_plate"]

    st.markdown("**🔧 Capacidades**")
    # people_counting siempre activo
    st.checkbox("people_counting", value=True, disabled=True,
                help="Siempre activo — no se puede apagar", key="cap_people_counting")

    # Capacidades implementadas — todas toggleables (las no activas en pipeline
    # aplican solo a workers Python que chequean Redis; SGIEs requieren reinicio)
    for cap in IMPLEMENTED[1:]:
        in_pipeline = cap in active_caps
        help_text = None if in_pipeline else "No activo en este paquete — activar reinicia el pipeline"
        if _r:
            try:
                current_val = _r.hget("nx:qa:capabilities", cap)
                is_on = (current_val is None and in_pipeline) or current_val == "1"
                new_val = st.checkbox(cap, value=is_on, key=f"cap_{cap}", help=help_text)
                _r.hset("nx:qa:capabilities", cap, "1" if new_val else "0")
            except Exception:
                st.checkbox(cap, value=in_pipeline, disabled=True, key=f"cap_{cap}_err")
        else:
            st.checkbox(cap, value=in_pipeline, disabled=True, key=f"cap_{cap}_nr")

    # Capacidades pendientes de implementar
    for cap in UNIMPLEMENTED:
        st.checkbox(f"{cap} (pendiente)", value=False, disabled=True, key=f"cap_{cap}_pending")

    st.markdown("---")

    if _redis_ok():
        st.success("● Redis conectado")
    else:
        st.error("● Redis desconectado")
        st.caption("Verifica que el pipeline esté corriendo con `./qa.sh`")


# ── MAIN — Video + Detecciones ────────────────────────────────────────────────
col_video, col_det = st.columns([55, 45])

with col_video:
    st.markdown("### 📹 Video en vivo")
    stream_label = "Todas las cámaras (tiled)" if stream_key == "all" else stream_key
    # st.iframe preserva el iframe entre rerenders si el src no cambia
    # → sin interrupciones del stream. El HTML lo sirve el propio MjpegServer
    # desde /viewer/<key> con un <img src="/stream/<key>"> mismo-origen.
    viewer_url = f"http://{MJPEG_HOST}:{MJPEG_PORT}/viewer/{stream_key}"
    st.iframe(viewer_url, height=560)
    st.caption(f"Stream: `{stream_label}` · 640×360 · MJPEG nativo")

with col_det:
    st.markdown("### 📊 Detecciones")
    det_items = list(_bufs["detections"])
    if not det_items:
        st.info("Sin detecciones aún. El pipeline debe estar activo.")
    else:
        with st.container(height=360):
            for d in det_items[:60]:
                ts     = (d.get("ts") or "")[-12:-4]
                cam    = d.get("cam", "")
                tracks = d.get("tracks", [])
                if not tracks:
                    continue
                cam_short = cam.split("-ch")[-1] if "-ch" in cam else cam
                st.markdown(f"**{ts}** · `ch{cam_short}`")
                for t in tracks:
                    icon  = "⚠️" if t.get("fall") else "👤"
                    label = t.get("label", f"P#{t.get('track_id', '?')}")
                    conf  = t.get("confidence", 0)
                    st.markdown(
                        f"&nbsp;&nbsp;{icon} `{label}` &nbsp;conf={conf:.2f}",
                        unsafe_allow_html=True,
                    )


# ── API CALLS ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📨 API Calls")

api_items = list(_bufs["apicalls"])
if not api_items:
    st.info("Sin API calls aún.")
else:
    for call in api_items[:15]:
        ts      = (call.get("ts") or "")[-12:-4]
        ep      = call.get("endpoint", "/api/?")
        payload = call.get("payload", {})
        evt     = payload.get("event_type") or ep.rsplit("/", 1)[-1]
        cam_raw = payload.get("camera_id", "")
        cam_lbl = f"  ·  `{cam_raw}`" if cam_raw else ""
        with st.expander(f"`{ts}`  POST `{ep}`  **{evt}**{cam_lbl}"):
            st.json(payload)
