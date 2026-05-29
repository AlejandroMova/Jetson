"""
streamlit_app.py — NX Computing AI | QA Visual Dashboard

Remote inspection dashboard for deployed Jetsons.
Accessible via Tailscale at http://<jetson-tailscale-ip>:8501

Panels:
  - Live video with bounding boxes (MJPEG from deepstream:8080 via st.iframe)
  - Real-time detections (Redis pub/sub nx:qa:detections)
  - API calls log (Redis pub/sub nx:qa:apicalls)
  - Capability toggles (Redis hash nx:qa:capabilities)
  - Camera selector (all / individual)
  - Recordings tab: clip gallery, preview, and inference playback

Start with: ./qa.sh  (from deploy/)
"""

import json
import os
import shutil
import threading
import time
from collections import deque
from pathlib import Path

import redis
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST  = os.getenv("REDIS_HOST", "redis")
MJPEG_PORT  = int(os.getenv("MJPEG_PORT", "8080"))

# Extract just the IP from the Host header so the browser can reach the MJPEG server
# directly. st.context.headers["host"] = "100.67.192.58:8501" — strip the port.
try:
    _host_hdr = st.context.headers.get("host", "")
    MJPEG_HOST = _host_hdr.split(":")[0] if _host_hdr else "localhost"
except Exception:
    MJPEG_HOST = "localhost"
MAX_DETECTIONS  = 200
MAX_APICALLS    = 50
RECORDINGS_DIR  = Path("/nx_tech/recordings")

st.set_page_config(
    page_title="NX QA Visual",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Autorefresh para detecciones y API calls — el video lo maneja el browser directamente.
# 2000 ms en vez de 500 ms: la tab Grabaciones lee muchos archivos por rerun y con 500 ms
# el overlay gris de Streamlit era constante. El stream MJPEG no se ve afectado (va por HTTP
# directo al browser). Las detecciones se actualizan desde el buffer del subscriber thread.
st_autorefresh(interval=2000, limit=None, key="qa_tick")


# ── Redis helpers ─────────────────────────────────────────────────────────────
@st.cache_resource
def _get_redis():
    """Create and cache a single Redis connection per Streamlit process.

    cache_resource ensures one connection per process — avoids a new connection on
    every 2 s autorefresh cycle. Returns None if Redis is unreachable.
    """
    try:
        r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True,
                        socket_connect_timeout=2, socket_timeout=2)
        r.ping()
        return r
    except Exception:
        return None


def _redis_ok() -> bool:
    """Return True if Redis is reachable, False if it is down or unavailable."""
    r = _get_redis()
    if r is None:
        return False
    try:
        r.ping()
        return True
    except Exception:
        return False


# ── Process-level buffers — shared across rerenders, no ScriptRunContext needed ──
# cache_resource creates a process singleton: the subscriber thread writes here and
# the main script reads on every rerender without threading issues.
@st.cache_resource
def _get_buffers():
    """Create and cache circular detection and API-call buffers for the process.

    Shared between the subscriber thread (writer) and the main script (reader on each
    rerender). cache_resource guarantees one instance per process.
    """
    return {
        "detections": deque(maxlen=MAX_DETECTIONS),
        "apicalls":   deque(maxlen=MAX_APICALLS),
    }

@st.cache_resource
def _ensure_subscriber(_host=REDIS_HOST):
    """Start the Redis pub/sub subscriber thread once per process.

    Auto-reconnects if Redis goes down — the 2 s sleep prevents a tight
    reconnect loop from flooding logs during an outage.
    """
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
                time.sleep(2)  # avoid tight reconnect loop during Redis outage

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


# ── Config editor helpers ──────────────────────────────────────────────────────
_PACKAGES = [
    "comercio_basico", "comercio_avanzado", "comercio_total", "comercio_enterprise",
    "industrial_basico", "industrial_avanzado", "industrial_total", "industrial_enterprise",
    "hogar_basico", "hogar_avanzado", "hogar_total",
    "manual",
]


def _save_config_yaml() -> tuple[bool, str]:
    """Persist all current config overrides and live Redis values to the client's config.yaml.

    Reads nx:qa:config_overrides (restart-required fields) and nx:qa:entry_exit /
    nx:qa:external_channels / nx:qa:count_* (hot-reloadable fields) from Redis,
    merges them into the existing YAML, and writes it back using ruamel.yaml so
    inline comments in the file are preserved.

    Returns:
        (True, success_message) on success, (False, error_message) on failure.
    """
    client = status.get("client", "")
    if not client:
        return False, "No hay cliente activo en el pipeline"
    config_path = Path(f"/nx_tech/clients/{client}/config.yaml")
    if not config_path.exists():
        return False, f"No existe {config_path}"
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedSeq

        def _flist(items):
            """Wrap a list in ruamel.yaml's flow style so it serializes on one line.

            Without this, `channels: [1, 2, 3]` becomes a multi-line block sequence,
            which is harder to read and edit directly in config.yaml.
            """
            seq = CommentedSeq(items)
            seq.fa.set_flow_style()
            return seq

        yml = YAML()
        yml.preserve_quotes = True
        with open(config_path) as f:
            cfg_data = yml.load(f)

        # Restart-required fields (package, tracker, channels, PGIE/SGIE, DVR)
        if _r:
            raw_ov = _r.get("nx:qa:config_overrides")
            if raw_ov:
                for key, val in json.loads(raw_ov).items():
                    cfg_data[key] = _flist(val) if key == "channels" else val

        # Hot-reloadable fields — applied without restarting the pipeline
        if _r:
            raw_ee = _r.get("nx:qa:entry_exit")
            if raw_ee is not None:
                cfg_data["entry_exit_channels"] = _flist(sorted(json.loads(raw_ee)))
            raw_ext = _r.get("nx:qa:external_channels")
            if raw_ext is not None:
                cfg_data["external_channels"] = _flist(sorted(json.loads(raw_ext)))
            ci = _r.get("nx:qa:count_internal")
            if ci is not None:
                cfg_data["count_internal"] = ci == "1"
            ce = _r.get("nx:qa:count_external")
            if ce is not None:
                cfg_data["count_external"] = ce == "1"

        with open(config_path, "w") as f:
            yml.dump(cfg_data, f)
        return True, f"Guardado en {config_path.name}"
    except Exception as exc:
        return False, f"Error al guardar: {exc}"

pipeline_stats: dict = {}
if _r:
    try:
        raw_stats = _r.get("nx:qa:pipeline_stats")
        if raw_stats:
            pipeline_stats = json.loads(raw_stats)
    except Exception:
        pass


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

    # Camera selector
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

    # ── Entry / Exit toggles ──────────────────────────────────────────────────
    if channels:
        st.markdown("**📍 Entrada / Salida**")
        st.caption("Cámaras que cubren la puerta de entrada")

        # Read current list from Redis; fall back to what the pipeline published
        ee_channels: set = set()
        if _r:
            try:
                raw_ee = _r.get("nx:qa:entry_exit")
                if raw_ee is not None:
                    ee_channels = set(json.loads(raw_ee))
                else:
                    ee_channels = set(status.get("entry_exit_channels", []))
                    _r.set("nx:qa:entry_exit", json.dumps(sorted(ee_channels)))
            except Exception:
                ee_channels = set(status.get("entry_exit_channels", []))
        else:
            ee_channels = set(status.get("entry_exit_channels", []))

        new_ee_channels: set = set()
        for ch in channels:
            is_ee = ch in ee_channels
            label = f"Cám {ch:02d}  {'🚪' if is_ee else ''}"
            new_val = st.checkbox(label, value=is_ee, key=f"ee_ch_{ch}",
                                  disabled=(_r is None))
            if new_val:
                new_ee_channels.add(ch)

        if _r and new_ee_channels != ee_channels:
            try:
                _r.set("nx:qa:entry_exit", json.dumps(sorted(new_ee_channels)))
            except Exception:
                pass

    st.markdown("---")

    # ── Camera type ───────────────────────────────────────────────────────────
    if channels:
        st.markdown("**🏠 Tipo de Cámara**")
        st.caption("Sin marcar = interna (default). Marcar = externa")

        ext_channels: set = set()
        if _r:
            try:
                raw_ext = _r.get("nx:qa:external_channels")
                if raw_ext is not None:
                    ext_channels = set(json.loads(raw_ext))
                else:
                    ext_channels = set(status.get("external_channels", []))
                    _r.set("nx:qa:external_channels", json.dumps(sorted(ext_channels)))
            except Exception:
                ext_channels = set(status.get("external_channels", []))
        else:
            ext_channels = set(status.get("external_channels", []))

        new_ext_channels: set = set()
        for ch in channels:
            is_ext = ch in ext_channels
            label = f"Cám {ch:02d}  {'🏢 externa' if is_ext else '🏠 interna'}"
            new_val = st.checkbox(label, value=is_ext, key=f"ext_ch_{ch}",
                                  disabled=(_r is None))
            if new_val:
                new_ext_channels.add(ch)

        if _r and new_ext_channels != ext_channels:
            try:
                _r.set("nx:qa:external_channels", json.dumps(sorted(new_ext_channels)))
            except Exception:
                pass

        count_int_val = (_r.get("nx:qa:count_internal") or "1") == "1" if _r else True
        count_ext_val = (_r.get("nx:qa:count_external") or "1") == "1" if _r else True

        new_count_int = st.checkbox("Internas cuentan", value=count_int_val,
                                    key="count_internal", disabled=(_r is None))
        new_count_ext = st.checkbox("Externas cuentan", value=count_ext_val,
                                    key="count_external", disabled=(_r is None))

        if _r:
            try:
                if new_count_int != count_int_val:
                    _r.set("nx:qa:count_internal", "1" if new_count_int else "0")
                if new_count_ext != count_ext_val:
                    _r.set("nx:qa:count_external", "1" if new_count_ext else "0")
            except Exception:
                pass

    st.markdown("---")

    # Capability toggles
    IMPLEMENTED = [
        "people_counting", "age_gender", "fall_detection", "face_recognition",
    ]
    UNIMPLEMENTED = ["epp_detection", "fire_smoke", "license_plate"]

    st.markdown("**🔧 Capacidades**")
    # people_counting is always on — the pipeline has no code path without it
    st.checkbox("people_counting", value=True, disabled=True,
                help="Siempre activo — no se puede apagar", key="cap_people_counting")

    # Implemented capabilities — all toggleable at runtime.
    # Caps not active in the pipeline apply only to Python workers that check Redis;
    # SGIEs (age_gender) require a pipeline restart to take effect.
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

    # ── Pipeline resolutions ──────────────────────────────────────────────────
    st.markdown("**🔬 Resoluciones**")
    comp = status.get("component_resolutions", {})
    if comp:
        res_rows = [
            ("Fuente DVR",         comp.get("source")),
            ("Probe analytics",    comp.get("probe_a_frame")),
            ("PGIE PeopleNet",     comp.get("pgie_input", "960x544")),
            ("SGIE Edad/Género",   comp.get("age_gender_input", "224x224")),
            ("PeopleNet faces",     comp.get("facedetect_input")),
            ("MoveNet caídas",     comp.get("movenet_input", "192x192")),
            ("OSNet re-ID",        comp.get("osnet_input", "128x256")),
            ("Tiler QA display",   comp.get("probe_b_frame", "640x360")),
        ]
        for label, res in res_rows:
            if res:
                st.caption(f"`{res}` · {label}")
    else:
        st.caption("Disponible cuando el pipeline está activo.")

    st.markdown("---")

    # ── Pipeline FPS ─────────────────────────────────────────────────────────
    st.markdown("**⚡ FPS del Pipeline**")
    if pipeline_stats:
        fps_per_cam = pipeline_stats.get("fps_per_camera", {})
        fps_avg = (sum(fps_per_cam.values()) / len(fps_per_cam)) if fps_per_cam else 0.0
        st.metric("Promedio por cámara", f"{fps_avg:.1f} fps")
        for cam_id, fps_val in pipeline_stats.get("fps_per_camera", {}).items():
            ch = cam_id.split("-ch")[-1] if "-ch" in cam_id else cam_id
            st.caption(f"`ch{ch}` → {fps_val:.1f} fps")
        if ts := pipeline_stats.get("ts", "")[-12:-4]:
            st.caption(f"Act: {ts}")
    else:
        st.caption("Pipeline inactivo.")

    st.markdown("---")

    # ── Config editor ─────────────────────────────────────────────────────────
    st.markdown("**⚙️ Configuración del pipeline**")
    st.caption("Edita aquí para experimentar. El botón Guardar escribe en config.yaml.")

    if not status:
        st.info("Inicia el pipeline para editar la config.")
    else:
        # ── Initialize session_state from pipeline status ──────────────────────
        # Correct Streamlit pattern: set session_state BEFORE rendering widgets,
        # then render WITHOUT value=/index= to avoid the double-render flicker.
        # Re-initialize when: first load, pipeline restarted (new config_gen),
        # or user clicks Reset.
        _cfg_gen = (_r.get("nx:qa:config_gen") if _r else None) or ""
        _needs_init = (
            "_cfg_gen" not in st.session_state or
            _cfg_gen != st.session_state.get("_cfg_gen")
        )
        if _needs_init:
            if _r:
                try:
                    _r.delete("nx:qa:config_overrides")
                except Exception:
                    pass
            _ch_init = status.get("channels") or [1]
            st.session_state.update({
                "_cfg_gen":          _cfg_gen,
                "cfg_package":       status.get("package") or "manual",
                "cfg_stream_type":   status.get("stream_type") or "main",
                "cfg_tracker":       status.get("tracker") or "nvdcf",
                "cfg_channels":      ", ".join(str(c) for c in _ch_init),
                "cfg_pgie_interval": int(status["pgie_interval"])  if status.get("pgie_interval")  is not None else -1,
                "cfg_pgie_batch":    int(status["pgie_batch_size"]) if status.get("pgie_batch_size") is not None else 0,
                "cfg_sgie_interval": int(status["sgie_interval"])  if status.get("sgie_interval")  is not None else -1,
                "cfg_reid_gallery":  int(status["reid_gallery_size"]) if status.get("reid_gallery_size") is not None else 10,
                "cfg_dvr_port":       int(status["dvr_port"]) if status.get("dvr_port") is not None else 554,
                "cfg_rtsp_pattern":   status.get("rtsp_url_pattern") or
                                      "rtsp://{user}:{password}@{dvr_ip}:{port}/ch{ch:02d}/main/av_stream",
                "cfg_recording_enabled": bool(status.get("recording_enabled", False)),
            })

        # Persisted overrides from Redis — used only for comparison and the Save button
        _cfg_ov: dict = {}
        if _r:
            try:
                _raw_ov = _r.get("nx:qa:config_overrides")
                if _raw_ov:
                    _cfg_ov = json.loads(_raw_ov)
            except Exception:
                pass

        # ── Widgets — no value=/index= to avoid double-render flicker ──────────
        _new_pkg    = st.selectbox("Paquete", _PACKAGES, key="cfg_package")
        _new_st     = st.selectbox("Stream type", ["main", "sub"], key="cfg_stream_type")
        _new_tr     = st.selectbox("Tracker", ["nvdcf", "iou"], key="cfg_tracker")
        _new_ch_str = st.text_input("Canales activos", key="cfg_channels",
                                    help="Números separados por coma: 1, 2, 3")
        try:
            _new_chs = sorted({int(x.strip()) for x in _new_ch_str.split(",")
                               if x.strip().isdigit()}) or (status.get("channels") or [1])
        except Exception:
            _new_chs = status.get("channels") or [1]

        st.markdown("**Inferencia GPU**")
        _new_pgi = st.number_input("PGIE interval", min_value=-1, max_value=10,
                                   step=1, key="cfg_pgie_interval",
                                   help="-1 = desde nvinfer_config.txt")
        _new_pgb = st.number_input("PGIE batch", min_value=0, max_value=16,
                                   step=1, key="cfg_pgie_batch",
                                   help="0 = desde nvinfer_config.txt")
        _new_sgi = st.number_input("SGIE interval", min_value=-1, max_value=10,
                                   step=1, key="cfg_sgie_interval",
                                   help="-1 = desde config_infer.txt")
        _new_rgs = st.number_input("ReID gallery", min_value=1, max_value=20,
                                   step=1, key="cfg_reid_gallery",
                                   help="Embeddings máx por persona")

        st.markdown("**Grabación**")
        _new_rec = st.checkbox("recording_enabled", key="cfg_recording_enabled",
                               help="Grabar video cuando se detectan personas (producción + QA). Aplica en caliente al guardar.")

        st.markdown("**DVR**")
        _new_port = st.number_input("Puerto DVR", min_value=1, max_value=65535,
                                    step=1, key="cfg_dvr_port")
        _new_pat  = st.text_input("RTSP URL pattern", key="cfg_rtsp_pattern")

        # Persist to Redis only when values changed — avoids a write on every 2 s rerun
        _new_ov = {
            "package":            _new_pkg,
            "stream_type":        _new_st,
            "tracker":            _new_tr,
            "channels":           _new_chs,
            "pgie_interval":      _new_pgi,
            "pgie_batch_size":    _new_pgb,
            "sgie_interval":      _new_sgi,
            "reid_gallery_size":  _new_rgs,
            "recording_enabled":  _new_rec,
            "dvr_port":           _new_port,
            "rtsp_url_pattern":   _new_pat,
        }
        if _r and _new_ov != _cfg_ov:
            try:
                _r.set("nx:qa:config_overrides", json.dumps(_new_ov))
            except Exception:
                pass

        # ── Actions ───────────────────────────────────────────────────────────
        st.caption("🔄 Package, stream type, tracker, channels, PGIE/SGIE, and DVR require a pipeline restart.")
        if st.button("💾 Guardar en config.yaml", width="stretch",
                     type="primary", key="btn_save_config", disabled=(_r is None)):
            _ok, _msg = _save_config_yaml()
            st.session_state["_save_result"] = (_ok, _msg)
        if st.button("↺ Recargar del pipeline", width="stretch",
                     key="btn_reset_config", disabled=not status,
                     help="Descarta cambios y recarga los valores actuales del pipeline"):
            if _r:
                try:
                    _r.delete("nx:qa:config_overrides")
                except Exception:
                    pass
            for _k in ["_cfg_gen", "cfg_package", "cfg_stream_type", "cfg_tracker",
                       "cfg_channels", "cfg_pgie_interval", "cfg_pgie_batch",
                       "cfg_sgie_interval", "cfg_reid_gallery", "cfg_recording_enabled",
                       "cfg_dvr_port", "cfg_rtsp_pattern"]:
                st.session_state.pop(_k, None)
            # No st.rerun() — the button click already triggers a rerun automatically

        if "_save_result" in st.session_state:
            _ok_r, _msg_r = st.session_state["_save_result"]
            (st.success if _ok_r else st.error)(_msg_r)

    st.markdown("---")

    # Recording / playback state indicator
    if _r:
        try:
            _sb_rec = (_r.get("nx:qa:recording_active") or "0") == "1"
            _sb_pb  = bool(_r.get("nx:qa:playback_video"))
            if _sb_pb:
                st.warning("⏯ Modo playback")
            elif _sb_rec:
                st.success("⏺ Grabando")
            else:
                st.caption("⚫ Sin grabación activa")
        except Exception:
            pass

    if _redis_ok():
        st.success("● Redis conectado")
    else:
        st.error("● Redis desconectado")
        st.caption("Verifica que el pipeline esté corriendo con `./qa.sh`")


# ── Helper: clip gallery ──────────────────────────────────────────────────────
def _render_clips_list() -> None:
    """Render the saved clip gallery with thumbnail, metadata, and action buttons.

    Used in both the default recordings view and the expander inside playback mode.
    """
    clips: list = []
    if RECORDINGS_DIR.exists():
        clips = sorted(
            [d for d in RECORDINGS_DIR.iterdir() if d.is_dir()],
            reverse=True,
        )

    if not clips:
        st.info("No recordings yet. They are created automatically when people are detected with the QA pipeline active.")
        return

    st.markdown(f"**{len(clips)} clip(s) guardados** · máx 10 GB (auto-rotativo)")
    st.markdown("")

    for clip_dir in clips:
        meta_path  = clip_dir / "metadata.json"
        thumb_path = clip_dir / "thumbnail.jpg"
        tiled_path = clip_dir / "tiled.mp4"

        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:
            meta = {}  # corrupted metadata — render the clip with blank fields

        ts_label = meta.get("timestamp", clip_dir.name)
        duration = meta.get("duration_s", 0)
        max_ppl  = meta.get("max_people", 0)
        n_frames = meta.get("frame_count", 0)
        cam_ids  = meta.get("channels", [])

        with st.expander(f"📹 {ts_label}  ·  {duration:.0f}s  ·  {max_ppl} pers. máx"):
            col_thumb, col_info, col_act = st.columns([3, 4, 3])

            with col_thumb:
                if thumb_path.exists():
                    st.image(str(thumb_path), width="stretch")
                else:
                    st.caption("Sin thumbnail")

            with col_info:
                st.caption(f"Duración: {duration:.1f} s")
                st.caption(f"Frames tileados: {n_frames}")
                st.caption(f"Personas máx: {max_ppl}")
                st.caption(f"Cámaras: {len(cam_ids)}")
                if tiled_path.exists():
                    st.caption("✅ tiled.mp4 (640×360)")
                for cam_id in cam_ids:
                    if (clip_dir / f"{cam_id}.mp4").exists():
                        st.caption(f"✅ {cam_id}.mp4 (full-res)")

            with col_act:
                if tiled_path.exists():
                    if st.button("▶ Preview tiled", key=f"prev_{clip_dir.name}"):
                        st.session_state["_preview_path"] = str(tiled_path)
                        st.session_state["_preview_label"] = ts_label

                video_options = []
                if tiled_path.exists():
                    video_options.append(("tiled.mp4", str(tiled_path)))
                for cam_id in cam_ids:
                    cam_file = clip_dir / f"{cam_id}.mp4"
                    if cam_file.exists():
                        video_options.append((f"{cam_id}.mp4", str(cam_file)))

                if video_options and _r:
                    selected_idx = st.selectbox(
                        "Inferencia en:",
                        options=range(len(video_options)),
                        # Default-arg capture: `vo=video_options` binds this iteration's
                        # list value into the lambda. Without it, all lambdas in this loop
                        # would share the same reference and always use the last value.
                        format_func=lambda i, vo=video_options: vo[i][0],
                        key=f"inf_sel_{clip_dir.name}",
                    )
                    if st.button("▶ Correr Inferencia", key=f"inf_{clip_dir.name}",
                                 type="primary"):
                        try:
                            _r.set("nx:qa:playback_video", video_options[selected_idx][1])
                            st.success("Pipeline reiniciando con el video seleccionado (~30–60 s)...")
                        except Exception as exc:
                            st.error(f"Error al iniciar playback: {exc}")

                if st.button("🗑 Eliminar", key=f"del_{clip_dir.name}"):
                    shutil.rmtree(clip_dir, ignore_errors=True)
                    st.toast(f"Clip {ts_label} eliminado.")
                    st.rerun()


# ── MAIN — Tabs: En Vivo | Grabaciones ────────────────────────────────────────
tab_live, tab_recordings = st.tabs(["🔴 En Vivo", "📹 Grabaciones"])

# ── Tab: En Vivo ──────────────────────────────────────────────────────────────
with tab_live:
    col_video, col_det = st.columns([55, 45])

    with col_video:
        st.markdown("### 📹 Video en vivo")
        stream_label = "Todas las cámaras (tiled)" if stream_key == "all" else stream_key
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
                            f"&nbsp;&nbsp;{icon} `{label}` &nbsp;·&nbsp;`ch{cam_short}` &nbsp;conf={conf:.2f}",
                            unsafe_allow_html=True,
                        )

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


# ── Tab: Grabaciones ──────────────────────────────────────────────────────────
with tab_recordings:
    # Estado de grabación activa
    rec_active = False
    rec_info: dict = {}
    playback_video = ""
    if _r:
        try:
            rec_active = (_r.get("nx:qa:recording_active") or "0") == "1"
            if rec_active:
                rec_info = json.loads(_r.get("nx:qa:recording_info") or "{}")
            playback_video = _r.get("nx:qa:playback_video") or ""
        except Exception:
            pass

    col_status, col_playback = st.columns([1, 1])
    with col_status:
        if rec_active:
            ts_rec = rec_info.get("clip_name", "")
            n_ppl  = rec_info.get("people_count", 0)
            st.success(f"⏺ Grabando: {ts_rec} · {n_ppl} personas")
        else:
            st.info("⚫ Grabación inactiva — se activa al detectar personas")

    with col_playback:
        if playback_video:
            st.warning(f"⏯ Playback: `{Path(playback_video).name}`")
            if st.button("🔴 Volver a En Vivo", type="primary", key="btn_back_live"):
                if _r:
                    try:
                        _r.delete("nx:qa:playback_video")
                        st.toast("Volviendo a modo live en el próximo ciclo del pipeline...")
                    except Exception:
                        pass

    st.markdown("---")

    # When playback is active, show the MJPEG stream full-width so the user can
    # inspect inference results the same way they would in the live tab.
    if playback_video:
        st.markdown(f"### Inferencia en curso: `{Path(playback_video).name}`")
        col_pb_video, col_pb_det = st.columns([55, 45])

        with col_pb_video:
            # Playback always has a single stream so we always use /stream/all.
            # The sidebar capability toggles apply in real time during playback.
            viewer_url = f"http://{MJPEG_HOST}:{MJPEG_PORT}/viewer/all"
            st.iframe(viewer_url, height=560)
            st.caption("Playback con inferencia · 640×360 · MJPEG")

        with col_pb_det:
            st.markdown("### Detecciones")
            # NOTE: this detection log block is duplicated from tab_live.
            # Extracting a shared helper would require restructuring the module-level
            # rendering flow — left as-is to avoid scope creep in this change.
            det_items = list(_bufs["detections"])
            if not det_items:
                st.info("Esperando detecciones... El pipeline puede tardar ~30–60 s en reiniciar.")
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
                                f"&nbsp;&nbsp;{icon} `{label}` &nbsp;·&nbsp;`ch{cam_short}` &nbsp;conf={conf:.2f}",
                                unsafe_allow_html=True,
                            )

        st.markdown("---")
        st.markdown("### API Calls")
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

        st.markdown("---")
        with st.expander("Ver grabaciones anteriores"):
            _render_clips_list()

    else:
        _render_clips_list()

    # ── Preview inline ────────────────────────────────────────────────────────
    if "_preview_path" in st.session_state:
        preview_path = st.session_state["_preview_path"]
        preview_lbl  = st.session_state.get("_preview_label", "")
        st.markdown("---")
        st.markdown(f"### Preview: {preview_lbl}")
        if Path(preview_path).exists():
            st.video(preview_path)
        else:
            st.warning("Archivo no encontrado.")
        if st.button("✕ Cerrar preview", key="btn_close_preview"):
            st.session_state.pop("_preview_path", None)
            st.session_state.pop("_preview_label", None)
            st.rerun()
