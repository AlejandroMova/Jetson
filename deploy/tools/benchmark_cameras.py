#!/usr/bin/env python3
"""
benchmark_cameras.py — NX Computing AI | Techo de cámaras por modalidad

Mide cuántas cámaras aguanta *este* Jetson en tiempo real, por cada modalidad de
config.yaml, corriendo el pipeline de producción real (app.py) con N fuentes RTSP
y midiendo FPS por stream + carga de GPU/RAM.

Se corre A MANO en el Jetson (no forma parte de setup.sh). Detiene el deepstream de
producción mientras mide y lo restaura al terminar.

Cómo mide N cámaras: cicla los canales configurados del cliente (channels × repeticiones)
para llegar a N, así reparte las sesiones entre canales del DVR.

Modalidades por defecto (todas con tracker=nvdcf_reid, pgie_batch_size=0, pgie_interval=2):
    fp32        — osnet_precision=fp32 (ideal, más preciso)
    fp16        — osnet_precision=fp16 (menos precisión, menos costo — el bottleneck es OSNet)
    fp16_sgie2  — fp16 + sgie_interval=2 (OSNet cada 2 frames; el cuello de botella real)

"Aguanta" = FPS por stream ≥ margen × FPS de la fuente (gst-discoverer) Y GPU no saturada
Y RAM con headroom. Es el test compute-bound-vs-source de systemrefactor.md §2.5, en vivo.

Requiere: correr en el host del Jetson (usa tegrastats + docker compose + gst-discoverer,
todos nativos de JetPack). El probe de FPS lo aporta app.py cuando NX_BENCH_FPS=1.

Este script corre en el HOST, no dentro de Docker (a diferencia de la mayoría de tools/*.py) —
por eso solo usa pyyaml (ya presente en el host) + stdlib, nunca ruamel.yaml ni python-dotenv:
el config.yaml de __bench__/ es descartable (se borra al terminar), así que no hace falta
preservar comentarios al escribirlo, y .env se puede leer con un parser de dos líneas.

Usage (desde deploy/):
    python3 tools/benchmark_cameras.py --list-variants               # qué config aplica cada modalidad
    python3 tools/benchmark_cameras.py                              # 3 modalidades, counts 1,2,4,6,8
    python3 tools/benchmark_cameras.py --counts 1,4,8,12,16
    python3 tools/benchmark_cameras.py --variants fp16 --max 16 --step 2
    python3 tools/benchmark_cameras.py --client demo --target-fps 15 --measure 60
"""

import argparse
import atexit
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT   = Path(__file__).resolve().parent.parent   # .../deploy
BENCH_CLIENT = "__bench__"                              # cliente temporal que escribimos/borramos
BENCH_CONTAINER = "nx_bench"                            # nombre fijo del container de medición

# Cada modalidad = overrides sobre config.yaml. tracker/pgie son iguales en todas (los "ideales");
# solo cambian osnet_precision y sgie_interval. sgie_interval=None → usar el default del archivo.
VARIANTS = {
    "fp32": {
        "osnet_precision": "fp32", "sgie_interval": None,
        "desc": "Ideal completo: máxima precisión de ReID (OSNet en FP32).",
    },
    "fp16": {
        "osnet_precision": "fp16", "sgie_interval": None,
        "desc": "OSNet en FP16 — más rápido, mínima pérdida de precisión reportada.",
    },
    "fp16_sgie2": {
        "osnet_precision": "fp16", "sgie_interval": 2,
        "desc": "fp16 + OSNet cada 2 frames en vez de cada frame (el cuello de botella real).",
    },
}


def _print_variants():
    """Imprime qué config aplica cada modalidad — para --list-variants."""
    print("Modalidades disponibles (todas con tracker=nvdcf_reid, pgie_batch_size=0, pgie_interval=2):\n")
    for name, v in VARIANTS.items():
        sgie = v["sgie_interval"] if v["sgie_interval"] is not None else "(default del archivo)"
        print(f"  {name}")
        print(f"    osnet_precision : {v['osnet_precision']}")
        print(f"    sgie_interval   : {sgie}")
        print(f"    {v['desc']}\n")

# Estado para el cleanup: si detuvimos el deepstream de producción, hay que restaurarlo.
_prod_was_up = False


# ── Docker helpers ────────────────────────────────────────────────────────────

def _compose(*args, check=True, capture=False):
    """Corre `docker compose <args>` desde deploy/. Retorna el CompletedProcess."""
    cmd = ["docker", "compose", *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def _prod_is_up() -> bool:
    """True si el servicio deepstream de producción está corriendo."""
    r = _compose("ps", "-q", "deepstream", capture=True, check=False)
    return bool(r.stdout.strip())


def _stop_bench_container():
    """Elimina el container de medición si quedó vivo (idempotente)."""
    subprocess.run(["docker", "rm", "-f", BENCH_CONTAINER],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


# ── Config temporal por modalidad ─────────────────────────────────────────────

def _cycle_channels(base: list, n: int) -> list:
    """Repite/cicla la lista base de canales hasta longitud n (ej. [1,2,3], n=8 → [1,2,3,1,2,3,1,2])."""
    base = base or [1]
    return [base[i % len(base)] for i in range(n)]


def _write_bench_config(base_channels: list, n: int, variant: dict):
    """Escribe clients/__bench__/config.yaml con la modalidad + N canales.

    Parte de la copia del cliente base (hecha en _setup_bench_client) y sobrescribe solo los
    knobs de la prueba. sgie_interval se borra si la variante no lo fija (usa el default del archivo).
    Usa yaml.safe_dump en vez de ruamel: __bench__/config.yaml es descartable (se borra al
    terminar), así que perder los comentarios del template no importa — nadie lo edita a mano.
    """
    cfg_path = REPO_ROOT / "clients" / BENCH_CLIENT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    # Knobs "ideales" fijos en todas las modalidades.
    cfg["tracker"]         = "nvdcf_reid"
    cfg["pgie_batch_size"] = 0
    cfg["pgie_interval"]   = 2
    # Knobs que varían por modalidad.
    cfg["osnet_precision"] = variant["osnet_precision"]
    if variant["sgie_interval"] is not None:
        cfg["sgie_interval"] = variant["sgie_interval"]
    else:
        cfg.pop("sgie_interval", None)
    # N cámaras simuladas ciclando los canales reales.
    cfg["channels"] = _cycle_channels(base_channels, n)

    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _setup_bench_client(base_client: str) -> list:
    """Copia clients/<base_client>/ a clients/__bench__/ (incluye .env) y retorna sus canales base.

    El container lee esta carpeta vía el volume ./clients y NX_CLIENT=__bench__.
    """
    base_dir  = REPO_ROOT / "clients" / base_client
    bench_dir = REPO_ROOT / "clients" / BENCH_CLIENT
    if not (base_dir / "config.yaml").exists():
        sys.exit(f"[ERR] No existe config del cliente base: {base_dir}/config.yaml")

    if bench_dir.exists():
        shutil.rmtree(bench_dir)
    shutil.copytree(base_dir, bench_dir)   # copia config.yaml + .env

    with open(base_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    channels = cfg.get("channels") or [1]
    return list(channels)


def _cleanup():
    """Restaura el estado: borra container + cliente temporal y reinicia producción si estaba arriba."""
    _stop_bench_container()
    bench_dir = REPO_ROOT / "clients" / BENCH_CLIENT
    if bench_dir.exists():
        shutil.rmtree(bench_dir, ignore_errors=True)
    if _prod_was_up:
        print("\n[cleanup] Restaurando deepstream de producción…")
        _compose("up", "-d", "deepstream", check=False)


# ── FPS de la fuente (target de tiempo real) ──────────────────────────────────

def _read_env_file(path: Path) -> dict:
    """Parser mínimo de .env (KEY=value por línea, comillas opcionales, # = comentario).

    Reemplaza python-dotenv (no garantizado en el host, a diferencia del contenedor) —
    el formato de .env acá es siempre simple, no amerita una dependencia extra.
    """
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _source_fps(base_client: str) -> float:
    """Sondea la fuente RTSP con gst-discoverer-1.0 y retorna el framerate real del DVR.

    Arma la URL desde config.yaml (dvr_port, rtsp_url_pattern, primer channel), .env
    (DVR_USER/DVR_PASS) y /etc/nx_dvr_ip. Es el método del usuario (systemrefactor.md §12).
    La contraseña se enmascara en la salida. Retorna 0.0 si no se pudo determinar.
    """
    base_dir = REPO_ROOT / "clients" / base_client
    with open(base_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    env = _read_env_file(base_dir / ".env")

    dvr_ip = (Path("/etc/nx_dvr_ip").read_text().strip()
              if Path("/etc/nx_dvr_ip").exists() else "")
    port   = cfg.get("dvr_port", 554)
    pattern = cfg.get("rtsp_url_pattern", "")
    ch      = (cfg.get("channels") or [1])[0]
    user    = env.get("DVR_USER", "")
    password = env.get("DVR_PASS", "")
    if not (dvr_ip and pattern and user):
        print("[warn] Falta DVR IP/patrón/credenciales — no se puede sondear la fuente.")
        return 0.0

    url = (pattern.replace("{user}", user).replace("{password}", password)
           .replace("{dvr_ip}", dvr_ip).replace("{port}", str(port))
           .replace("{ch:02d}", f"{ch:02d}").replace("{ch}", str(ch)))
    masked = url.replace(password, "****") if password else url
    print(f"[fuente] gst-discoverer sobre {masked}")

    try:
        r = subprocess.run(["gst-discoverer-1.0", "-t", "15", url],
                           capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[warn] gst-discoverer falló ({e}) — usa --target-fps para fijar el objetivo.")
        return 0.0

    # gst-discoverer imprime "frame rate: 15/1" (num/den) para el stream de video.
    m = re.search(r"frame ?rate:\s*(\d+)/(\d+)", r.stdout + r.stderr, re.I)
    if not m or int(m.group(2)) == 0:
        print("[warn] No se pudo leer el framerate del gst-discoverer — usa --target-fps.")
        return 0.0
    fps = int(m.group(1)) / int(m.group(2))
    print(f"[fuente] FPS que entrega el DVR: {fps:.1f}")
    return fps


# ── Medición de una corrida (una modalidad × un N) ────────────────────────────

def _run_pipeline(start_timeout: int) -> bool:
    """Lanza el container de medición y espera hasta la primera línea BENCH_FPS.

    Retorna True si el pipeline arrancó y emite FPS dentro de start_timeout (la 1ª corrida
    fp16 puede tardar minutos construyendo el engine). Si no, imprime las últimas líneas del
    log para diagnóstico y retorna False.
    """
    _stop_bench_container()
    _compose("run", "-d", "--name", BENCH_CONTAINER,
             "-e", f"NX_CLIENT={BENCH_CLIENT}", "-e", "NX_BENCH_FPS=1",
             "deepstream", "python3", "pipelines/app.py")

    deadline = time.monotonic() + start_timeout
    while time.monotonic() < deadline:
        logs = subprocess.run(["docker", "logs", BENCH_CONTAINER],
                             capture_output=True, text=True, check=False)
        blob = logs.stdout + logs.stderr
        if "BENCH_FPS" in blob:
            return True
        # Si el container ya murió, no tiene sentido seguir esperando.
        alive = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", BENCH_CONTAINER],
                              capture_output=True, text=True, check=False)
        if alive.stdout.strip() != "true":
            print("[err] El pipeline murió antes de emitir FPS. Últimas líneas:")
            print("\n".join(blob.strip().splitlines()[-20:]))
            return False
        time.sleep(3)

    print(f"[err] Timeout ({start_timeout}s) esperando FPS. Últimas líneas del log:")
    logs = subprocess.run(["docker", "logs", "--tail", "20", BENCH_CONTAINER],
                         capture_output=True, text=True, check=False)
    print(logs.stdout + logs.stderr)
    return False


def _parse_tegrastats(text: str):
    """Extrae medianas de GPU%, RAM% y EMC% de la salida de tegrastats.

    Formato típico por línea: 'RAM 3200/7772MB ... EMC_FREQ 12% ... GR3D_FREQ 45%'.
    Retorna (gpu_pct, ram_pct, emc_pct) — cada uno mediana sobre las muestras, o None si no hubo.
    """
    gpu, emc, ram_pct = [], [], []
    for line in text.splitlines():
        g = re.search(r"GR3D_FREQ\s+(\d+)%", line)
        if g:
            gpu.append(int(g.group(1)))
        e = re.search(r"EMC_FREQ\s+(\d+)%", line)
        if e:
            emc.append(int(e.group(1)))
        r = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
        if r and int(r.group(2)) > 0:
            ram_pct.append(100.0 * int(r.group(1)) / int(r.group(2)))
    med = lambda xs: statistics.median(xs) if xs else None
    return med(gpu), med(ram_pct), med(emc)


def _parse_fps(text: str):
    """Agrupa las líneas BENCH_FPS por stream y retorna (min_fps_por_stream, n_streams_vivos).

    Toma la mediana de fps de cada stream y luego el mínimo entre streams (el stream peor,
    el que decide si el sistema 'aguanta'). n_streams_vivos = cuántos streams emitieron FPS.
    """
    per_stream: dict = {}
    for m in re.finditer(r"BENCH_FPS stream=(\d+) fps=([\d.]+)", text):
        per_stream.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    if not per_stream:
        return None, 0
    medians = [statistics.median(v) for v in per_stream.values()]
    return min(medians), len(per_stream)


def _measure(measure_s: int):
    """Corre tegrastats durante la ventana de medición y luego lee los FPS de esa misma ventana.

    Retorna (gpu_pct, ram_pct, emc_pct, min_fps, n_streams). tegrastats corre en paralelo;
    docker logs --since <measure_s> recorta justo la ventana (descarta el warmup).
    """
    # stdbuf -oL: sin esto, tegrastats bufferea stdout en bloques de 4KB al escribir a un pipe
    # (en vez de línea-por-línea como cuando corre en una terminal interactiva) — al matarlo con
    # terminate(), el último bloque sin flushear se pierde, dejando EMC_FREQ/GR3D_FREQ vacíos o
    # con solo la primera línea que alcanzó a salir. stdbuf (coreutils) fuerza flush por línea.
    teg = subprocess.Popen(["stdbuf", "-oL", "tegrastats", "--interval", "1000"],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    time.sleep(measure_s)
    teg.terminate()
    try:
        teg_out, _ = teg.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        teg.kill()
        teg_out, _ = teg.communicate()

    logs = subprocess.run(["docker", "logs", BENCH_CONTAINER, "--since", f"{measure_s}s"],
                         capture_output=True, text=True, check=False)
    gpu, ram, emc = _parse_tegrastats(teg_out)
    min_fps, n_streams = _parse_fps(logs.stdout + logs.stderr)
    return gpu, ram, emc, min_fps, n_streams


# ── Barrido ───────────────────────────────────────────────────────────────────

def _verdict(min_fps, n_streams, n_target, gpu, ram, target_fps, args) -> bool:
    """Decide si una corrida 'aguanta': todos los streams vivos, en tiempo real, y sin saturar."""
    if min_fps is None or n_streams < n_target:
        return False   # algún stream no arrancó (límite de decode o de sesiones del DVR)
    if min_fps < args.fps_margin * target_fps:
        return False   # compute-bound: el pipeline no alcanza el FPS de la fuente
    if gpu is not None and gpu >= args.gpu_max:
        return False
    if ram is not None and ram >= args.ram_max:
        return False
    return True


def _sweep_variant(name: str, base_channels: list, counts: list, target_fps: float, args):
    """Corre el barrido de N para una modalidad; imprime la tabla y retorna el máximo N que aguanta."""
    print(f"\n{'='*72}\n  Modalidad: {name}   (target tiempo real: {target_fps:.1f} fps × {args.fps_margin})\n{'='*72}")
    print(f"  {'N':>3} | {'min FPS/stream':>14} | {'streams':>7} | {'GPU%':>5} | {'RAM%':>5} | {'EMC%':>5} | veredicto")
    print(f"  {'-'*3}-+-{'-'*14}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+---------")

    max_ok = 0
    for n in counts:
        _write_bench_config(base_channels, n, VARIANTS[name])
        if not _run_pipeline(args.start_timeout):
            print(f"  {n:>3} | {'--':>14} | {'0':>7} | {'--':>5} | {'--':>5} | {'--':>5} | ERROR (no arrancó)")
            break
        gpu, ram, emc, min_fps, n_streams = _measure(args.measure)
        _stop_bench_container()

        ok = _verdict(min_fps, n_streams, n, gpu, ram, target_fps, args)
        fps_s = f"{min_fps:.1f}" if min_fps is not None else "--"
        gpu_s = f"{gpu:.0f}" if gpu is not None else "--"
        ram_s = f"{ram:.0f}" if ram is not None else "--"
        emc_s = f"{emc:.0f}" if emc is not None else "--"
        print(f"  {n:>3} | {fps_s:>14} | {n_streams:>7} | {gpu_s:>5} | {ram_s:>5} | {emc_s:>5} | {'OK' if ok else 'NO'}")
        if ok:
            max_ok = n
        else:
            break   # una vez que falla, subir N solo empeora — cortar la modalidad

    print(f"\n  → Máx cámaras sostenibles ({name}): {max_ok}")
    return max_ok


def main():
    """Orquesta el benchmark: sondea la fuente, barre N por cada modalidad, imprime el resumen."""
    ap = argparse.ArgumentParser(description="Mide cuántas cámaras aguanta el Jetson por modalidad")
    ap.add_argument("--client", default=None,
                    help="Cliente base a copiar (default: lee /etc/nx_client)")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help=f"Modalidades a medir, coma-separadas. Opciones: {list(VARIANTS)}")
    ap.add_argument("--list-variants", action="store_true",
                    help="Imprime qué config aplica cada modalidad (osnet_precision/sgie_interval) y termina")
    ap.add_argument("--counts", default=None,
                    help="Cuentas de cámaras a probar, coma-separadas (ej. 1,2,4,6,8)")
    ap.add_argument("--max", type=int, default=None, help="Alternativa a --counts: probar hasta este N")
    ap.add_argument("--step", type=int, default=2, help="Paso para --max (default 2)")
    ap.add_argument("--target-fps", type=float, default=None,
                    help="FPS objetivo de tiempo real (default: sondear la fuente con gst-discoverer)")
    ap.add_argument("--fps-margin", type=float, default=0.9,
                    help="Fracción del FPS de la fuente que se exige sostener (default 0.9)")
    ap.add_argument("--gpu-max", type=float, default=95.0, help="GPU%% mediana máxima aceptable (default 95)")
    ap.add_argument("--ram-max", type=float, default=90.0, help="RAM%% máxima aceptable (default 90)")
    ap.add_argument("--start-timeout", type=int, default=300,
                    help="Máx segundos esperando el 1er FPS (cubre build de engine fp16) (default 300)")
    ap.add_argument("--measure", type=int, default=45, help="Segundos de medición por corrida (default 45)")
    args = ap.parse_args()

    if args.list_variants:
        _print_variants()
        return

    # Preflight: esto corre en el host del Jetson.
    for tool in ("docker", "tegrastats"):
        if shutil.which(tool) is None:
            sys.exit(f"[ERR] '{tool}' no está en PATH. Este script corre en el host del Jetson.")

    base_client = args.client or (Path("/etc/nx_client").read_text().strip()
                                  if Path("/etc/nx_client").exists() else "demo")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"[ERR] Modalidad desconocida '{v}'. Opciones: {list(VARIANTS)}")

    if args.counts:
        counts = [int(c) for c in args.counts.split(",")]
    elif args.max:
        counts = sorted(set([1] + list(range(args.step, args.max + 1, args.step))))
    else:
        counts = [1, 2, 4, 6, 8]

    global _prod_was_up
    _prod_was_up = _prod_is_up()
    atexit.register(_cleanup)   # restaura producción + borra temporales pase lo que pase

    print(f"Cliente base : {base_client}")
    print(f"Modalidades  : {variants}")
    print(f"Cuentas (N)  : {counts}")

    base_channels = _setup_bench_client(base_client)
    print(f"Canales base : {base_channels} (se ciclan para llegar a cada N)")

    target_fps = args.target_fps or _source_fps(base_client)
    if not target_fps:
        sys.exit("[ERR] Sin FPS objetivo. Pasa --target-fps <n> (ej. 15 para main, 7 para sub).")

    if _prod_was_up:
        print("\n[setup] Deteniendo deepstream de producción para no contender por GPU/DVR…")
        _compose("stop", "deepstream", check=False)

    results = {}
    for name in variants:
        results[name] = _sweep_variant(name, base_channels, counts, target_fps, args)

    print(f"\n{'='*72}\n  RESUMEN — máx cámaras sostenibles por modalidad (target {target_fps:.1f} fps)\n{'='*72}")
    for name in variants:
        print(f"  {name:>12} : {results[name]}")
    print()


if __name__ == "__main__":
    main()
