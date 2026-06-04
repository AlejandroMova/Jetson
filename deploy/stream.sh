#!/bin/bash
# stream.sh — NX Computing AI
#
# Activa el stream MJPEG con bounding boxes y labels en tiempo real.
# No hay Streamlit ni dashboard — solo el video con overlays accesible via HTTP.
#
# Uso:
#   ./stream.sh           # arrancar stream (Ctrl+C para detener y restaurar producción)
#   ./stream.sh stop      # detener stream y restaurar producción (desde otra terminal)

set -euo pipefail
cd "$(dirname "$0")"

CMD="${1:-start}"

# ── Obtener IP accesible (Tailscale > IP local) ───────────────────────────────
_get_ip() {
    local ip
    ip=$(tailscale ip -4 2>/dev/null | head -1)
    if [ -z "$ip" ]; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    echo "$ip"
}

# ── Leer canales activos desde config del cliente ─────────────────────────────
_get_channels() {
    local client
    client=$(cat /etc/nx_client 2>/dev/null || echo "demo")
    local config="clients/${client}/config.yaml"
    if [ -f "$config" ]; then
        # Extraer la lista de channels del yaml (línea tipo: channels: [ch01, ch02])
        grep -oP '(?<=channels:\s)\[.*?\]' "$config" 2>/dev/null || echo ""
    fi
}

# ── STOP ──────────────────────────────────────────────────────────────────────
if [ "$CMD" = "stop" ]; then
    echo ""
    echo "  Deteniendo stream mode..."
    docker compose -f docker-compose.yml -f docker-compose.stream.yml \
        stop deepstream 2>/dev/null || true
    echo "  Reiniciando pipeline de producción..."
    docker compose up -d
    echo ""
    echo "  Producción restaurada."
    echo ""
    exit 0
fi

# ── START ─────────────────────────────────────────────────────────────────────
echo ""
echo "  Activando NX Stream Mode..."
echo ""

# Eliminar el container para que Docker lo recree con NX_STREAM_ENABLED del override.
# Un simple stop no aplica cambios de entorno del override file.
docker compose rm -sf deepstream 2>/dev/null || true

docker compose \
    -f docker-compose.yml \
    -f docker-compose.stream.yml \
    up -d deepstream

# ── Cleanup ───────────────────────────────────────────────────────────────────
_cleanup() {
    trap '' INT TERM
    echo ""
    echo "  Deteniendo stream y restaurando producción..."
    docker compose -f docker-compose.yml -f docker-compose.stream.yml \
        kill deepstream 2>/dev/null || true
    docker compose up -d || true
    echo "  Producción restaurada."
    echo ""
    exit 0
}
trap _cleanup INT TERM

# Esperar a que el servidor MJPEG esté listo (máx 60 s, ping cada 2 s)
echo "  Esperando que el stream arranque..."
READY=0
for i in $(seq 1 30); do
    if curl -sf "http://localhost:8080/stream/" > /dev/null 2>&1 || \
       docker compose -f docker-compose.yml -f docker-compose.stream.yml \
           exec -T deepstream curl -sf "http://localhost:8080/" > /dev/null 2>&1 || \
       docker logs deepstream 2>&1 | grep -q "StreamServer iniciado" ; then
        READY=1
        break
    fi
    sleep 2
done

IP=$(_get_ip)

# Leer canales para mostrar URLs individuales
CLIENT=$(cat /etc/nx_client 2>/dev/null || echo "demo")
CONFIG="clients/${CLIENT}/config.yaml"

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║              NX Stream Mode — Listo                     ║"
echo "  ╠══════════════════════════════════════════════════════════╣"
echo "  ║                                                          ║"
printf "  ║  Streams:    http://%-34s║\n" "${IP}:8080"

# Mostrar URL por cada cámara si podemos leer los canales
if [ -f "$CONFIG" ]; then
    CHANNELS=$(python3 -c "
import yaml, sys
try:
    with open('$CONFIG') as f:
        cfg = yaml.safe_load(f)
    channels = cfg.get('channels', [])
    jetson = open('/etc/nx_client').read().strip() if __import__('os').path.exists('/etc/nx_client') else 'jetson'
    for ch in channels:
        print(f'  ║    /viewer/{jetson}-{ch}' + ' ' * max(0, 38 - len(f'/viewer/{jetson}-{ch}')) + '║')
except Exception:
    pass
" 2>/dev/null || true)
    if [ -n "$CHANNELS" ]; then
        echo "$CHANNELS"
    fi
fi

echo "  ║                                                          ║"
echo "  ║  Ctrl+C aquí  →  detiene stream y restaura producción   ║"
echo "  ║  Desde otra terminal:  ./stream.sh stop                 ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

set +e
docker compose \
    -f docker-compose.yml \
    -f docker-compose.stream.yml \
    logs -f deepstream &
LOGS_PID=$!

while kill -0 "$LOGS_PID" 2>/dev/null; do
    sleep 1
done

_cleanup
