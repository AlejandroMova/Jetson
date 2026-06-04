#!/bin/bash
# stream.sh — NX Computing AI
#
# Activa stream mode: inserta nvmultistreamtiler al pipeline y sirve
# la vista tileada de todas las cámaras con bboxes e inferencia en:
#
#   http://<tailscale-ip>:8080/viewer/all
#
# Uso:
#   ./stream.sh           — arrancar stream (Ctrl+C detiene y restaura producción)
#   ./stream.sh stop      — detener stream desde otra terminal

set -euo pipefail
cd "$(dirname "$0")"

CMD="${1:-start}"

# ── IP accesible (Tailscale preferido) ───────────────────────────────────────
_get_ip() {
    local ip
    ip=$(tailscale ip -4 2>/dev/null | head -1)
    [ -z "$ip" ] && ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo "$ip"
}

# ── STOP ──────────────────────────────────────────────────────────────────────
if [ "$CMD" = "stop" ]; then
    echo ""
    echo "  Deteniendo stream mode..."
    docker compose -f docker-compose.yml -f docker-compose.stream.yml \
        stop deepstream 2>/dev/null || true
    echo "  Reiniciando pipeline de producción..."
    docker compose up -d
    echo "  Producción restaurada."
    echo ""
    exit 0
fi

# ── START ─────────────────────────────────────────────────────────────────────
echo ""
echo "  Activando NX Stream Mode (tiled con inferencia)..."
echo ""

# Eliminar y recrear para que Docker aplique la variable de entorno del override
docker compose rm -sf deepstream 2>/dev/null || true
docker compose -f docker-compose.yml -f docker-compose.stream.yml up -d deepstream

# Esperar a que el servidor MJPEG esté listo (máx 60 s)
echo "  Esperando que arranque el servidor MJPEG..."
for i in $(seq 1 30); do
    if docker logs deepstream 2>&1 | grep -q "MjpegServer en :8080"; then
        break
    fi
    sleep 2
done

IP=$(_get_ip)

echo ""
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║            NX Stream Mode — Listo                        ║"
echo "  ╠═══════════════════════════════════════════════════════════╣"
echo "  ║                                                           ║"
printf "  ║  Stream:  http://%-39s║\n" "${IP}:8080/viewer/all"
echo "  ║                                                           ║"
echo "  ║  Ctrl+C aquí  →  detiene stream y restaura producción    ║"
echo "  ║  Desde otra terminal:  ./stream.sh stop                  ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo ""

# ── Cleanup al salir ──────────────────────────────────────────────────────────
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

# Seguir logs en primer plano hasta Ctrl+C
set +e
docker compose -f docker-compose.yml -f docker-compose.stream.yml logs -f deepstream &
LOGS_PID=$!

while kill -0 "$LOGS_PID" 2>/dev/null; do
    sleep 1
done

_cleanup
