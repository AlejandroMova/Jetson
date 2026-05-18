#!/bin/bash
# qa.sh — NX Computing AI | QA Visual
#
# Arranca el pipeline con NX_QA_ENABLED=true + dashboard Streamlit.
# Accesible vía Tailscale desde cualquier dispositivo del equipo NX.
#
# Uso:
#   ./qa.sh           # arrancar QA (Ctrl+C para detener y restaurar producción)
#   ./qa.sh stop      # detener QA y restaurar producción (desde otra terminal)

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

# ── STOP ──────────────────────────────────────────────────────────────────────
if [ "$CMD" = "stop" ]; then
    echo ""
    echo "  Deteniendo modo QA..."
    docker compose -f docker-compose.yml -f docker-compose.qa.yml \
        stop deepstream qa_app 2>/dev/null || true
    docker compose -f docker-compose.yml -f docker-compose.qa.yml \
        rm -f qa_app 2>/dev/null || true
    echo "  Reiniciando pipeline de producción..."
    docker compose up -d
    echo ""
    echo "  Producción restaurada."
    echo ""
    exit 0
fi

# ── START ─────────────────────────────────────────────────────────────────────
echo ""
echo "  Iniciando NX QA Visual..."
echo ""

# Detener el pipeline de producción si está corriendo (para liberar GPU)
docker compose stop deepstream 2>/dev/null || true

# Construir qa_app si cambió, luego arrancar en background
docker compose \
    -f docker-compose.yml \
    -f docker-compose.qa.yml \
    up --build -d --remove-orphans deepstream qa_app redis

# ── Cleanup: solo responde a señales explícitas (INT/TERM), nunca a EXIT.
# Llamar exit 0 al final garantiza que el script termina exactamente una vez.
_cleanup() {
    trap '' INT TERM   # bloquear señales adicionales durante el cleanup
    echo ""
    echo "  Deteniendo QA y restaurando producción..."
    docker compose -f docker-compose.yml -f docker-compose.qa.yml \
        kill deepstream qa_app 2>/dev/null || true
    docker compose -f docker-compose.yml -f docker-compose.qa.yml \
        rm -f qa_app 2>/dev/null || true
    docker compose up -d 2>/dev/null || true
    echo "  Producción restaurada."
    echo ""
    exit 0
}

# Registrar SOLO INT y TERM — no EXIT.
# Con EXIT también registrado, bash dispara el handler dos veces cuando Ctrl+C
# mata docker compose logs (una por INT, otra porque set -e hace salir el script).
trap _cleanup INT TERM

# A partir de aquí los errores no deben abortar el script automáticamente:
# docker compose logs -f termina con código != 0 cuando lo mata Ctrl+C.
set +e

# Esperar a que Streamlit esté listo (máx 3 min, ping cada 3 s)
echo "  Esperando que Streamlit arranque..."
READY=0
for i in $(seq 1 60); do
    if curl -sf "http://localhost:8501/_stcore/health" > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 3
done

IP=$(_get_ip)

echo ""
if [ "$READY" = "1" ]; then
    echo "  ╔══════════════════════════════════════════════════════════╗"
    echo "  ║          NX QA Visual Dashboard — Listo                 ║"
    echo "  ╠══════════════════════════════════════════════════════════╣"
    echo "  ║                                                          ║"
    printf "  ║  Dashboard:  http://%-34s║\n" "${IP}:8501"
    printf "  ║  Stream:     http://%-34s║\n" "${IP}:8080/stream/all"
    echo "  ║                                                          ║"
    echo "  ║  Ctrl+C aquí  →  detiene QA y restaura producción       ║"
    echo "  ║  Desde otra terminal:  ./qa.sh stop                     ║"
    echo "  ╚══════════════════════════════════════════════════════════╝"
else
    echo "  AVISO: Streamlit no respondió en 90 s."
    echo "  Revisa los logs:  docker compose -f docker-compose.yml -f docker-compose.qa.yml logs qa_app"
    echo ""
    echo "  Si el pipeline arrancó, el dashboard debería estar en:"
    printf "    http://%s:8501\n" "$IP"
fi
echo ""

# Seguir los logs en background + wait: así bash recibe INT inmediatamente
# (con foreground, bash difiere el trap hasta que docker compose logs salga,
# y en algunas versiones de Compose ese proceso ignora SIGINT y cuelga).
docker compose \
    -f docker-compose.yml \
    -f docker-compose.qa.yml \
    logs -f deepstream qa_app &
LOGS_PID=$!
wait $LOGS_PID

# Si los containers se pararon solos (sin Ctrl+C), limpiar igualmente
_cleanup
