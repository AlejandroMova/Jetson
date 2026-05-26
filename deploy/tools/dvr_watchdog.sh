#!/bin/bash
# dvr_watchdog.sh — NX Computing AI | DVR IP Auto-Recovery Service
#
# Corre como daemon systemd en el host del Jetson (fuera de Docker).
# Revisa cada 10 s los logs del container 'deepstream' buscando errores RTSP.
# Cuando TODOS los streams configurados fallan en una ventana de 30 s
# (señal de que el DVR cambió de IP por DHCP), hace:
#   1. nmap -p 554 en la subred /24 del DVR actual
#   2. Si encuentra una IP distinta → escribe /etc/nx_dvr_ip → docker restart deepstream
#   3. Si no encuentra nada → espera COOLDOWN s y reintenta
#
# Instalado por setup.sh a /usr/local/bin/nx_dvr_watchdog.sh (con @@WORK_DIR@@ sustituido).
# Ver logs: journalctl -u nx-dvr-watchdog -f

set -euo pipefail

# @@WORK_DIR@@ es sustituido por setup.sh con la ruta real del repo (deploy/).
WORK_DIR="@@WORK_DIR@@"
DVR_IP_FILE="/etc/nx_dvr_ip"
CONTAINER="deepstream"
POLL_INTERVAL=10     # segundos entre revisiones de logs
FAILURE_WINDOW=30    # ventana de tiempo (s) sobre la que se cuentan fallos RTSP
COOLDOWN=300         # segundos de espera tras un intento de redescubrimiento

log()  { echo "[nx-dvr-watchdog $(date '+%H:%M:%S')] $*"; }
warn() { log "WARN: $*"; }

# ── Devuelve el número de canales RTSP configurados para el cliente activo ────
# Lee /etc/nx_client para saber el nombre del cliente y luego parsea channels[]
# en clients/<cliente>/config.yaml. Devuelve 0 si no se puede determinar.
get_n_channels() {
    local client
    client=$(cat /etc/nx_client 2>/dev/null | tr -d '[:space:]') || true
    [[ -z "$client" ]] && echo 0 && return
    python3 - <<PYEOF 2>/dev/null || echo 0
import yaml
try:
    with open("${WORK_DIR}/clients/${client}/config.yaml") as f:
        cfg = yaml.safe_load(f)
    print(len(cfg.get("channels", [])))
except Exception:
    print(0)
PYEOF
}

# ── Escanea la subred /24 del DVR actual buscando un host con puerto 554 abierto ──
# Imprime la nueva IP si es distinta a CURRENT_IP, o nada si no encontró cambio.
find_new_dvr_ip() {
    local current="$1"

    # Derivar la subred /24 a partir de la IP actual del DVR
    local subnet
    subnet=$(python3 -c "
import ipaddress
print(ipaddress.ip_network('${current}/24', strict=False))
" 2>/dev/null) || {
        warn "No se pudo derivar subred de ${current}"
        return
    }

    log "Escaneando ${subnet} buscando DVR en puerto 554 (nmap -T4)..."

    # -T4: agresivo (más rápido en LAN, ~15 s para /24)
    # -oG: formato grep-friendly para parsear IPs fácilmente
    nmap -p 554 "$subnet" --open -T4 -oG - 2>/dev/null \
        | awk '/^Host:/ && /554\/open/ { print $2 }' \
        | grep -Fv "$current" \
        | head -1
}

# ── Bucle principal ───────────────────────────────────────────────────────────
log "Iniciado. Container: '${CONTAINER}' | poll: ${POLL_INTERVAL}s | ventana: ${FAILURE_WINDOW}s"

while true; do

    # Saltar si el container no está corriendo
    if ! docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Obtener número de canales; si no se puede leer, esperar y reintentar
    N=$(get_n_channels)
    if [[ "$N" -eq 0 ]]; then
        warn "No se puede leer número de canales (/etc/nx_client o config.yaml falta) — reintentando en 30s"
        sleep 30
        continue
    fi

    # Leer logs del container de los últimos FAILURE_WINDOW segundos
    recent=$(docker logs --since "${FAILURE_WINDOW}s" "$CONTAINER" 2>&1 || echo "")

    # Contar cuántos source-N únicos fallaron en esa ventana
    # El log de app.py emite: WARNING ... RTSP 'source-N' failed: ... — pipeline continues.
    n_failed=$(echo "$recent" \
        | grep "RTSP '.*' failed" \
        | grep -oE "source-[0-9]+" \
        | sort -u \
        | wc -l)

    if [[ "$n_failed" -ge "$N" ]]; then
        log "Todos los ${N} streams fallaron en los últimos ${FAILURE_WINDOW}s — iniciando redescubrimiento del DVR"

        current_ip=$(tr -d '[:space:]' < "$DVR_IP_FILE" 2>/dev/null || echo "")
        if [[ -z "$current_ip" ]]; then
            warn "No se puede leer IP actual de ${DVR_IP_FILE} — esperando ${COOLDOWN}s"
            sleep "$COOLDOWN"
            continue
        fi

        new_ip=$(find_new_dvr_ip "$current_ip" || echo "")

        if [[ -n "$new_ip" ]]; then
            log "DVR encontrado en nueva IP: ${current_ip} → ${new_ip}"
            echo "$new_ip" > "$DVR_IP_FILE"
            log "Escrito ${DVR_IP_FILE} → ${new_ip}"
            log "Reiniciando container '${CONTAINER}'..."
            docker restart "$CONTAINER"
            log "Container reiniciado. El pipeline se reconectará con IP ${new_ip}."
        else
            log "No se encontró DVR en la subred de ${current_ip}. ¿DVR apagado o subnet distinto?"
            log "Reintentando en ${COOLDOWN}s."
        fi

        # Cooldown antes de volver a monitorear — evita loops agresivos si el DVR sigue caído
        sleep "$COOLDOWN"
    else
        sleep "$POLL_INTERVAL"
    fi

done
