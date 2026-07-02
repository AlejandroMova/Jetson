#!/bin/bash
# dvr_watchdog.sh — NX Computing AI | DVR IP Auto-Recovery Service
#
# Corre como daemon systemd en el host del Jetson (fuera de Docker).
# Verifica directamente, cada POLL_INTERVAL s, si la IP del DVR configurada en
# /etc/nx_dvr_ip sigue respondiendo en el puerto RTSP del cliente. Tras FAILURE_THRESHOLD
# chequeos consecutivos fallidos (señal de que el DVR cambió de IP por DHCP), hace:
#   1. nmap -p <puerto> en la subred /24 de la IP actual
#   2. Si encuentra una IP distinta → escribe /etc/nx_dvr_ip → docker restart deepstream
#   3. Si no encuentra nada → espera COOLDOWN s y reintenta
#
# NOTA histórica: la versión anterior parseaba `docker logs` buscando líneas
# "RTSP 'source-N' failed" y comparaba el conteo contra los canales de config.yaml.
# Se abandonó porque ese conteo no coincidía con los streams reales cuando el cliente
# tenía `external_channels` configurados — el watchdog nunca disparaba aunque todas
# las cámaras reales fallaran (ver ErrorHistory.md 2026-07-01). El chequeo directo de
# conectividad no depende de cuántas cámaras haya ni del formato de logs del pipeline.
#
# Instalado por setup.sh a /usr/local/bin/nx_dvr_watchdog.sh (con @@WORK_DIR@@ sustituido).
# Ver logs: journalctl -u nx-dvr-watchdog -f

set -euo pipefail

# @@WORK_DIR@@ es sustituido por setup.sh con la ruta real del repo (deploy/).
WORK_DIR="@@WORK_DIR@@"
DVR_IP_FILE="/etc/nx_dvr_ip"
# Container name includes the Docker Compose project prefix (e.g. "deploy-deepstream-1").
# Detect dynamically each iteration — don't hardcode the prefix.
CONTAINER_PATTERN="deepstream"
POLL_INTERVAL=10        # segundos entre chequeos de conectividad al DVR
CHECK_TIMEOUT=3         # timeout (s) por intento de conexión TCP al puerto RTSP
FAILURE_THRESHOLD=3     # chequeos consecutivos fallidos antes de asumir cambio de IP
COOLDOWN=300            # segundos de espera tras un intento de redescubrimiento

log()  { echo "[nx-dvr-watchdog $(date '+%H:%M:%S')] $*"; }
warn() { log "WARN: $*"; }

# Resolves the actual running container name that matches CONTAINER_PATTERN.
# Docker Compose prepends the project directory name (e.g. "deploy-deepstream-1").
# Returns empty string if no matching container is currently running.
get_container() {
    docker ps --filter "name=${CONTAINER_PATTERN}" --format "{{.Names}}" | head -1
}

# ── Devuelve el puerto RTSP configurado para el cliente activo ─────────────────
# Intenta leer /etc/nx_client + el campo `dvr_port` de su config.yaml primero;
# 554 solo entra como fallback (cliente desconocido, campo ausente, o error de
# lectura) — mismo default que usa config_loader.py en el pipeline, para que
# ambos lados queden sincronizados sin duplicar la lógica de canales que causó
# el bug anterior (esto es un solo entero simple, no una lista a filtrar).
get_dvr_port() {
    local client
    client=$(cat /etc/nx_client 2>/dev/null | tr -d '[:space:]') || true
    if [[ -z "$client" ]]; then
        echo 554
        return
    fi
    python3 - <<PYEOF 2>/dev/null || echo 554
import yaml
try:
    with open("${WORK_DIR}/clients/${client}/config.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    print(int(cfg.get("dvr_port", 554)))
except Exception:
    print(554)
PYEOF
}

# ── Verifica si el DVR responde en el puerto RTSP ──────────────────────────────
# Usa /dev/tcp de bash (sin depender de nc/ncat u otras herramientas externas —
# cero dependencias nuevas en el host, ver regla 6/7 de CLAUDE.md). Retorna 0 si
# logra conectar dentro de CHECK_TIMEOUT, distinto de 0 si falla o hace timeout.
check_dvr_reachable() {
    local ip="$1"
    local port="$2"
    timeout "$CHECK_TIMEOUT" bash -c ": > /dev/tcp/${ip}/${port}" 2>/dev/null
}

# ── Escanea la subred /24 del DVR actual buscando un host con el puerto abierto ──
# Imprime la nueva IP si es distinta a CURRENT_IP, o nada si no encontró un candidato.
find_new_dvr_ip() {
    local current="$1"
    local port="$2"

    # Derivar la subred /24 a partir de la IP actual del DVR
    local subnet
    subnet=$(python3 -c "
import ipaddress
print(ipaddress.ip_network('${current}/24', strict=False))
" 2>/dev/null) || {
        warn "No se pudo derivar subred de ${current}"
        return
    }

    log "Escaneando ${subnet} buscando DVR en puerto ${port} (nmap -T4)..."

    # -T4: agresivo (más rápido en LAN, ~15 s para /24)
    # -oG: formato grep-friendly para parsear IPs fácilmente
    nmap -p "$port" "$subnet" --open -T4 -oG - 2>/dev/null \
        | awk -v openmark="${port}/open" 'index($0, openmark) && /^Host:/ { print $2 }' \
        | grep -Fv "$current" \
        | head -1
}

# ── Bucle principal ───────────────────────────────────────────────────────────
DVR_PORT=$(get_dvr_port)
log "Iniciado. Verificando DVR:${DVR_PORT} cada ${POLL_INTERVAL}s (timeout ${CHECK_TIMEOUT}s, umbral ${FAILURE_THRESHOLD} fallos consecutivos)"

consecutive_failures=0

while true; do

    current_ip=$(tr -d '[:space:]' < "$DVR_IP_FILE" 2>/dev/null || echo "")
    if [[ -z "$current_ip" ]]; then
        warn "No se puede leer IP actual de ${DVR_IP_FILE} — reintentando en ${POLL_INTERVAL}s"
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Chequeo directo: ¿la IP configurada sigue respondiendo en el puerto RTSP?
    if check_dvr_reachable "$current_ip" "$DVR_PORT"; then
        consecutive_failures=0
        sleep "$POLL_INTERVAL"
        continue
    fi

    # No respondió — puede ser un blip momentáneo de red, no necesariamente un
    # cambio de IP. Exigimos FAILURE_THRESHOLD fallos seguidos antes de actuar.
    consecutive_failures=$((consecutive_failures + 1))
    warn "DVR ${current_ip}:${DVR_PORT} no responde (intento ${consecutive_failures}/${FAILURE_THRESHOLD})"

    if [[ "$consecutive_failures" -lt "$FAILURE_THRESHOLD" ]]; then
        sleep "$POLL_INTERVAL"
        continue
    fi

    log "DVR ${current_ip} no responde tras ${FAILURE_THRESHOLD} intentos seguidos — iniciando redescubrimiento"
    new_ip=$(find_new_dvr_ip "$current_ip" "$DVR_PORT" || echo "")

    if [[ -n "$new_ip" ]]; then
        log "DVR encontrado en nueva IP: ${current_ip} → ${new_ip}"
        echo "$new_ip" > "$DVR_IP_FILE"
        log "Escrito ${DVR_IP_FILE} → ${new_ip}"

        CONTAINER=$(get_container)
        if [[ -n "$CONTAINER" ]]; then
            log "Reiniciando container '${CONTAINER}'..."
            docker restart "$CONTAINER"
            log "Container reiniciado. El pipeline se reconectará con IP ${new_ip}."
        else
            warn "No se encontró container '${CONTAINER_PATTERN}' corriendo — IP corregida, sin reinicio."
        fi
    else
        log "No se encontró DVR en la subred de ${current_ip}. ¿Está apagado o cambió de subred?"
        log "Reintentando en ${COOLDOWN}s."
    fi

    # Cooldown antes de volver a monitorear — evita loops agresivos si el DVR sigue caído
    consecutive_failures=0
    sleep "$COOLDOWN"

done
