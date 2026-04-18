#!/bin/bash
# ============================================================
#  NX Computing — Jetson Remote Access Setup
#  Configura: SSH · Tailscale · VNC · Docker
#
#  Uso:
#    sudo bash setup.sh
#
#  Flags opcionales:
#    --authkey    Auth key de Tailscale
#    --compose    Nombre del archivo compose (default: docker-compose.yml)
#    --hostname   Nombre visible en Tailscale (default: hostname actual)
#    --no-vnc     Omite la instalación de VNC
#    --no-docker  Omite docker compose
#
#  Prerequisito:
#    Clonar el repo manualmente antes de correr este script:
#    git clone https://<token>@github.com/AlejandroMova/NX-JETSON.git
#    cd NX-JETSON && sudo bash setup.sh --authkey <key>
# ============================================================

set -eo pipefail

# ── Colores ─────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[NX]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[ERR]${NC} $*"; exit 1; }

# ── Defaults ────────────────────────────────────────────────
TS_AUTHKEY=""
COMPOSE_FILE="docker-compose.yml"
TS_HOSTNAME="$(hostname)"
SKIP_VNC=false
SKIP_DOCKER=false

# El script corre desde la carpeta del repo
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Parse args ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --authkey)   TS_AUTHKEY="$2";   shift 2 ;;
    --compose)   COMPOSE_FILE="$2"; shift 2 ;;
    --hostname)  TS_HOSTNAME="$2";  shift 2 ;;
    --no-vnc)    SKIP_VNC=true;     shift ;;
    --no-docker) SKIP_DOCKER=true;  shift ;;
    *) die "Flag desconocido: $1" ;;
  esac
done

[[ $EUID -ne 0 ]] && die "Corre con sudo"

echo -e "\n${BOLD}══════════════════════════════════════════${NC}"
echo -e "${BOLD}   NX Computing — Jetson Setup v2.2       ${NC}"
echo -e "${BOLD}   Host: $(hostname) | $(date '+%Y-%m-%d %H:%M')${NC}"
echo -e "${BOLD}══════════════════════════════════════════${NC}\n"

# ════════════════════════════════════════════════════════════
# 1. SSH
# ════════════════════════════════════════════════════════════
log "Configurando SSH..."

apt-get install -y openssh-server -qq

systemctl enable ssh
systemctl start ssh

SSHD_CONF="/etc/ssh/sshd_config"

grep -q "^PermitRootLogin" "$SSHD_CONF" \
  && sed -i 's/^PermitRootLogin.*/PermitRootLogin no/' "$SSHD_CONF" \
  || echo "PermitRootLogin no" >> "$SSHD_CONF"

grep -q "^X11Forwarding" "$SSHD_CONF" \
  && sed -i 's/^X11Forwarding.*/X11Forwarding yes/' "$SSHD_CONF" \
  || echo "X11Forwarding yes" >> "$SSHD_CONF"

systemctl restart ssh
ok "SSH activo en puerto 22"

# ════════════════════════════════════════════════════════════
# 2. Tailscale
# ════════════════════════════════════════════════════════════
log "Instalando Tailscale..."

if ! command -v tailscale &>/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
  ok "Tailscale instalado"
else
  ok "Tailscale ya estaba instalado ($(tailscale version | head -1))"
fi

systemctl enable --now tailscaled

if [[ -n "$TS_AUTHKEY" ]]; then
  log "Autenticando con Tailscale..."
  tailscale up \
    --authkey="$TS_AUTHKEY" \
    --hostname="$TS_HOSTNAME" \
    --accept-routes \
    --ssh
  ok "Tailscale conectado"
  TS_IP=$(tailscale ip -4 2>/dev/null || echo "pendiente")
  log "IP Tailscale: ${BOLD}${TS_IP}${NC}"
else
  log "Iniciando Tailscale login — abre el link en tu browser y luego regresa aquí..."
  tailscale up --hostname="$TS_HOSTNAME" --accept-routes --ssh || true
  TS_IP=$(tailscale ip -4 2>/dev/null || echo "no conectado")
  if [[ "$TS_IP" != "no conectado" ]]; then
    ok "Tailscale conectado — IP: ${BOLD}${TS_IP}${NC}"
  else
    warn "Tailscale no autenticado todavía. Corre: sudo tailscale up"
  fi
fi

# ════════════════════════════════════════════════════════════
# 3. Login automático (necesario para VNC sin acceso físico)
# ════════════════════════════════════════════════════════════
log "Configurando login automático..."

REAL_USER=$(logname 2>/dev/null || who | awk '{print $1}' | head -1 || echo "nxcomputingdemo")
GDM_CONF="/etc/gdm3/custom.conf"
mkdir -p /etc/gdm3

# Habilita AutomaticLoginEnable (maneja lineas comentadas)
if grep -q "AutomaticLoginEnable" "$GDM_CONF" 2>/dev/null; then
  sed -i "s/.*AutomaticLoginEnable.*/AutomaticLoginEnable=true/" "$GDM_CONF"
else
  sed -i "/^\[daemon\]/a AutomaticLoginEnable=true" "$GDM_CONF"
fi

# Habilita AutomaticLogin (maneja lineas comentadas con formato "= user1")
if grep -q "AutomaticLogin" "$GDM_CONF" 2>/dev/null; then
  sed -i "s/.*AutomaticLogin.*/AutomaticLogin=${REAL_USER}/" "$GDM_CONF"
  # Elimina duplicados si quedaron dos lineas
  awk '!seen[$0]++' "$GDM_CONF" > /tmp/gdm_tmp && mv /tmp/gdm_tmp "$GDM_CONF"
else
  sed -i "/^\[daemon\]/a AutomaticLogin=${REAL_USER}" "$GDM_CONF"
fi

ok "Login automático configurado para: ${BOLD}${REAL_USER}${NC}"

# ════════════════════════════════════════════════════════════
# 4. VNC (x11vnc)
# ════════════════════════════════════════════════════════════
if [[ "$SKIP_VNC" == false ]]; then
  log "Instalando VNC (x11vnc)..."

  apt-get install -y x11vnc -qq

  # ── Detecta display y Xauthority automáticamente ─────────
  log "Detectando display activo..."

  # Busca el número de display desde el proceso de Xorg
  # Busca el display activo del proceso Xorg — múltiples patrones para compatibilidad
  XORG_NUM=$(ps aux | grep -oP '(?<=Xorg vt\d ):\d+' | head -1 || true)
  if [[ -z "$XORG_NUM" ]]; then
    XORG_NUM=$(ps aux | grep -oP '(?<=Xorg ):\d+' | head -1 || true)
  fi
  if [[ -z "$XORG_NUM" ]]; then
    XORG_NUM=$(ps aux | grep Xorg | grep -oP ':\d+' | head -1 || true)
  fi
  # Fallback a :1 (más común en JetPack/Ubuntu con GDM)
  XORG_DISPLAY="${XORG_NUM:-:1}"

  # Busca el Xauthority desde el proceso de Xorg
  XAUTH_FILE=$(ps aux | grep -oP '(?<=-auth )/run/user/\d+/gdm/Xauthority' | head -1 || true)
  if [[ -z "$XAUTH_FILE" ]]; then
    # Busca en ubicaciones comunes
    for f in /run/user/*/gdm/Xauthority /var/lib/gdm3/:0.Xauth /var/gdm/:0.Xauth; do
      if [[ -f "$f" ]]; then
        XAUTH_FILE="$f"
        break
      fi
    done
  fi
  XAUTH_FILE="${XAUTH_FILE:-/run/user/1000/gdm/Xauthority}"

  ok "Display: ${BOLD}${XORG_DISPLAY}${NC} | Xauthority: ${BOLD}${XAUTH_FILE}${NC}"

  # ── Contraseña VNC ────────────────────────────────────────
  VNC_PASS_FILE="/etc/x11vnc.pass"
  if [[ ! -f "$VNC_PASS_FILE" ]]; then
    VNC_PASS=$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 12 || true)
    printf '%s\n%s\n' "$VNC_PASS" "$VNC_PASS" | x11vnc -storepasswd /dev/stdin "$VNC_PASS_FILE" 2>/dev/null || \
      x11vnc -storepasswd "$VNC_PASS" "$VNC_PASS_FILE" 2>/dev/null || true
    chmod 600 "$VNC_PASS_FILE"
    warn "╔══════════════════════════════════════╗"
    warn "  Contraseña VNC: ${BOLD}${VNC_PASS}${NC}"
    warn "  Guárdala en 1Password ahora."
    warn "╚══════════════════════════════════════╝"
  else
    ok "Contraseña VNC ya existe"
  fi

  # ── Servicio systemd con display detectado ────────────────
  cat > /etc/systemd/system/x11vnc.service << EOF
[Unit]
Description=NX VNC Server (x11vnc)
After=network.target display-manager.service
Wants=display-manager.service

[Service]
Type=simple
ExecStartPre=/bin/sleep 5
ExecStart=/usr/bin/x11vnc \\
  -display ${XORG_DISPLAY} \\
  -auth ${XAUTH_FILE} \\
  -rfbauth /etc/x11vnc.pass \\
  -rfbport 5900 \\
  -forever \\
  -loop \\
  -noxdamage \\
  -repeat \\
  -shared
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable x11vnc
  systemctl restart x11vnc
  sleep 2

  if systemctl is-active x11vnc -q; then
    ok "VNC corriendo en background en puerto 5900"
  else
    warn "VNC no pudo arrancar — revisa: sudo systemctl status x11vnc"
  fi

  log "Conectar desde VNC Viewer: ${BOLD}<tailscale-ip>:5900${NC}"
else
  warn "VNC omitido (--no-vnc)"
fi

# ════════════════════════════════════════════════════════════
# 4. Docker Compose
# ════════════════════════════════════════════════════════════
if [[ "$SKIP_DOCKER" == false ]]; then
  log "Arrancando contenedores Docker desde: ${WORK_DIR}"

  if ! command -v docker &>/dev/null; then
    die "Docker no está instalado. Instala con: curl -fsSL https://get.docker.com | sh"
  fi

  COMPOSE_PATH="${WORK_DIR}/${COMPOSE_FILE}"

  if [[ ! -f "$COMPOSE_PATH" ]]; then
    warn "No se encontró ${COMPOSE_PATH} — saltando Docker."
    warn "Agrega docker-compose.yml al repo para que arranque automático."
  else
    cd "$WORK_DIR"

    if docker compose version &>/dev/null 2>&1; then
      COMPOSE_CMD="docker compose"
    elif command -v docker-compose &>/dev/null; then
      COMPOSE_CMD="docker-compose"
    else
      die "docker compose no encontrado."
    fi

    log "Usando: ${COMPOSE_CMD}"
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d --build --remove-orphans

    ok "Contenedores arriba"
    echo ""
    $COMPOSE_CMD -f "$COMPOSE_FILE" ps
  fi
else
  warn "Docker omitido (--no-docker)"
fi

# ════════════════════════════════════════════════════════════
# 5. Resumen final
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}══════════════════════════════════════════${NC}"
echo -e "${BOLD}   Resumen — $(hostname)${NC}"
echo -e "${BOLD}══════════════════════════════════════════${NC}"

LOCAL_IP=$(hostname -I | awk '{print $1}')
TS_IP=$(tailscale ip -4 2>/dev/null || echo "no conectado")

echo -e "  IP local:     ${BOLD}${LOCAL_IP}${NC}"
echo -e "  IP Tailscale: ${BOLD}${TS_IP}${NC}"
echo ""
echo -e "  ${GREEN}SSH:${NC}  ssh NxComputingDemo@${TS_IP}"
echo -e "  ${GREEN}VNC:${NC}  ${TS_IP}:5900  (VNC Viewer)"
echo ""

for svc in ssh tailscaled x11vnc; do
  STATUS=$(systemctl is-active "$svc" 2>/dev/null || echo "inactive")
  if [[ "$STATUS" == "active" ]]; then
    echo -e "  ${GREEN}●${NC} ${svc}"
  else
    echo -e "  ${RED}○${NC} ${svc} (${STATUS})"
  fi
done

if [[ "$SKIP_DOCKER" == false ]] && command -v docker &>/dev/null; then
  echo ""
  log "Contenedores activos:"
  docker ps --format "  {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}Setup completado.${NC}"
echo ""