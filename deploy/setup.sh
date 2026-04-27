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
#    --client     Nombre del cliente (escribe /etc/nx_client, e.g. demo)
#    --package    Paquete contratado (escribe /etc/nx_pipeline con las capabilities)
#                 Opciones: comercio_basico | comercio_avanzado | comercio_total |
#                           industrial_basico | industrial_avanzado | industrial_total |
#                           hogar_basico | hogar_avanzado | hogar_total
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
NX_CLIENT=""
NX_PACKAGE=""

# Mapeo paquete → capabilities (comma-separated, written to /etc/nx_pipeline)
declare -A PACKAGE_CAPABILITIES=(
  [comercio_basico]="people_counting"
  [comercio_avanzado]="people_counting"
  [comercio_total]="people_counting,age_gender"
  [comercio_enterprise]="people_counting,age_gender"
  [industrial_basico]="people_counting"
  [industrial_avanzado]="people_counting,epp_detection"
  [industrial_total]="people_counting,epp_detection,license_plate,fire_smoke"
  [industrial_enterprise]="people_counting,epp_detection,license_plate,fire_smoke"
  [hogar_basico]="people_counting"
  [hogar_avanzado]="people_counting,fall_detection"
  [hogar_total]="people_counting,fall_detection,fire_smoke"
)

# El script corre desde la carpeta del repo
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Parse args ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --authkey)   TS_AUTHKEY="$2";   shift 2 ;;
    --compose)   COMPOSE_FILE="$2"; shift 2 ;;
    --hostname)  TS_HOSTNAME="$2";  shift 2 ;;
    --client)    NX_CLIENT="$2";    shift 2 ;;
    --package)   NX_PACKAGE="$2";   shift 2 ;;
    --no-vnc)    SKIP_VNC=true;     shift ;;
    --no-docker) SKIP_DOCKER=true;  shift ;;
    *) die "Flag desconocido: $1" ;;
  esac
done

[[ $EUID -ne 0 ]] && die "Corre con sudo"

echo -e "\n${BOLD}══════════════════════════════════════════${NC}"
echo -e "${BOLD}   NX Computing — Jetson Setup v2.5       ${NC}"
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
    --accept-routes
  ok "Tailscale conectado"
  TS_IP=$(tailscale ip -4 2>/dev/null || echo "pendiente")
  log "IP Tailscale: ${BOLD}${TS_IP}${NC}"
else
  log "Iniciando Tailscale login — abre el link en tu browser y luego regresa aquí..."
  tailscale up --hostname="$TS_HOSTNAME" --accept-routes || true
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

# Escribe el bloque [daemon] completo con autologin — evita problemas con líneas comentadas
if grep -q "^\[daemon\]" "$GDM_CONF" 2>/dev/null; then
  # Elimina líneas previas de AutomaticLogin para evitar duplicados
  sed -i '/.*AutomaticLogin.*/d' "$GDM_CONF"
  # Inserta las dos líneas limpias después de [daemon]
  sed -i "/^\[daemon\]/a AutomaticLogin=${REAL_USER}\nAutomaticLoginEnable=true" "$GDM_CONF"
else
  printf '[daemon]\nAutomaticLoginEnable=true\nAutomaticLogin=%s\n' "$REAL_USER" >> "$GDM_CONF"
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
  # Detecta display desde /tmp/.X11-unix (más confiable que ps)
  XORG_DISPLAY=$(ls /tmp/.X11-unix/ 2>/dev/null | head -1 | sed 's/X/:/')
  XORG_DISPLAY="${XORG_DISPLAY:-:0}"

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

  # ── xorg.conf headless (fuerza resolución sin monitor físico) ─
  log "Configurando xorg.conf para operación headless..."
  XORG_CONF="/etc/X11/xorg.conf"

  if ! grep -q "DummyMonitor" "$XORG_CONF" 2>/dev/null; then
    cat > "$XORG_CONF" << 'XORGEOF'
Section "Module"
    Disable     "dri"
    SubSection  "extmod"
        Option  "omit xfree86-dga"
    EndSubSection
EndSection

Section "Device"
    Identifier  "Tegra0"
    Driver      "nvidia"
    Option      "AllowEmptyInitialConfiguration" "true"
EndSection

Section "Monitor"
    Identifier "DummyMonitor"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
    Modeline "1920x1080" 172.80 1920 2040 2248 2576 1080 1081 1084 1118
EndSection

Section "Screen"
    Identifier "Default Screen"
    Device "Tegra0"
    Monitor "DummyMonitor"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1920x1080"
    EndSubSection
EndSection
XORGEOF
    ok "xorg.conf configurado para headless 1920x1080"
  else
    ok "xorg.conf headless ya estaba configurado"
  fi

  # ── Servicio de sistema que corre como el usuario ────────────
  chmod 644 /etc/x11vnc.pass

  cat > /etc/systemd/system/x11vnc.service << EOF
[Unit]
Description=x11vnc VNC Server
After=graphical.target
Wants=graphical.target

[Service]
Type=simple
User=${REAL_USER}
Environment=DISPLAY=${XORG_DISPLAY}
Environment=XAUTHORITY=${XAUTH_FILE}
ExecStartPre=/bin/sleep 10
ExecStartPre=/bin/sh -c 'fuser -k 5900/tcp 2>/dev/null || true'
ExecStartPre=/bin/sh -c 'DISPLAY=${XORG_DISPLAY} XAUTHORITY=${XAUTH_FILE} xrandr --fb 1920x1080 2>/dev/null || true'
ExecStart=/usr/bin/x11vnc -display ${XORG_DISPLAY} -auth ${XAUTH_FILE} -rfbauth /etc/x11vnc.pass -rfbport 5900 -forever -shared -noxdamage -noshm
Restart=always
RestartSec=5

[Install]
WantedBy=graphical.target
EOF

  systemctl daemon-reload
  systemctl enable x11vnc
  systemctl restart x11vnc
  sleep 3

  if systemctl is-active x11vnc -q; then
    ok "VNC corriendo en background en puerto 5900"
  else
    warn "VNC arrancara despues del proximo reboot con sesion grafica activa"
  fi

  log "Conectar desde VNC Viewer: ${BOLD}<tailscale-ip>:5900${NC}"
else
  warn "VNC omitido (--no-vnc)"
fi

# ════════════════════════════════════════════════════════════
# 4. Descubrimiento de DVR
# ════════════════════════════════════════════════════════════
log "Buscando DVR en la red local..."

if ! command -v nmap &>/dev/null; then
  apt-get install -y nmap -qq
fi

LOCAL_IFACE=$(ip route | grep default | awk '{print $5}' | head -1)
LOCAL_SUBNET=$(ip -o -f inet addr show "$LOCAL_IFACE" 2>/dev/null | awk '{print $4}' | head -1)

DVR_IP=""
if [[ -n "$LOCAL_SUBNET" ]]; then
  log "Escaneando ${LOCAL_SUBNET} en busca de puerto 554..."
  DVR_IP=$(nmap -p 554 --open -oG - "$LOCAL_SUBNET" 2>/dev/null \
    | awk '/554\/open/{print $2}' | head -1)
fi

if [[ -n "$DVR_IP" ]]; then
  ok "DVR encontrado: ${BOLD}${DVR_IP}${NC}"
  echo "$DVR_IP" > /etc/nx_dvr_ip
  ok "IP guardada en /etc/nx_dvr_ip"
else
  warn "No se encontró DVR con puerto 554 en ${LOCAL_SUBNET:-red local}"
  warn "Verifica conexión del DVR o corre: nmap -p 554 --open <subnet>"
  DVR_IP="no encontrado"
fi

# ════════════════════════════════════════════════════════════
# 5. Cliente NX + Paquete contratado
# ════════════════════════════════════════════════════════════
log "Registrando nombre de cliente..."

if [[ -n "$NX_CLIENT" ]]; then
  echo "$NX_CLIENT" > /etc/nx_client
  ok "Cliente guardado en /etc/nx_client: ${BOLD}${NX_CLIENT}${NC}"
elif [[ -f /etc/nx_client ]]; then
  ok "Cliente ya registrado: ${BOLD}$(cat /etc/nx_client)${NC}"
else
  warn "/etc/nx_client no existe. Usa: sudo bash setup.sh --client <nombre>"
  warn "O escríbelo manualmente: echo 'demo' | sudo tee /etc/nx_client"
fi

# ── Paquete contratado → capabilities del pipeline ───────────────────────────
log "Registrando paquete contratado..."

if [[ -z "$NX_PACKAGE" && -t 0 ]]; then
  # Modo interactivo: mostrar menú si no se pasó --package
  echo ""
  echo -e "${BOLD}Selecciona el paquete contratado:${NC}"
  echo "  1) comercio_basico      — Conteo de personas"
  echo "  2) comercio_avanzado    — Conteo + analytics de backend"
  echo "  3) comercio_total       — + Clasificación edad/género"
  echo "  4) industrial_basico    — Conteo de personas"
  echo "  5) industrial_avanzado  — + Detección EPP (requiere modelo)"
  echo "  6) industrial_total     — + Placas + Fuego (requieren modelos)"
  echo "  7) hogar_basico         — Detección de personas"
  echo "  8) hogar_avanzado       — + Detección de caídas (requiere modelo)"
  echo "  9) hogar_total          — + Detección de incendios (requiere modelo)"
  echo "  0) Omitir (configurar después)"
  echo ""
  read -rp "Opción [0-9]: " _pkg_opt
  case $_pkg_opt in
    1) NX_PACKAGE="comercio_basico" ;;
    2) NX_PACKAGE="comercio_avanzado" ;;
    3) NX_PACKAGE="comercio_total" ;;
    4) NX_PACKAGE="industrial_basico" ;;
    5) NX_PACKAGE="industrial_avanzado" ;;
    6) NX_PACKAGE="industrial_total" ;;
    7) NX_PACKAGE="hogar_basico" ;;
    8) NX_PACKAGE="hogar_avanzado" ;;
    9) NX_PACKAGE="hogar_total" ;;
    *) NX_PACKAGE="" ;;
  esac
fi

if [[ -n "$NX_PACKAGE" ]]; then
  _caps="${PACKAGE_CAPABILITIES[$NX_PACKAGE]}"
  if [[ -z "$_caps" ]]; then
    warn "Paquete desconocido: '${NX_PACKAGE}'. Usa uno de: ${!PACKAGE_CAPABILITIES[*]}"
  else
    echo "$_caps" > /etc/nx_pipeline
    ok "Paquete '${BOLD}${NX_PACKAGE}${NC}' → capabilities: ${BOLD}${_caps}${NC}"
    ok "Guardado en /etc/nx_pipeline"
  fi
elif [[ -f /etc/nx_pipeline ]]; then
  ok "Pipeline ya configurado: ${BOLD}$(cat /etc/nx_pipeline)${NC}"
else
  warn "/etc/nx_pipeline no configurado — el pipeline usará el valor de config.yaml"
  warn "Para configurar después: echo 'people_counting,age_gender' | sudo tee /etc/nx_pipeline"
fi

# ════════════════════════════════════════════════════════════
# 6. Docker Compose
# ════════════════════════════════════════════════════════════
if [[ "$SKIP_DOCKER" == false ]]; then
  log "Configurando pipeline Docker desde: ${WORK_DIR}"

  if ! command -v docker &>/dev/null; then
    die "Docker no está instalado. Instala con: curl -fsSL https://get.docker.com | sh"
  fi

  COMPOSE_PATH="${WORK_DIR}/${COMPOSE_FILE}"

  if [[ ! -f "$COMPOSE_PATH" ]]; then
    warn "No se encontró ${COMPOSE_PATH} — saltando Docker."
  else
    cd "$WORK_DIR"

    if docker compose version &>/dev/null 2>&1; then
      COMPOSE_CMD="docker compose"
    elif command -v docker-compose &>/dev/null; then
      COMPOSE_CMD="docker-compose"
    else
      die "docker compose no encontrado."
    fi

    # ── 6a. Build ────────────────────────────────────────────
    log "Construyendo imagen Docker (~10 min la primera vez)..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" build deepstream
    ok "Imagen construida"

    # ── 6b. Auto-identificar DVR (patrón URL + resolución) ───
    CLIENT_NAME=$(cat /etc/nx_client 2>/dev/null || echo "")
    ENV_FILE="${WORK_DIR}/clients/${CLIENT_NAME}/.env"

    if [[ -n "$CLIENT_NAME" && -f "$ENV_FILE" && "$DVR_IP" != "no encontrado" ]]; then
      # Crear directorio y config.yaml mínimo si no existen
      CLIENT_DIR="${WORK_DIR}/clients/${CLIENT_NAME}"
      mkdir -p "$CLIENT_DIR"
      if [[ ! -f "${CLIENT_DIR}/config.yaml" ]]; then
        cat > "${CLIENT_DIR}/config.yaml" << CFGEOF
dvr_port: 554
rtsp_url_pattern: ""
stream_width: 1920
stream_height: 1080
channels: []
CFGEOF
        ok "config.yaml inicial creado en ${CLIENT_DIR}/"
      fi

      log "Identificando marca/patrón del DVR..."
      if $COMPOSE_CMD -f "$COMPOSE_FILE" run --rm deepstream \
          python3 tools/identify_dvr.py --update-config; then
        ok "Patrón de URL del DVR identificado y config.yaml actualizado"
      else
        warn "No se pudo identificar el patrón de URL del DVR (IP sí encontrada: ${DVR_IP})."
        warn "Corre manualmente: docker compose run --rm deepstream python3 tools/identify_dvr.py --update-config"
      fi

      log "Detectando canales activos..."
      if $COMPOSE_CMD -f "$COMPOSE_FILE" run --rm deepstream \
          python3 tools/probe_cameras.py --update-config; then
        ok "Canales activos detectados y config.yaml actualizado"
      else
        warn "No se pudieron detectar los canales activos."
        warn "Corre manualmente: docker compose run --rm deepstream python3 tools/probe_cameras.py --update-config"
      fi
    else
      if [[ -z "$CLIENT_NAME" ]]; then
        warn "Sin /etc/nx_client — saltando identificación de DVR. Usa --client <nombre>"
      elif [[ ! -f "$ENV_FILE" ]]; then
        warn "Sin credenciales DVR en ${ENV_FILE} — saltando identificación de DVR."
        warn "Crea el archivo y corre: docker compose run --rm deepstream python3 tools/identify_dvr.py --update-config"
      fi
    fi

    # ── 6c. Arrancar pipeline ────────────────────────────────
    log "Arrancando pipeline..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d --remove-orphans
    ok "Pipeline arriba"
    echo ""
    $COMPOSE_CMD -f "$COMPOSE_FILE" ps
  fi
else
  warn "Docker omitido (--no-docker)"
fi

# ════════════════════════════════════════════════════════════
# 7. Resumen final
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}══════════════════════════════════════════${NC}"
echo -e "${BOLD}   Resumen — $(hostname)${NC}"
echo -e "${BOLD}══════════════════════════════════════════${NC}"

LOCAL_IP=$(hostname -I | awk '{print $1}')
TS_IP=$(tailscale ip -4 2>/dev/null || echo "no conectado")

echo -e "  IP local:     ${BOLD}${LOCAL_IP}${NC}"
echo -e "  IP Tailscale: ${BOLD}${TS_IP}${NC}"
echo -e "  IP DVR:       ${BOLD}${DVR_IP}${NC}"
echo -e "  Cliente NX:   ${BOLD}$(cat /etc/nx_client 2>/dev/null || echo 'no configurado')${NC}"
echo -e "  Pipeline:     ${BOLD}$(cat /etc/nx_pipeline 2>/dev/null || echo 'default (config.yaml)')${NC}"
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