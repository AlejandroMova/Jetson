#!/bin/bash
# update.sh — NX Computing AI | Pipeline Update Script
#
# Run this on the Jetson to pull the latest code and restart the pipeline.
# Automatically detects whether a Docker rebuild is needed.
#
# Usage:
#   bash tools/update.sh
#   bash tools/update.sh --force-rebuild   # rebuilds image even if not needed

set -eo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[NX]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[ERR]${NC} $*"; exit 1; }

FORCE_REBUILD=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --force-rebuild) FORCE_REBUILD=true; shift ;;
    *) die "Unknown flag: $1" ;;
  esac
done

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORK_DIR"

COMPOSE_FILE="docker-compose.yml"
[[ -f "$COMPOSE_FILE" ]] || die "$COMPOSE_FILE not found. Run from repo root or tools/ directory."

echo -e "\n${BOLD}══════════════════════════════════════════${NC}"
echo -e "${BOLD}   NX Computing — Pipeline Update         ${NC}"
echo -e "${BOLD}   $(date '+%Y-%m-%d %H:%M')              ${NC}"
echo -e "${BOLD}══════════════════════════════════════════${NC}\n"

# ── 1. Pull latest code ───────────────────────────────────────────────────────
log "Pulling latest code from GitHub..."
git fetch origin

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [[ "$LOCAL" == "$REMOTE" ]]; then
  ok "Already up to date ($(git rev-parse --short HEAD))"
  if [[ "$FORCE_REBUILD" == false ]]; then
    warn "Nothing changed — use --force-rebuild to rebuild anyway."
    exit 0
  fi
fi

# Check which files changed since last pull
CHANGED=$(git diff --name-only HEAD origin/main)
echo ""
log "Changed files:"
echo "$CHANGED" | sed 's/^/    /'
echo ""

git pull origin main
ok "Code updated to $(git rev-parse --short HEAD)"

# ── 1b. Modelos faltantes (idempotente — skip si ya existen) ─────────────────
OSNET_PATH="${WORK_DIR}/models/osnet/osnet_x0_25_market1501.onnx"
if [[ ! -f "$OSNET_PATH" ]]; then
  log "OSNet no encontrado — instalando torchreid y exportando..."
  pip3 install --quiet torchreid
  python3 "${WORK_DIR}/tools/download_models.py" --reid \
    && ok "OSNet exportado" \
    || warn "Falló. Manual: pip3 install torchreid && python3 tools/download_models.py --reid"
fi

# ── 2. Decide: restart only, or full rebuild ──────────────────────────────────
NEEDS_REBUILD=false

if [[ "$FORCE_REBUILD" == true ]]; then
  NEEDS_REBUILD=true
  log "Rebuild forced via --force-rebuild"
elif echo "$CHANGED" | grep -qE '^(Dockerfile\.jetson|docker-entrypoint\.sh|requirements\.txt)'; then
  NEEDS_REBUILD=true
  log "Dockerfile or requirements changed — rebuild needed"
else
  log "Only pipeline code or configs changed — restart is enough"
fi

# ── 3. Apply update ───────────────────────────────────────────────────────────
if [[ "$NEEDS_REBUILD" == true ]]; then
  log "Rebuilding Docker image (this takes ~5 min)..."
  docker compose -f "$COMPOSE_FILE" build deepstream
  ok "Image rebuilt"
fi

log "Restarting deepstream container..."
docker compose -f "$COMPOSE_FILE" restart deepstream

# Wait a moment then check it came up
sleep 5
STATUS=$(docker compose -f "$COMPOSE_FILE" ps --format json deepstream 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['State'])" 2>/dev/null || echo "unknown")

echo ""
echo -e "${BOLD}══════════════════════════════════════════${NC}"
if [[ "$STATUS" == "running" ]]; then
  ok "deepstream is running"
else
  warn "deepstream status: ${STATUS}"
  warn "Check logs: docker compose -f $COMPOSE_FILE logs --tail=50 deepstream"
fi

CLIENT=$(cat /etc/nx_client 2>/dev/null || echo "unknown")
DVR_IP=$(cat /etc/nx_dvr_ip 2>/dev/null || echo "unknown")
TS_IP=$(tailscale ip -4 2>/dev/null || echo "no conectado")

echo -e "  Client  : ${BOLD}${CLIENT}${NC}"
echo -e "  DVR IP  : ${BOLD}${DVR_IP}${NC}"
echo -e "  RTSP    : ${BOLD}rtsp://${TS_IP}:8554/ds-test${NC}"
echo -e "${BOLD}══════════════════════════════════════════${NC}"
echo ""
