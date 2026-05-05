"""
ws_client.py — NX Computing AI | WebSocket Position Stream Client

Sends positions_snapshot frames to the backend over a persistent WebSocket connection.
The Jetson is the client; the backend is the server at WS_BASE_URL/ws/positions.

Why WebSocket and not REST for positions:
  - Persistent TCP connection: handshake once, then ~2-10 bytes overhead per message
  - REST: ~300 bytes HTTP headers per request × 96 req/min (16 cameras × 6/min) = wasted bandwidth
  - positions_snapshot is telemetry (not critical), no per-message idempotency needed
  - Auto-reconnect with exponential backoff: 1s → 2s → 4s → ... → 30s max
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

JETSON_ID: str = os.environ.get("JETSON_ID", os.uname().nodename)


class WsPositionClient:
    """
    Async WebSocket client that sends positions_snapshot messages.
    Falls back silently if the connection is unavailable.
    """

    _MAX_BACKOFF: float = 30.0
    _INITIAL_BACKOFF: float = 1.0

    def __init__(self, ws_url: str, api_key: str, sector: str = "comercio"):
        self._ws_url = ws_url.rstrip("/") + "/ws/positions"
        self._api_key = api_key
        self._sector = sector
        self._ws = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._connect_loop, daemon=True, name="ws-position-client"
        )
        self._thread.start()
        logger.info("WsPositionClient starting → %s", self._ws_url)

    def stop(self) -> None:
        self._running = False
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("WsPositionClient stopped.")

    # ── Public API ────────────────────────────────────────────────────────────

    def send_positions(self, camera_id: str, positions: List[dict]) -> None:
        """Enqueue a positions_snapshot frame. Non-blocking; drops if no connection."""
        if not positions:
            return
        msg = {
            "type": "positions_snapshot",
            "sector": self._sector,
            "jetson_id": JETSON_ID,
            "camera_id": camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": positions,
        }
        with self._lock:
            ws = self._ws
        if ws is None:
            return
        try:
            ws.send(json.dumps(msg))
            logger.debug("WS positions sent camera=%s n=%d", camera_id, len(positions))
        except Exception as e:
            logger.debug("WS send failed (%s) — will reconnect", e)
            with self._lock:
                self._ws = None

    # ── Internal reconnect loop ───────────────────────────────────────────────

    def _connect_loop(self) -> None:
        try:
            import websocket  # websocket-client library
        except ImportError:
            logger.error(
                "websocket-client not installed. "
                "Run: pip install websocket-client. WS positions disabled."
            )
            return

        backoff = self._INITIAL_BACKOFF
        while self._running:
            try:
                ws = websocket.WebSocket()
                ws.connect(
                    self._ws_url,
                    header={"X-API-Key": self._api_key},
                    timeout=10,
                )
                with self._lock:
                    self._ws = ws
                logger.info("WS connected → %s", self._ws_url)
                backoff = self._INITIAL_BACKOFF  # reset on success

                # Keep connection alive until it drops
                while self._running:
                    try:
                        ws.ping()
                        time.sleep(15)
                    except Exception:
                        break

            except Exception as e:
                logger.debug("WS connect failed: %s — retry in %.0fs", e, backoff)

            with self._lock:
                self._ws = None

            if self._running:
                time.sleep(backoff)
                backoff = min(backoff * 2, self._MAX_BACKOFF)
