"""
jetson_sync_client.py — NX Computing AI | Face Roster Sync Client

Maintains a persistent Socket.IO connection to the backend's /jetson namespace.
When the backend emits a "face_update" event (e.g. an employee was activated or
revoked), this client triggers FaceRecognizer.sync_from_backend() to pull the
updated roster from GET /api/employees/embeddings.

Architecture: same lifecycle pattern as WsPositionClient (start/stop).
  - python-socketio Client manages the connection and a background thread.
  - On "face_update": sync is dispatched to a separate thread so the socketio
    event loop is never blocked by the HTTP pull + file write.

Why Socket.IO and not polling:
  - The backend (Railway cloud) cannot initiate HTTP connections to Jetsons
    (on-premises, behind NAT/firewall). The Socket.IO connection is Jetson-initiated,
    so the backend can push through the existing persistent channel.
  - Polling every minute would add unnecessary load and delay; this approach
    triggers sync immediately on employee activation with zero wasted requests.
"""
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class JetsonSyncClient:
    """
    Socket.IO client that listens for face_update events from the backend.

    On each face_update event: calls the supplied sync_callback in a
    background thread so this client's event loop is not blocked.

    The connection authenticates using the Jetson's X-API-Key sent in the
    Socket.IO auth dict — same key used for REST API calls.
    """

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        sync_callback: Callable[[str, str], None],
    ):
        """Configura el cliente de sync. La conexión se establece en start().

        Args:
            api_base_url:   HTTP base URL del backend (e.g. https://api.nxcomputing.com).
            api_key:        API key del Jetson (raw, sin hash).
            sync_callback:  Función llamada en hilo separado cuando llega face_update.
                            Firma: callback(action: str, employee_id: str).
        """
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key
        self._sync_callback = sync_callback
        self._sio = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Arranca el hilo de conexión al backend. La conexión Socket.IO se establece asíncronamente."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="jetson-sync-client"
        )
        self._thread.start()
        logger.info("[JetsonSyncClient] starting → %s/jetson", self._api_base_url)

    def stop(self) -> None:
        """Desconecta Socket.IO y espera a que el hilo pare (máximo 5 s)."""
        self._running = False
        if self._sio is not None:
            try:
                self._sio.disconnect()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[JetsonSyncClient] stopped.")

    # ── Internal connection thread ─────────────────────────────────────────────

    def _run(self) -> None:
        """Hilo principal: inicializa el cliente Socket.IO, registra handlers y conecta.

        python-socketio maneja la reconexión automáticamente con reconnection=True.
        El hilo simplemente llama sio.wait() para mantener el bucle de eventos activo.
        """
        try:
            import socketio  # python-socketio[client]
        except ImportError:
            logger.error(
                "[JetsonSyncClient] python-socketio no está instalado. "
                "Instala con: pip install python-socketio. Face sync deshabilitado."
            )
            return

        self._sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,   # reintentar indefinidamente
            reconnection_delay=2,
            reconnection_delay_max=30,
            logger=False,
            engineio_logger=False,
        )

        # ── Registrar handlers ───────────────────────────────────────────────

        @self._sio.on("connect", namespace="/jetson")
        def on_connect():
            logger.info("[JetsonSyncClient] conectado al backend /jetson — disparando sync inicial.")
            # Sync en hilo separado para no bloquear el event loop de socketio.
            threading.Thread(
                target=self._sync_callback, args=("sync", ""),
                daemon=True, name="face-sync-on-connect"
            ).start()

        @self._sio.on("disconnect", namespace="/jetson")
        def on_disconnect():
            logger.info("[JetsonSyncClient] desconectado del backend.")

        @self._sio.on("face_update", namespace="/jetson")
        def on_face_update(data: dict):
            """Recibe face_update del backend y dispara sync en hilo separado."""
            action = data.get("action", "sync")
            employee_id = data.get("employee_id", "")
            logger.info("[JetsonSyncClient] face_update recibido: action=%s employee_id=%s",
                        action, employee_id)
            threading.Thread(
                target=self._sync_callback, args=(action, employee_id),
                daemon=True, name="face-sync-worker"
            ).start()

        # ── Conectar y mantener el bucle ─────────────────────────────────────

        while self._running:
            try:
                self._sio.connect(
                    self._api_base_url,
                    namespaces=["/jetson"],
                    auth={"api_key": self._api_key},
                    wait_timeout=10,
                )
                self._sio.wait()  # bloquea hasta disconnect
            except Exception as exc:
                if self._running:
                    logger.debug("[JetsonSyncClient] error de conexión: %s — python-socketio reintentará.", exc)
                # python-socketio ya gestiona reconexión; si estamos aquí es porque
                # falló el connect() inicial — esperar un poco antes de reintentar.
                if self._running:
                    import time
                    time.sleep(5)
            finally:
                # Asegurarse de que el socket quede en estado limpio para el próximo intento.
                try:
                    if self._sio.connected:
                        self._sio.disconnect()
                except Exception:
                    pass
