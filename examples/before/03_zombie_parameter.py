# Source: deploy/pipelines/face_recognizer.py — _worker_loop + _process
# Anti-patterns illustrated:
#   - "Zombie parameter": camera_id is unpacked from the queue item but never forwarded
#     to _process(). The reader has no way to know if this is intentional or a bug.
#   - Spanish string literal "Desconocido" in data that flows to the backend.
#     If the backend or any downstream consumer expects "Unknown", this breaks silently.
#   - Spanish docstring in a file whose module docstring is English.

def _worker_loop(self):
    """Loop principal del hilo worker: consume la cola y acumula votos de identidad.

    Usa get(timeout=1.0) para no bloquearse indefinidamente y poder detectar _running=False.
    Cada item procesado llama a _process(), que internamente aplica el sistema de votos.
    """
    if self._app is None:
        logger.error("FaceRecognizer: failed to load InsightFace — worker inactive.")
        return

    while self._running:
        try:
            item = self._queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            break
        face_crop, track_id, frame_num, camera_id = item  # camera_id silently dropped
        try:
            self._process(face_crop, track_id)
        except Exception as e:
            logger.warning("FaceRecognizer error track=%d: %s", track_id, e)
        self._queue.task_done()


# Excerpt from _process — the Spanish literal that reaches _locked and potentially the backend
        if name is None:
            name = "Desconocido"  # por debajo del threshold → cara desconocida
