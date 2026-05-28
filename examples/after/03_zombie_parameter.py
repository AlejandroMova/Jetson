# Source: deploy/pipelines/face_recognizer.py — _worker_loop + _process
# Changes applied:
#   - camera_id → _camera_id: underscore prefix signals "intentionally not forwarded yet"
#     + NOTE comment explains why it's captured but not passed
#   - "Desconocido" → "Unknown": consistent with the English codebase and backend contract
#   - Docstring translated to English
#
# NOTE (new anti-pattern, not in skill): "Zombie parameter"
# A zombie parameter is a value that:
#   (a) arrives in the function (via queue item, function arg, etc.)
#   (b) is explicitly destructured/unpacked
#   (c) is then silently dropped — not used, not forwarded, not logged
# Unlike a dead function parameter (which the skill covers via API/docstring clarity),
# a zombie parameter creates active doubt: "was this supposed to be wired up?"
# Fix: use underscore prefix AND add a comment if the value is reserved for future use.
# If it's truly dead, remove it from the enqueue() call and the tuple entirely.

def _worker_loop(self):
    """Main worker loop: consumes the queue and accumulates identity votes.
    Uses get(timeout=1.0) to remain responsive to _running=False without blocking indefinitely.
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
        # NOTE: _camera_id reserved for future per-camera routing — not yet wired to _process.
        face_crop, track_id, frame_num, _camera_id = item
        try:
            self._process(face_crop, track_id)
        except Exception as e:
            logger.warning("FaceRecognizer error track=%d: %s", track_id, e)
        self._queue.task_done()


# Excerpt from _process
        if name is None:
            name = "Unknown"  # below threshold — unrecognised face
