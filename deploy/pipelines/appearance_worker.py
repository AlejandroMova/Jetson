"""
appearance_worker.py — NX Computing AI | Cross-Camera Re-ID Worker

Runs OSNet-x0.25 ONNX in a background thread to generate 512-dim L2-normalized
appearance embeddings for each detected person. Used for cross-camera re-ID by the backend.

Architecture: same non-blocking queue pattern as PoseWorker / NxApiClient.
  probe → enqueue(crop_bgr, track_id, frame_num)   ← O(1)
  worker thread → resize + normalize + OSNet ONNX → 512-dim L2-norm vector
  probe (next frame) → get_result(track_id) → np.ndarray | None

Input: BGR crop of any size (resized internally to 128×256)
Output: 512-dim float32 vector, L2-normalized (cosine sim = dot product)

Model: OSNet-x0.25 from torchreid (KaiyangZhou), Apache 2.0 license
       Input: NCHW float32, RGB, ImageNet-normalized, 3×256×128
       Output: (1, 512) float32
Download: python3 tools/download_models.py --reid
"""
import logging
import queue
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_HEIGHT = 256
INPUT_WIDTH  = 128


class AppearanceWorker:
    """
    Non-blocking OSNet ONNX worker for appearance embeddings.
    Once a result is ready for a track_id it is cached until get_result() is called.
    """

    def __init__(self, model_path: str, queue_size: int = 64):
        """Configura el worker. El modelo ONNX se carga en start(), no aquí.

        La carga diferida es necesaria porque TensorRT inicializa su contexto CUDA
        al hacer pipeline.set_state(PLAYING). Si cargamos ONNX antes, hay conflictos
        de contexto CUDA entre TRT y ONNX Runtime.

        _results usa (pad_index, track_id) como clave porque DeepStream asigna
        track_ids locales por cámara — dos cámaras pueden tener el mismo track_id.
        """
        self._model_path = model_path
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        # Clave (pad_index, track_id) para diferenciar el mismo track_id en distintas cámaras
        self._results: Dict[Tuple[int, int], np.ndarray] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._session = None  # se carga en start() después de que TRT inicialice CUDA

    def start(self) -> None:
        """Carga el modelo ONNX y arranca el hilo worker. Llamar después de pipeline.set_state(PLAYING)."""
        self._session = self._load_model()  # debe correr después de set_state(PLAYING)
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="appearance-worker"
        )
        self._thread.start()
        logger.info("AppearanceWorker started — model: %s", self._model_path)

    def stop(self) -> None:
        """Señaliza al worker que pare y espera a que el hilo termine (máximo 5 s)."""
        self._running = False
        self._queue.put(None)  # sentinel: desbloquea el get() bloqueante en _worker_loop
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("AppearanceWorker stopped.")

    def enqueue(self, crop_bgr: np.ndarray, track_id: int, pad_index: int, frame_num: int) -> None:
        """Encola un crop para extracción de embedding. No bloqueante — descarta si la cola está llena."""
        try:
            self._queue.put_nowait((crop_bgr.copy(), track_id, pad_index, frame_num))
        except queue.Full:
            pass  # descarte silencioso — el probe intentará en el próximo frame

    def get_result(self, track_id: int, pad_index: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._results.get((pad_index, track_id))

    def clear_result(self, track_id: int, pad_index: int) -> None:
        with self._lock:
            self._results.pop((pad_index, track_id), None)

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Loop principal del hilo worker: consume la cola y almacena resultados en _results.

        Usa get(timeout=1.0) para no bloquearse indefinidamente y poder detectar
        la señal de parada (_running=False) aunque no haya items en la cola.
        """
        if self._session is None:
            logger.error("AppearanceWorker: ONNX model failed to load — worker inactive.")
            return

        while self._running:
            try:
                item = self._queue.get(timeout=1.0)  # timeout permite verificar _running periódicamente
            except queue.Empty:
                continue
            if item is None:
                break  # sentinel de parada enviado por stop()

            crop_bgr, track_id, pad_index, frame_num = item
            try:
                vec = self._infer(crop_bgr)
                if vec is not None:
                    with self._lock:
                        # Guardar con clave compuesta para no confundir tracks de distintas cámaras
                        self._results[(pad_index, track_id)] = vec
                    logger.debug("Appearance vector computed pad=%d track=%d frame=%d",
                                 pad_index, track_id, frame_num)
            except Exception as e:
                logger.warning("AppearanceWorker error track=%d: %s", track_id, e)
            self._queue.task_done()

    def _load_model(self):
        """Carga el modelo OSNet ONNX en CPU. Retorna la sesión ONNX o None si falla."""
        try:
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]  # CPU para no competir con TRT por la GPU
            sess = ort.InferenceSession(self._model_path, providers=providers)
            logger.info("OSNet ONNX loaded (providers: %s)", sess.get_providers())
            return sess
        except Exception as e:
            logger.error("Failed to load OSNet ONNX from %s: %s", self._model_path, e)
            return None

    def _infer(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Ejecuta OSNet sobre un crop BGR y retorna el vector de apariencia L2-normalizado.

        El vector resultante tiene 512 dimensiones y norma ≈ 1.0, lo que hace que
        el dot product sea equivalente a la similitud coseno (sin necesidad de dividir por normas).
        Esto es lo que permite a ReIdManager comparar embeddings con un simple producto matricial.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        # ── Preprocesamiento ─────────────────────────────────────────────────
        img = cv2.resize(crop_bgr, (INPUT_WIDTH, INPUT_HEIGHT))                 # escalar a 128×256 (W×H)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # BGR→RGB, normalizar [0,1]
        img = (img - _MEAN) / _STD                                              # normalización ImageNet estándar
        inp = img.transpose(2, 0, 1)[np.newaxis]                                # HWC → NCHW: (1, 3, 256, 128)

        # ── Inferencia ONNX ──────────────────────────────────────────────────
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: inp})[0]  # salida: (1, 512)
        vec = output[0].astype(np.float32)                      # extraer el vector del batch

        # ── L2-normalización ─────────────────────────────────────────────────
        # Normalizar para que sim coseno = dot product (evita división en cada comparación)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:              # guard contra vector nulo (no debería ocurrir con OSNet)
            vec = vec / norm
        return vec
