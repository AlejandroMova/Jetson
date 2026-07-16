"""
test_reid_manager.py — Chequeo standalone del buffer multi-frame y del umbral acotado
de re-match rápido agregados 2026-07-16 (calibración ronda 3) a reid_manager.py.

reid_manager.py no importa pyds/gi (solo json, logging, threading, time, uuid, numpy),
así que se puede probar sin DeepStream ni el Jetson real. No es una suite exhaustiva —
es el chequeo mínimo que falla si la lógica de match_or_create()/_find_best_match() se
rompe. Correr directo: python3 test_reid_manager.py
"""
import tempfile
import time
from pathlib import Path

import numpy as np

import reid_manager
from reid_manager import ReIdManager


def _orthonormal_pair(dim: int = 512, seed: int = 0):
    """Dos vectores unitarios ortogonales entre sí (Gram-Schmidt sobre vectores random)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=dim)
    a /= np.linalg.norm(a)
    b = rng.normal(size=dim)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    return a.astype(np.float32), b.astype(np.float32)


def _vector_at_similarity(base: np.ndarray, orthogonal: np.ndarray, sim: float) -> np.ndarray:
    """Vector unitario con similitud coseno exacta `sim` contra `base` (base y orthogonal
    deben ser ortonormales entre sí — construido así el resultado ya queda en la esfera unidad)."""
    v = sim * base + (1.0 - sim ** 2) ** 0.5 * orthogonal
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
    return v.astype(np.float32)


def _new_manager() -> ReIdManager:
    tmp = Path(tempfile.mkdtemp()) / "reid_db.json"
    return ReIdManager(db_path=str(tmp))


def test_single_ndarray_backward_compat():
    """Un solo np.ndarray (no lista) sigue funcionando como antes del cambio."""
    mgr = _new_manager()
    a, b = _orthonormal_pair(seed=1)

    gid1, event1, prev1, _ = mgr.match_or_create(a, "cam1")
    assert event1 == "new_person"

    e_match = _vector_at_similarity(a, b, 0.90)
    gid2, event2, prev2, _ = mgr.match_or_create(e_match, "cam1")
    assert gid2 == gid1, "0.90 >= SIMILARITY_THRESHOLD debería matchear"
    assert event2 == "channel_change"  # mismo cam, sim >= 0.85


def test_best_of_n_picks_the_winner():
    """De 3 embeddings en la lista, solo el 3ro matchea de verdad — debe ganar ese."""
    mgr = _new_manager()
    a, b = _orthonormal_pair(seed=2)
    gid, _, _, _ = mgr.match_or_create(a, "camA")

    e_low1 = _vector_at_similarity(a, b, 0.50)
    e_low2 = _vector_at_similarity(a, b, 0.60)
    e_good = _vector_at_similarity(a, b, 0.90)

    gid2, event, _, _ = mgr.match_or_create([e_low1, e_low2, e_good], "camA")
    assert gid2 == gid, "el mejor-de-3 (0.90) debería matchear aunque los otros 2 no"
    assert event == "channel_change"

    entry = mgr._db[gid]
    assert len(entry.gallery) == 2, "e_good debería haberse agregado a la galería (ventana de diversidad)"
    assert np.allclose(entry.gallery[-1], e_good), "la galería debe usar el embedding GANADOR, no e_low1/e_low2"


def test_quick_rematch_same_camera_recent():
    """Similitud entre QUICK_REMATCH y SIMILARITY_THRESHOLD matchea solo si misma cámara + reciente."""
    a, b = _orthonormal_pair(seed=3)
    e_mid = _vector_at_similarity(a, b, 0.78)  # entre 0.75 y 0.85
    assert reid_manager.SIMILARITY_THRESHOLD_QUICK_REMATCH <= 0.78 < reid_manager.SIMILARITY_THRESHOLD

    # Caso 1: misma cámara, 44s (< QUICK_REMATCH_WINDOW_S=45) -> debe aceptar
    mgr = _new_manager()
    gid, _, _, _ = mgr.match_or_create(a, "camX")
    mgr._db[gid].last_seen_ts = time.time() - 44
    gid2, event, _, _ = mgr.match_or_create(e_mid, "camX")
    assert gid2 == gid, "misma cámara + 44s + sim=0.78 debería aceptar por quick-rematch"
    assert event == "channel_change"

    # Caso 2: cámara distinta, mismo timing -> debe rechazar (new_person)
    mgr = _new_manager()
    gid, _, _, _ = mgr.match_or_create(a, "camX")
    mgr._db[gid].last_seen_ts = time.time() - 44
    gid3, event3, _, _ = mgr.match_or_create(e_mid, "camY")
    assert gid3 != gid, "cámara distinta no debería activar quick-rematch"
    assert event3 == "new_person"

    # Caso 3: misma cámara pero 46s (> QUICK_REMATCH_WINDOW_S) -> debe rechazar
    mgr = _new_manager()
    gid, _, _, _ = mgr.match_or_create(a, "camX")
    mgr._db[gid].last_seen_ts = time.time() - 46
    gid4, event4, _, _ = mgr.match_or_create(e_mid, "camX")
    assert gid4 != gid, "fuera de la ventana de 45s no debería activar quick-rematch"
    assert event4 == "new_person"


def test_low_similarity_never_matches_even_same_camera_recent():
    """Una similitud por debajo de SIMILARITY_THRESHOLD_QUICK_REMATCH nunca matchea,
    aunque sea la misma cámara y muy reciente."""
    mgr = _new_manager()
    a, b = _orthonormal_pair(seed=4)
    gid, _, _, _ = mgr.match_or_create(a, "camX")
    mgr._db[gid].last_seen_ts = time.time() - 1  # hace 1 segundo

    e_low = _vector_at_similarity(a, b, 0.50)
    gid2, event, _, _ = mgr.match_or_create(e_low, "camX")
    assert gid2 != gid
    assert event == "new_person"


if __name__ == "__main__":
    tests = [
        test_single_ndarray_backward_compat,
        test_best_of_n_picks_the_winner,
        test_quick_rematch_same_camera_recent,
        test_low_similarity_never_matches_even_same_camera_recent,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
