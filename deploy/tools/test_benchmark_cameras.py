#!/usr/bin/env python3
"""
test_benchmark_cameras.py — self-check de las funciones puras de benchmark_cameras.py

Solo cubre el parsing y la lógica de veredicto (lo que se rompe silenciosamente si
cambia un regex o un umbral). No arranca Docker ni el pipeline. Correr en el host:
    python3 tools/test_benchmark_cameras.py
"""

from types import SimpleNamespace

import benchmark_cameras as b


def test_cycle_channels():
    assert b._cycle_channels([1, 2, 3], 8) == [1, 2, 3, 1, 2, 3, 1, 2]
    assert b._cycle_channels([], 3) == [1, 1, 1]      # fallback a canal 1
    assert b._cycle_channels([5], 2) == [5, 5]


def test_parse_tegrastats():
    sample = (
        "RAM 3200/7772MB EMC_FREQ 10% GR3D_FREQ 40%\n"
        "RAM 3600/7772MB EMC_FREQ 20% GR3D_FREQ 60%\n"
    )
    gpu, ram, emc = b._parse_tegrastats(sample)
    assert gpu == 50            # mediana de 40,60
    assert emc == 15            # mediana de 10,20
    assert abs(ram - 100 * 3400 / 7772) < 0.5   # mediana de las dos RAM%
    assert b._parse_tegrastats("linea sin nada") == (None, None, None)


def test_parse_fps():
    logs = (
        "BENCH_FPS stream=0 fps=15.0\nBENCH_FPS stream=0 fps=13.0\n"
        "BENCH_FPS stream=1 fps=9.0\n"
    )
    min_fps, n = b._parse_fps(logs)
    assert n == 2
    assert min_fps == 9.0        # min entre medianas (stream0=14, stream1=9)
    assert b._parse_fps("nada") == (None, 0)


def test_verdict():
    args = SimpleNamespace(fps_margin=0.9, gpu_max=95.0, ram_max=90.0)
    # Aguanta: 14 fps ≥ 0.9×15, todos los streams vivos, GPU/RAM con headroom.
    assert b._verdict(14.0, 4, 4, 80.0, 70.0, 15.0, args) is True
    # No aguanta: un stream no arrancó.
    assert b._verdict(14.0, 3, 4, 80.0, 70.0, 15.0, args) is False
    # No aguanta: compute-bound (fps por debajo del margen).
    assert b._verdict(10.0, 4, 4, 80.0, 70.0, 15.0, args) is False
    # No aguanta: GPU saturada.
    assert b._verdict(14.0, 4, 4, 96.0, 70.0, 15.0, args) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("todos los self-checks pasaron")
