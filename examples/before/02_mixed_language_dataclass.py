# Source: deploy/pipelines/pose_worker.py — PoseResult dataclass
# Anti-pattern illustrated:
#   - Spanish docstring + field comments in a file whose module docstring is English
#   - Inconsistency: the "official" documentation (module-level) sets English as the language,
#     but class and field docs drift into Spanish

@dataclass
class PoseResult:
    """Resultado de la clasificación de pose para un track. Producido por _run_inference()."""
    is_falling: bool  # True si ≥ FALL_SCORE_THRESHOLD reglas se cumplen
    fall_score: int   # número de reglas (0-3) que se cumplieron
    avg_conf: float   # confianza promedio de los 6 keypoints relevantes
