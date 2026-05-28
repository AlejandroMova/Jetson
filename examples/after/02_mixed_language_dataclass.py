# Source: deploy/pipelines/pose_worker.py — PoseResult dataclass
# Changes applied:
#   - Docstring and field comments translated to English
#   - Consistent with the module-level docstring language (English)
#
# Rule: pick the language of the module docstring as the source of truth.
# If the module docstring is English, ALL docstrings and comments in that file must be English.
# Mixed-language files require the reader to mentally switch context on every other line.

@dataclass
class PoseResult:
    """Pose classification result for a single track. Produced by _run_inference()."""
    is_falling: bool  # True when ≥ FALL_SCORE_THRESHOLD rules fired
    fall_score: int   # number of rules that fired (0–3)
    avg_conf: float   # mean confidence across the 6 relevant keypoints
