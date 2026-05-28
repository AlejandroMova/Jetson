# Source: deploy/pipelines/pose_worker.py — _run_inference + _classify_fall
# Changes applied:
#   - 18 variables renamed to full domain names (lsh_x → left_shoulder_x, etc.)
#   - Inner-function renamed: valid(c) → has_confidence(c)
#   - Midpoint variables renamed: sh_x/sh_y → shoulder_mid_x/shoulder_mid_y, etc.
#   - Docstring translated to English (consistent with module-level docstring)
#   - Removed comments that described WHAT the code does (already obvious from API names)
#   - Kept tensor shape annotations — these save meaningful time in ML code and are NOT obvious
#
# NOTE (new anti-pattern, not in skill): "Tensor shape annotations" (# (1, 3, 192, 192))
# are a GOOD pattern in ML/computer-vision code. Unlike "commenting the obvious", shape
# comments encode information that cannot be derived by reading the function call.
# They should be kept — but only on the line where the shape is actually established,
# not repeated on every subsequent line that uses the same variable.

def _run_inference(self, crop_bgr: np.ndarray, bbox: dict) -> PoseResult:
    """Run MoveNet on a person crop and classify whether the person is falling.

    Preprocessing: resize to 192×192, BGR→RGB, normalise to [0, 1], NCHW.
    MoveNet output: tensor (1, 1, 17, 3) — 17 COCO keypoints, each (y_norm, x_norm, conf).
    Classification uses 3 geometric rules on the resulting keypoints.
    """
    img = cv2.resize(crop_bgr, (192, 192))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.transpose(img_rgb, (2, 0, 1))[np.newaxis]  # HWC → NCHW: (1, 3, 192, 192)

    inp_name = self._session.get_inputs()[0].name
    out = self._session.run(None, {inp_name: inp})[0]  # (1, 1, 17, 3)

    keypoints = out[0, 0]  # (17, 3): each row = (y_norm, x_norm, confidence)
    score, avg_conf = self._classify_fall(keypoints, bbox)
    return PoseResult(
        is_falling=(score >= FALL_SCORE_THRESHOLD),  # fall if ≥2 of 3 rules fire
        fall_score=score,
        avg_conf=avg_conf,
    )

@staticmethod
def _classify_fall(kps: np.ndarray, bbox: dict) -> tuple:
    """
    3-rule geometric fall classifier.
    Returns (fall_score 0-3, avg_keypoint_confidence).
    """
    def kp(idx):
        y, x, c = float(kps[idx, 0]), float(kps[idx, 1]), float(kps[idx, 2])
        return x, y, c

    left_shoulder_x,  left_shoulder_y,  left_shoulder_conf  = kp(_KP_LEFT_SHOULDER)
    right_shoulder_x, right_shoulder_y, right_shoulder_conf = kp(_KP_RIGHT_SHOULDER)
    left_hip_x,       left_hip_y,       left_hip_conf       = kp(_KP_LEFT_HIP)
    right_hip_x,      right_hip_y,      right_hip_conf      = kp(_KP_RIGHT_HIP)
    left_ankle_x,     left_ankle_y,     left_ankle_conf     = kp(_KP_LEFT_ANKLE)
    right_ankle_x,    right_ankle_y,    right_ankle_conf    = kp(_KP_RIGHT_ANKLE)

    confs = [left_shoulder_conf, right_shoulder_conf,
             left_hip_conf,      right_hip_conf,
             left_ankle_conf,    right_ankle_conf]
    avg_conf = float(np.mean(confs))

    def has_confidence(c): return c >= FALL_MIN_KP_CONF

    score = 0

    # Rule 1: torso angle from vertical — mid-shoulder to mid-hip vector
    if has_confidence(left_shoulder_conf) and has_confidence(right_shoulder_conf) \
            and has_confidence(left_hip_conf) and has_confidence(right_hip_conf):
        shoulder_mid_x = (left_shoulder_x + right_shoulder_x) / 2
        shoulder_mid_y = (left_shoulder_y + right_shoulder_y) / 2
        hip_mid_x = (left_hip_x + right_hip_x) / 2
        hip_mid_y = (left_hip_y + right_hip_y) / 2
        dx = hip_mid_x - shoulder_mid_x
        dy = hip_mid_y - shoulder_mid_y
        if dy != 0 or dx != 0:
            angle_from_vertical = abs(math.degrees(math.atan2(abs(dx), abs(dy))))
            if angle_from_vertical > (90 - FALL_TORSO_ANGLE_MAX):
                score += 1

    # Rule 2: bounding box aspect ratio (width > height means person is lying down)
    w = bbox.get("width", 1)
    h = bbox.get("height", 1)
    if h > 0 and w / h > 1.0:
        score += 1

    # Rule 3: hip Y close to ankle Y (hips near ground level)
    if has_confidence(left_hip_conf) and has_confidence(right_hip_conf) \
            and has_confidence(left_ankle_conf) and has_confidence(right_ankle_conf):
        hip_mid_y   = (left_hip_y   + right_hip_y)   / 2
        ankle_mid_y = (left_ankle_y + right_ankle_y) / 2
        if ankle_mid_y > 0 and hip_mid_y >= ankle_mid_y * 0.80:
            score += 1

    return score, avg_conf
