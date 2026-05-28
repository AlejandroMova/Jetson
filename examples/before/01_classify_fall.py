# Source: deploy/pipelines/pose_worker.py — _run_inference + _classify_fall
# Anti-patterns illustrated:
#   - 18 cryptic abbreviation variables (lsh_x, rsh_x, lhi_x, rhi_x, lan_x, ran_x)
#   - Generic inner-function name: valid(c) instead of domain term
#   - Spanish docstring in a module whose top-level docstring is English
#   - Spanish inline comments describing WHAT the code does (not WHY)

def _run_inference(self, crop_bgr: np.ndarray, bbox: dict) -> PoseResult:
    """Ejecuta MoveNet sobre el crop de persona y clasifica si está cayendo.

    Preprocesamiento: resize a 192×192, BGR→RGB, normalizar a [0, 1], NCHW.
    Salida de MoveNet: tensor (1, 1, 17, 3) con 17 keypoints COCO, cada uno (y_norm, x_norm, conf).
    La clasificación se hace con 3 reglas geométricas sobre los keypoints resultantes.
    """
    # ── Preprocesamiento ──────────────────────────────────────────────────
    img = cv2.resize(crop_bgr, (192, 192))                       # escalar al tamaño de entrada del modelo
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # BGR→RGB y normalizar [0,1]
    inp = np.transpose(img_rgb, (2, 0, 1))[np.newaxis]           # HWC → NCHW: (1, 3, 192, 192)

    # ── Inferencia ONNX ───────────────────────────────────────────────────
    inp_name = self._session.get_inputs()[0].name
    out = self._session.run(None, {inp_name: inp})[0]             # salida: (1, 1, 17, 3)

    # ── Extraer keypoints y clasificar ────────────────────────────────────
    keypoints = out[0, 0]  # (17, 3): cada fila = (y_norm, x_norm, confidence)
    score, avg_conf = self._classify_fall(keypoints, bbox)
    return PoseResult(
        is_falling=(score >= FALL_SCORE_THRESHOLD),  # caída si ≥2 de 3 reglas se cumplen
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

    # Gather keypoints with confidence check
    lsh_x, lsh_y, lsh_c = kp(_KP_LEFT_SHOULDER)
    rsh_x, rsh_y, rsh_c = kp(_KP_RIGHT_SHOULDER)
    lhi_x, lhi_y, lhi_c = kp(_KP_LEFT_HIP)
    rhi_x, rhi_y, rhi_c = kp(_KP_RIGHT_HIP)
    lan_x, lan_y, lan_c = kp(_KP_LEFT_ANKLE)
    ran_x, ran_y, ran_c = kp(_KP_RIGHT_ANKLE)

    confs = [lsh_c, rsh_c, lhi_c, rhi_c, lan_c, ran_c]
    avg_conf = float(np.mean(confs))

    # Filter out low-confidence keypoints by zeroing their contribution
    def valid(c): return c >= FALL_MIN_KP_CONF

    score = 0

    # Rule 1: torso angle from vertical
    # Mid-shoulder → mid-hip vector; angle from vertical axis
    if valid(lsh_c) and valid(rsh_c) and valid(lhi_c) and valid(rhi_c):
        sh_x = (lsh_x + rsh_x) / 2
        sh_y = (lsh_y + rsh_y) / 2
        hi_x = (lhi_x + rhi_x) / 2
        hi_y = (lhi_y + rhi_y) / 2
        dx = hi_x - sh_x
        dy = hi_y - sh_y
        if dy != 0 or dx != 0:
            angle_from_vertical = abs(math.degrees(math.atan2(abs(dx), abs(dy))))
            if angle_from_vertical > (90 - FALL_TORSO_ANGLE_MAX):
                score += 1

    # Rule 2: bounding box aspect ratio (width > height = person lying down)
    w = bbox.get("width", 1)
    h = bbox.get("height", 1)
    if h > 0 and w / h > 1.0:
        score += 1

    # Rule 3: hip Y close to ankle Y (hips near ground level)
    if valid(lhi_c) and valid(rhi_c) and valid(lan_c) and valid(ran_c):
        hi_y = (lhi_y + rhi_y) / 2
        an_y = (lan_y + ran_y) / 2
        if an_y > 0 and hi_y >= an_y * 0.80:
            score += 1

    return score, avg_conf
