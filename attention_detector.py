# ============================================================
# AI Attention Detector (v2 — Improved Accuracy)
# Uses MediaPipe FaceMesh (468 + iris landmarks) with:
#   1. 3D head pose estimation via solvePnP (pitch/yaw/roll)
#   2. Eye Aspect Ratio (EAR) with relaxed threshold
#   3. Mouth Aspect Ratio (MAR) for yawn detection
#   4. Iris-based gaze direction estimation
#   5. Weighted scoring — soft decision instead of hard binary
# ============================================================

import cv2
import mediapipe as mp
import numpy as np

# ─── MediaPipe FaceMesh init ─────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,         # processes one cropped face at a time
    refine_landmarks=True,
    min_detection_confidence=0.3
)

# ─── Haar cascade for initial face detection ─────────────────
# Used as a first-pass detector; each crop is then passed to FaceMesh
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ─── MediaPipe face detection — validates Haar boxes ─────────
# model_selection=0: short-range model (best for typical classroom/webcam range)
# Used to distinguish real-but-non-frontal faces from Haar false positives
_mp_face_detection = mp.solutions.face_detection
_face_detector = _mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.3
)

# ─── Landmark indices ────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks for MAR (top/bottom lip vertical pairs + corners)
UPPER_LIP = [13]    # top inner lip
LOWER_LIP = [14]    # bottom inner lip
LEFT_MOUTH  = [78]  # left corner
RIGHT_MOUTH = [308] # right corner
# Additional vertical pair for better MAR
UPPER_LIP2 = [82]
LOWER_LIP2 = [87]

# Iris center landmarks (from refine_landmarks)
LEFT_IRIS_CENTER  = 473   # center of left iris
RIGHT_IRIS_CENTER = 468   # center of right iris

# 6 key 3D model points for solvePnP head pose estimation
# These correspond to a generic 3D face model
POSE_LANDMARKS_3D = {
    1:   (0.0, 0.0, 0.0),          # nose tip
    33:  (-225.0, 170.0, -135.0),   # left eye outer corner
    263: (225.0, 170.0, -135.0),    # right eye outer corner
    61:  (-150.0, -150.0, -125.0),  # left mouth corner
    291: (150.0, -150.0, -125.0),   # right mouth corner
    199: (0.0, -330.0, -65.0),      # chin
}

# ─── Thresholds & weights ────────────────────────────────────
EAR_THRESHOLD      = 0.18   # lowered from 0.20 to reduce false negatives
MAR_THRESHOLD      = 0.75   # mouth open ratio indicating yawning
YAW_THRESHOLD      = 30.0   # degrees — looking left/right (hard rule)
PITCH_DOWN_THRESH  = 22.0   # degrees — looking down (hard rule)
PITCH_UP_THRESH    = 20.0   # degrees — looking up
ROLL_THRESHOLD     = 35.0   # degrees — head tilted sideways
GAZE_H_THRESHOLD   = 0.30   # horizontal iris offset — looking left/right
GAZE_V_THRESHOLD   = 0.28   # vertical iris offset — relaxed from 0.20 to reduce false positives

# Weighted scoring: each signal contributes a portion of the total
WEIGHT_EYES    = 0.15   # are eyes open?
WEIGHT_YAW     = 0.20   # facing forward horizontally?
WEIGHT_PITCH   = 0.40   # not looking down/up? (most reliable signal)
WEIGHT_GAZE    = 0.15   # iris pointing forward?
WEIGHT_YAWN    = 0.10   # not yawning?
ATTENTION_CUTOFF = 0.35  # score >= this = attentive


# ─── Helper functions ────────────────────────────────────────

def _lm_to_px(lm, idx, w, h):
    """Convert a normalized landmark to pixel coordinates."""
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float64)


def calculate_EAR(landmarks, eye_indices, w, h):
    """Eye Aspect Ratio — how open the eye is."""
    pts = [_lm_to_px(landmarks, i, w, h) for i in eye_indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0


def calculate_MAR(landmarks, w, h):
    """Mouth Aspect Ratio — high value means mouth wide open (yawning)."""
    top1    = _lm_to_px(landmarks, UPPER_LIP[0], w, h)
    bottom1 = _lm_to_px(landmarks, LOWER_LIP[0], w, h)
    top2    = _lm_to_px(landmarks, UPPER_LIP2[0], w, h)
    bottom2 = _lm_to_px(landmarks, LOWER_LIP2[0], w, h)
    left    = _lm_to_px(landmarks, LEFT_MOUTH[0], w, h)
    right   = _lm_to_px(landmarks, RIGHT_MOUTH[0], w, h)

    A = np.linalg.norm(top1 - bottom1)
    B = np.linalg.norm(top2 - bottom2)
    C = np.linalg.norm(left - right)
    return (A + B) / (2.0 * C) if C > 0 else 0


def estimate_head_pose(landmarks, w, h):
    """
    3D head pose estimation using solvePnP.
    Returns (pitch, yaw, roll) in degrees.
    Pitch: +down / -up
    Yaw:   +right / -left
    Roll:  head tilt
    """
    # 2D image points from detected landmarks
    image_points = np.array([
        [landmarks[idx].x * w, landmarks[idx].y * h]
        for idx in POSE_LANDMARKS_3D
    ], dtype=np.float64)

    # Corresponding 3D model points
    model_points = np.array(
        list(POSE_LANDMARKS_3D.values()), dtype=np.float64
    )

    # Approximate camera matrix (no real calibration needed)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]

    # RQDecomp3x3 can return angles outside [-90, 90] due to gimbal lock.
    # Unwrap: if |pitch| > 90 the decomposition flipped by 180°.
    # e.g. pitch=-151 (wrapped) → -151+180 = 29° (looking down 29°)
    if pitch < -90:
        pitch = pitch + 180.0
    elif pitch > 90:
        pitch = pitch - 180.0
    if yaw < -90:
        yaw = yaw + 180.0
    elif yaw > 90:
        yaw = yaw - 180.0

    return pitch, yaw, roll


def estimate_gaze(landmarks, w, h):
    """
    Iris-based gaze estimation — horizontal and vertical components.
    Returns (h_offset, v_offset): how far iris is from eye center
    horizontally and vertically, each normalized by eye width.
    """
    def _iris_offsets(iris_idx, eye_indices):
        iris  = _lm_to_px(landmarks, iris_idx, w, h)
        inner = _lm_to_px(landmarks, eye_indices[0], w, h)
        outer = _lm_to_px(landmarks, eye_indices[3], w, h)
        top   = _lm_to_px(landmarks, eye_indices[1], w, h)
        bot   = _lm_to_px(landmarks, eye_indices[5], w, h)
        eye_center = (inner + outer) / 2
        eye_width  = np.linalg.norm(inner - outer)
        eye_height = np.linalg.norm(top - bot)
        if eye_width < 1:
            return 0.5, 0.5
        h_off = abs(iris[0] - eye_center[0]) / eye_width
        v_off = abs(iris[1] - eye_center[1]) / (eye_height if eye_height > 1 else eye_width * 0.5)
        return h_off, v_off

    lh, lv = _iris_offsets(LEFT_IRIS_CENTER,  LEFT_EYE)
    rh, rv = _iris_offsets(RIGHT_IRIS_CENTER, RIGHT_EYE)
    return (lh + rh) / 2, (lv + rv) / 2


# ─── Main analysis ───────────────────────────────────────────

def analyze_image(image_path):
    """
    Analyzes a classroom image for student attention using a robust two-stage pipeline:

    Stage 1 — Face detection:
      a) Run Haar cascade on full image for bounding boxes (works on group photos)
      b) Run MediaPipe FaceDetection on full image to CONFIRM which Haar boxes are real
         and to catch faces Haar missed (e.g., partially-turned faces looking at phones)

    Stage 2 — Per-face analysis:
      • FaceMesh succeeds → landmark-based scoring (EAR, head pose, gaze, yawn)
      • FaceMesh fails + face is FaceDetection-confirmed → non-frontal → DISTRACTED
      • FaceMesh fails + face is Haar-only → IGNORE (likely false positive)

    Angles are computed from Haar-based crops for consistency with tuned thresholds.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not load image", "attentive": 0,
                "distracted": 0, "total_faces": 0, "score": 0}

    h, w = img.shape[:2]

    # ── Upscale small images ──
    MIN_DIM = 960
    if max(h, w) < MIN_DIM:
        scale = MIN_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Stage 1a: Haar cascade — bounding boxes ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar_raw = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
    )
    haar_boxes = [(int(x), int(y), int(bw), int(bh))
                  for x, y, bw, bh in haar_raw] if len(haar_raw) > 0 else []

    # ── Stage 1b: FaceDetection on full image — confirmed face positions ──
    fd_result = _face_detector.process(rgb_img)
    fd_boxes = []
    if fd_result.detections:
        for det in fd_result.detections:
            bb = det.location_data.relative_bounding_box
            bx = max(0, int(bb.xmin * w))
            by = max(0, int(bb.ymin * h))
            bw2 = int(bb.width * w)
            bh2 = int(bb.height * h)
            if bw2 > 20 and bh2 > 20:
                fd_boxes.append((bx, by, bw2, bh2))

    # ── Build final face list ──
    # Start with Haar boxes, mark each as confirmed or not by FaceDetection overlap.
    # Also add any FaceDetection boxes that don't overlap with any Haar box.
    def _iou(a, b):
        ax1, ay1, aw, ah = a; ax2, ay2 = ax1 + aw, ay1 + ah
        bx1, by1, bw, bh = b; bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    IOU_THRESH = 0.20   # generous — FD and Haar boxes often differ in tightness

    # If FaceDetection finds zero faces but Haar found several,
    # FD failed on this image → trust all Haar boxes as confirmed.
    fd_failed = len(fd_boxes) == 0 and len(haar_boxes) >= 2

    face_list = []   # list of (box, is_fd_confirmed)
    for hb in haar_boxes:
        confirmed = fd_failed or any(_iou(hb, fb) >= IOU_THRESH for fb in fd_boxes)
        face_list.append((hb, confirmed))

    # Add FD boxes that have no Haar overlap (faces Haar missed)
    for fb in fd_boxes:
        has_haar_overlap = any(_iou(fb, hb) >= IOU_THRESH for hb, _ in face_list)
        if not has_haar_overlap:
            face_list.append((fb, True))   # FD-only, confirmed

    if not face_list:
        return {"attentive": 0, "distracted": 0,
                "total_faces": 0, "score": 0, "details": []}

    # ── Stage 2: FaceMesh per crop ──
    FACE_CROP_PAD = 0.30
    FACE_MIN_SIZE = 120

    attentive_count = 0
    distracted_count = 0
    details = []

    for face_idx, ((fx, fy, fw, fh), is_confirmed) in enumerate(face_list):
        pad_x = int(fw * FACE_CROP_PAD)
        pad_y = int(fh * FACE_CROP_PAD)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]

        if cw < FACE_MIN_SIZE or ch < FACE_MIN_SIZE:
            up = max(FACE_MIN_SIZE / cw, FACE_MIN_SIZE / ch)
            crop = cv2.resize(crop, (int(cw * up), int(ch * up)),
                              interpolation=cv2.INTER_CUBIC)
            ch, cw = crop.shape[:2]

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_crop)

        if not result.multi_face_landmarks:
            if is_confirmed:
                # FaceDetection confirmed real face, FaceMesh can't landmark it
                # → person is not facing camera → DISTRACTED
                details.append({
                    "face": face_idx + 1,
                    "ear": None, "mar": None,
                    "yaw": None, "pitch": None,
                    "gaze_h": None, "gaze_v": None,
                    "weighted_score": 0.0,
                    "eyes_open": None,
                    "facing_forward": False,
                    "status": "distracted"
                })
                distracted_count += 1
            # else: unconfirmed Haar box → false positive → skip
            continue

        lm = result.multi_face_landmarks[0].landmark

        # ── Signal 1: Eye Aspect Ratio ──
        left_ear  = calculate_EAR(lm, LEFT_EYE, cw, ch)
        right_ear = calculate_EAR(lm, RIGHT_EYE, cw, ch)
        avg_ear   = (left_ear + right_ear) / 2
        eyes_score = 1.0 if avg_ear >= EAR_THRESHOLD else (avg_ear / EAR_THRESHOLD)

        # ── Signal 2 & 3: Head pose (yaw + pitch) ──
        pitch, yaw, roll = estimate_head_pose(lm, cw, ch)
        abs_yaw        = abs(yaw)
        abs_pitch_down = max(0.0, pitch)
        abs_pitch_up   = max(0.0, -pitch)
        abs_roll       = abs(roll)

        # ── Size-adaptive pitch threshold ──
        # Small faces (< 100 px) in group photos produce noisier
        # solvePnP angles — use a more lenient pitch threshold.
        effective_pitch_down = 35.0 if fw < 100 else PITCH_DOWN_THRESH

        # ── Hard override: extreme pose → instantly distracted ──
        if abs_yaw >= YAW_THRESHOLD or abs_pitch_down >= effective_pitch_down:
            details.append({
                "face": face_idx + 1,
                "ear": round(float(avg_ear), 3),
                "mar": round(float(calculate_MAR(lm, cw, ch)), 3),
                "yaw": round(float(yaw), 1),
                "pitch": round(float(pitch), 1),
                "gaze_h": 0.0, "gaze_v": 0.0,
                "weighted_score": 0.0,
                "eyes_open": bool(avg_ear >= EAR_THRESHOLD),
                "facing_forward": False,
                "status": "distracted"
            })
            distracted_count += 1
            continue

        yaw_score   = max(0.0, 1.0 - abs_yaw / YAW_THRESHOLD)
        pitch_penalty = max(abs_pitch_down / effective_pitch_down,
                            abs_pitch_up / PITCH_UP_THRESH)
        pitch_score = max(0.0, 1.0 - pitch_penalty)
        if abs_roll > ROLL_THRESHOLD:
            yaw_score *= 0.5

        # ── Signal 4: Gaze direction ──
        gaze_h, gaze_v = estimate_gaze(lm, cw, ch)
        gaze_score = max(0.0, 1.0 - max(gaze_h / GAZE_H_THRESHOLD,
                                         gaze_v / GAZE_V_THRESHOLD))

        # ── Signal 5: Yawn detection ──
        mar = calculate_MAR(lm, cw, ch)
        yawn_score = 0.0 if mar > MAR_THRESHOLD else 1.0

        # ── Weighted final score ──
        attention_score = (
            WEIGHT_EYES  * eyes_score  +
            WEIGHT_YAW   * yaw_score   +
            WEIGHT_PITCH * pitch_score +
            WEIGHT_GAZE  * gaze_score  +
            WEIGHT_YAWN  * yawn_score
        )

        is_attentive = attention_score >= ATTENTION_CUTOFF

        if is_attentive:
            attentive_count += 1
        else:
            distracted_count += 1

        details.append({
            "face": face_idx + 1,
            "ear": round(float(avg_ear), 3),
            "mar": round(float(mar), 3),
            "yaw": round(float(yaw), 1),
            "pitch": round(float(pitch), 1),
            "gaze_h": round(float(gaze_h), 3),
            "gaze_v": round(float(gaze_v), 3),
            "weighted_score": round(float(attention_score), 3),
            "eyes_open": bool(avg_ear >= EAR_THRESHOLD),
            "facing_forward": bool(abs_yaw < YAW_THRESHOLD and pitch_penalty < 1.0),
            "status": "attentive" if is_attentive else "distracted"
        })

    total = attentive_count + distracted_count
    score = round((attentive_count / total) * 100, 1) if total > 0 else 0

    print(f"Faces: {total} | Attentive: {attentive_count} | Score: {score}%")
    return {
        "attentive": attentive_count,
        "distracted": distracted_count,
        "total_faces": total,
        "score": score,
        "details": details
    }
