# ============================================================
# AI Attention Detector
# Uses MediaPipe FaceMesh to detect 468 facial landmarks
# and determine if each person is attentive or distracted
# ============================================================

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh
# This loads a pre-trained AI model — no training needed by us!
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,    # True = process single images
    max_num_faces=10,           # Detect up to 10 faces per image
    min_detection_confidence=0.5  # 50% confidence threshold
)

# ─── EAR (Eye Aspect Ratio) ──────────────────────────────────
# EAR measures how "open" the eyes are.
# A high EAR (>0.2) = eyes open = potentially attentive
# A low EAR (<0.2) = eyes closed = sleeping/blinking
#
# Formula: EAR = (vertical distance) / (horizontal distance)
# We use specific landmark indices from MediaPipe's 468-point map

# Left eye landmark indices (from MediaPipe documentation)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_EAR(landmarks, eye_indices, w, h):
    """
    Calculate Eye Aspect Ratio.
    landmarks: list of (x, y) normalized coordinates
    eye_indices: which of the 468 points to use
    w, h: image width and height (to convert to pixels)
    """
    # Get the 6 key eye points as pixel coordinates
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
           for i in eye_indices]

    # Vertical distances (top to bottom of eye)
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))

    # Horizontal distance (corner to corner)
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    # EAR formula
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear

def check_head_pose(landmarks, w, h):
    """
    Simple head pose check using nose tip and eye positions.
    Returns True if the person appears to be facing forward.
    This is a simplified version — good enough for university project!
    """
    # Key landmarks: nose tip = 1, left eye outer = 33, right eye outer = 263
    nose  = landmarks[1]
    l_eye = landmarks[33]
    r_eye = landmarks[263]

    # Convert to pixel coords
    nose_x  = nose.x * w
    l_eye_x = l_eye.x * w
    r_eye_x = r_eye.x * w

    # Face center is midpoint between eyes
    eye_center_x = (l_eye_x + r_eye_x) / 2
    eye_width     = abs(r_eye_x - l_eye_x)

    # If nose is within 35% of eye width from center → facing forward
    offset_ratio = abs(nose_x - eye_center_x) / eye_width if eye_width > 0 else 1
    is_forward = offset_ratio < 0.35

    # Also check: is nose tip too low? (head nodding down = distracted)
    nose_y  = nose.y * h
    l_eye_y = l_eye.y * h
    r_eye_y = r_eye.y * h
    eye_center_y = (l_eye_y + r_eye_y) / 2
    nod_ratio = (nose_y - eye_center_y) / (h * 0.3)
    not_nodding = nod_ratio < 1.5

    return is_forward and not_nodding

def analyze_image(image_path):
    """
    Main function — analyzes a classroom image for attention.

    Returns a dict:
    {
      "attentive": int,    # number of attentive students
      "distracted": int,   # number of distracted students
      "total_faces": int,  # total faces detected
      "score": float,      # attention percentage (0-100)
      "details": [...]     # per-face analysis
    }
    """
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not load image", "attentive": 0,
                "distracted": 0, "total_faces": 0, "score": 0}

    h, w = img.shape[:2]

    # Convert BGR (OpenCV format) to RGB (MediaPipe format)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run MediaPipe face detection
    results = face_mesh.process(rgb_img)

    # If no faces found in the image
    if not results.multi_face_landmarks:
        return {"attentive": 0, "distracted": 0,
                "total_faces": 0, "score": 0, "details": []}

    attentive_count = 0
    distracted_count = 0
    details = []

    # Analyze each detected face
    for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
        lm = face_landmarks.landmark  # shorthand for landmarks

        # Calculate Eye Aspect Ratio for both eyes
        left_ear  = calculate_EAR(lm, LEFT_EYE, w, h)
        right_ear = calculate_EAR(lm, RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2
        eyes_open = avg_ear > 0.20  # threshold for open eyes

        # Check head pose
        facing_forward = check_head_pose(lm, w, h)

        # Final decision: attentive only if BOTH conditions are true
        is_attentive = eyes_open and facing_forward

        if is_attentive:
            attentive_count += 1
        else:
            distracted_count += 1

        # details.append({
        #     "face": face_idx + 1,
        #     "ear": round(avg_ear, 3),
        #     "eyes_open": eyes_open,
        #     "facing_forward": facing_forward,
        #     "status": "attentive" if is_attentive else "distracted"
        # })
        details.append({
        "face": face_idx + 1,
        "ear": round(float(avg_ear), 3),   # also convert float to be safe
        "eyes_open": bool(eyes_open),       # ← fix
        "facing_forward": bool(facing_forward),  # ← fix
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
