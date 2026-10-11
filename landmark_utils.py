# landmark_utils.py — Pure math on MediaPipe face landmarks. No side-effects.

import math
import numpy as np
from config import (
    LEFT_EYE_TOP_BOTTOM, RIGHT_EYE_TOP_BOTTOM,
    LEFT_EYE_LEFT_RIGHT, RIGHT_EYE_LEFT_RIGHT,
    MOUTH_TOP_BOTTOM, MOUTH_LEFT_RIGHT,
    NOSE_TIP, CHIN, LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD,
)


def get_landmark_coords(face_landmarks, index: int, frame_w: int, frame_h: int) -> tuple:
    """Return (x, y) pixel coordinates for a single landmark index."""
    lm = face_landmarks.landmark[index]
    return int(lm.x * frame_w), int(lm.y * frame_h)


def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Standard 2-D Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(face_landmarks, eye_top_idx: int, eye_bot_idx: int,
                     eye_left_idx: int, eye_right_idx: int,
                     w: int, h: int) -> float:
    """
    Eye Aspect Ratio (EAR) — Soukupová & Čech (2016).

    EAR = vertical_distance / horizontal_distance

    A high EAR means the eye is open; it drops sharply when the eye closes
    (blink). Using a single vertical pair keeps the computation fast while
    remaining robust enough for real-time use.
    """
    top   = get_landmark_coords(face_landmarks, eye_top_idx,  w, h)
    bot   = get_landmark_coords(face_landmarks, eye_bot_idx,  w, h)
    left  = get_landmark_coords(face_landmarks, eye_left_idx, w, h)
    right = get_landmark_coords(face_landmarks, eye_right_idx, w, h)

    vertical   = euclidean_distance(top, bot)
    horizontal = euclidean_distance(left, right)

    # Guard against degenerate geometry (face very close to edge)
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def mouth_aspect_ratio(face_landmarks, w: int, h: int) -> float:
    """
    Mouth Aspect Ratio (MAR).

    MAR = vertical_lip_distance / horizontal_mouth_width

    Rises clearly when the mouth opens wide, stays low when closed.
    """
    top   = get_landmark_coords(face_landmarks, MOUTH_TOP_BOTTOM[0], w, h)
    bot   = get_landmark_coords(face_landmarks, MOUTH_TOP_BOTTOM[1], w, h)
    left  = get_landmark_coords(face_landmarks, MOUTH_LEFT_RIGHT[0], w, h)
    right = get_landmark_coords(face_landmarks, MOUTH_LEFT_RIGHT[1], w, h)

    vertical   = euclidean_distance(top, bot)
    horizontal = euclidean_distance(left, right)

    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def head_pose_ratios(face_landmarks, w: int, h: int) -> dict:
    """
    Estimate head pose from landmark geometry — no PnP solver needed.

    Strategy
    --------
    * Face centre  = midpoint of left-cheek / right-cheek (horizontal axis)
                   = midpoint of forehead  / chin          (vertical axis)
    * Nose tip displacement from that centre, normalised by face dimensions,
      gives a scale-independent ratio.

    Returns
    -------
    {
        'horizontal': float   # negative → looking left, positive → right
        'vertical'  : float   # negative → looking up,   positive → down
    }
    """
    lc = get_landmark_coords(face_landmarks, LEFT_CHEEK,  w, h)
    rc = get_landmark_coords(face_landmarks, RIGHT_CHEEK, w, h)
    fh = get_landmark_coords(face_landmarks, FOREHEAD,    w, h)
    ch = get_landmark_coords(face_landmarks, CHIN,        w, h)
    ns = get_landmark_coords(face_landmarks, NOSE_TIP,    w, h)

    face_cx = (lc[0] + rc[0]) / 2.0
    face_cy = (fh[1] + ch[1]) / 2.0

    face_width  = max(rc[0] - lc[0], 1)
    face_height = max(ch[1] - fh[1], 1)

    h_ratio = (ns[0] - face_cx) / face_width   # negative = turned left
    v_ratio = (ns[1] - face_cy) / face_height  # negative = looking up

    return {'horizontal': h_ratio, 'vertical': v_ratio}


def extract_feature_vector(face_landmarks, w: int, h: int) -> list:
    """
    Build a 10-element feature vector for the SVM spoof classifier.

    Features
    --------
    0  left_EAR
    1  right_EAR
    2  avg_EAR
    3  MAR
    4  head_horizontal ratio
    5  head_vertical ratio
    6  left_eye_open  (binary 0/1 — coarse, fixed threshold)
    7  right_eye_open (binary 0/1)
    8  mouth_open     (binary 0/1)
    9  face_width_ratio (face width / frame width — normalises for distance)
    """
    left_ear  = eye_aspect_ratio(face_landmarks,
                                 LEFT_EYE_TOP_BOTTOM[0],  LEFT_EYE_TOP_BOTTOM[1],
                                 LEFT_EYE_LEFT_RIGHT[0],  LEFT_EYE_LEFT_RIGHT[1],
                                 w, h)
    right_ear = eye_aspect_ratio(face_landmarks,
                                 RIGHT_EYE_TOP_BOTTOM[0], RIGHT_EYE_TOP_BOTTOM[1],
                                 RIGHT_EYE_LEFT_RIGHT[0], RIGHT_EYE_LEFT_RIGHT[1],
                                 w, h)
    avg_ear = (left_ear + right_ear) / 2.0
    mar     = mouth_aspect_ratio(face_landmarks, w, h)
    pose    = head_pose_ratios(face_landmarks, w, h)

    lc = get_landmark_coords(face_landmarks, LEFT_CHEEK,  w, h)
    rc = get_landmark_coords(face_landmarks, RIGHT_CHEEK, w, h)
    face_width_ratio = (rc[0] - lc[0]) / max(w, 1)

    # Binary flags use conservative fixed thresholds — only for feature encoding
    left_eye_open  = 1 if left_ear  > 0.20 else 0
    right_eye_open = 1 if right_ear > 0.20 else 0
    mouth_open     = 1 if mar       > 0.40 else 0

    return [
        left_ear, right_ear, avg_ear, mar,
        pose['horizontal'], pose['vertical'],
        float(left_eye_open), float(right_eye_open),
        float(mouth_open), face_width_ratio,
    ]
