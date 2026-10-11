# ui.py — All OpenCV rendering. Zero application logic lives here.
# NOTE: No Unicode emoji — OpenCV on Windows does not render them.

import cv2
import numpy as np
from config import (
    COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE,
    COLOR_BLACK, COLOR_DARK, COLOR_BLUE,
    LEFT_EYE_TOP_BOTTOM, RIGHT_EYE_TOP_BOTTOM,
    LEFT_EYE_LEFT_RIGHT, RIGHT_EYE_LEFT_RIGHT,
    MOUTH_TOP_BOTTOM, MOUTH_LEFT_RIGHT,
    CHALLENGE_TIMEOUT,
)
from landmark_utils import get_landmark_coords

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Internal helpers ──────────────────────────────────────────────────────

def _text(frame, text: str, pos: tuple, scale: float = 0.6,
          color=COLOR_WHITE, thickness: int = 1) -> None:
    """Single-call wrapper for cv2.putText with LINE_AA."""
    cv2.putText(frame, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def _filled_rect(frame, x1: int, y1: int, x2: int, y2: int,
                 color, alpha: float = 1.0) -> None:
    """Draw a filled rectangle, optionally blended with the frame."""
    if alpha >= 1.0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _text_size(text: str, scale: float, thickness: int = 1) -> tuple:
    """Return (width, height) of rendered text."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return tw, th


# ── Public drawing functions ──────────────────────────────────────────────

def draw_status_bar(frame, text: str, color) -> None:
    """Top banner — current application state label."""
    h, w = frame.shape[:2]
    _filled_rect(frame, 0, 0, w, 42, color)
    _text(frame, text, (12, 28), scale=0.65, color=COLOR_WHITE, thickness=2)


def draw_calibration_bar(frame, progress: float) -> None:
    """
    Bottom progress bar shown during CALIBRATING.
    Fills left-to-right in blue; turns fully bright at 100 %.
    """
    h, w = frame.shape[:2]
    bar_h = 36
    y1, y2 = h - bar_h, h

    _filled_rect(frame, 0, y1, w, y2, COLOR_DARK)

    fill_w = int(w * min(progress, 1.0))
    if fill_w > 0:
        _filled_rect(frame, 0, y1, fill_w, y2, COLOR_BLUE)

    label = f"Calibrating... {int(progress * 100)}%"
    _text(frame, label, (12, h - 10), scale=0.6, color=COLOR_WHITE)
    cv2.rectangle(frame, (0, y1), (w, y2), COLOR_WHITE, 1)


def draw_challenge_box(frame, challenge_text: str, time_remaining: float) -> None:
    """
    Centred instruction card with a colour-coded countdown bar.
    Timer colour: green (>60 %) -> yellow (>30 %) -> red (<=30 %).
    """
    h, w = frame.shape[:2]

    box_w, box_h = 460, 88
    x1 = (w - box_w) // 2
    y1 = h - 158
    x2, y2 = x1 + box_w, y1 + box_h

    _filled_rect(frame, x1, y1, x2, y2, COLOR_DARK, alpha=0.85)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WHITE, 1)

    # Challenge text — centred horizontally
    tw, th = _text_size(challenge_text, 0.75, 2)
    tx = x1 + (box_w - tw) // 2
    _text(frame, challenge_text, (tx, y1 + 40), scale=0.75,
          color=COLOR_WHITE, thickness=2)

    # Countdown bar
    ratio     = max(time_remaining / CHALLENGE_TIMEOUT, 0.0)
    bar_color = (COLOR_GREEN if ratio > 0.6
                 else COLOR_YELLOW if ratio > 0.3
                 else COLOR_RED)

    bx1, by1 = x1 + 10, y2 - 18
    bx2, by2 = x2 - 10, y2 - 6
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
    fill_x = bx1 + int((bx2 - bx1) * ratio)
    if fill_x > bx1:
        cv2.rectangle(frame, (bx1, by1), (fill_x, by2), bar_color, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_WHITE, 1)

    _text(frame, f"{time_remaining:.1f}s", (bx2 + 6, by2),
          scale=0.45, color=COLOR_WHITE)


def draw_landmarks(frame, face_landmarks, w: int, h: int) -> None:
    """Draw coloured dots on eye and mouth landmarks only (not all 468)."""
    eye_indices = (LEFT_EYE_TOP_BOTTOM + RIGHT_EYE_TOP_BOTTOM +
                   LEFT_EYE_LEFT_RIGHT + RIGHT_EYE_LEFT_RIGHT)
    mouth_indices = list(MOUTH_TOP_BOTTOM) + list(MOUTH_LEFT_RIGHT)

    for idx in eye_indices:
        pt = get_landmark_coords(face_landmarks, idx, w, h)
        cv2.circle(frame, pt, 3, COLOR_BLUE, -1)

    for idx in mouth_indices:
        pt = get_landmark_coords(face_landmarks, idx, w, h)
        cv2.circle(frame, pt, 3, COLOR_GREEN, -1)


def draw_classifier_badge(frame, prediction: dict) -> None:
    """
    Top-right badge: REAL / SPOOF + confidence %.
    Green = REAL, Red = SPOOF.
    """
    if prediction is None:
        return

    h, w  = frame.shape[:2]
    label = prediction['label']
    conf  = prediction['confidence']
    color = COLOR_GREEN if label == 'REAL' else COLOR_RED

    badge_text = f"{label}  {conf * 100:.0f}%"
    tw, th     = _text_size(badge_text, 0.65, 2)
    pad = 10
    x2, y1 = w - 10, 50
    x1, y2 = x2 - tw - pad * 2, y1 + th + pad * 2

    _filled_rect(frame, x1, y1, x2, y2, color, alpha=0.88)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WHITE, 1)
    _text(frame, badge_text, (x1 + pad, y2 - pad),
          scale=0.65, color=COLOR_WHITE, thickness=2)


def draw_completed_challenges(frame, completed: list, failed: list) -> None:
    """
    Bottom-left panel: [OK] or [XX] for each resolved challenge.
    Lets the viewer see what has been passed/failed so far.
    """
    h, w = frame.shape[:2]
    y    = h - 168
    x    = 14

    for name in completed:
        _text(frame, f"[OK] {name}", (x, y), scale=0.5, color=COLOR_GREEN)
        y += 22

    for name in failed:
        _text(frame, f"[XX] {name}", (x, y), scale=0.5, color=COLOR_RED)
        y += 22


def draw_no_face_warning(frame) -> None:
    """Centred amber warning when no face is detected."""
    h, w = frame.shape[:2]
    msg  = "No face detected -- look at the camera"
    tw, _ = _text_size(msg, 0.65, 2)
    _text(frame, msg, ((w - tw) // 2, h // 2),
          scale=0.65, color=COLOR_YELLOW, thickness=2)


def draw_result_screen(frame, result: str) -> None:
    """
    Full-frame translucent overlay for the final VERIFIED / REJECTED verdict.
    Green tint + [  OK  ] for VERIFIED.
    Red tint   + [ FAIL ] for REJECTED.
    """
    h, w = frame.shape[:2]

    color  = COLOR_GREEN if result == 'VERIFIED' else COLOR_RED
    icon   = "[  OK  ]" if result == 'VERIFIED' else "[ FAIL ]"
    sub    = "Identity Verified" if result == 'VERIFIED' else "Verification Failed"

    # Semi-transparent colour wash
    _filled_rect(frame, 0, 0, w, h, color, alpha=0.40)

    # Large centred icon
    iw, ih = _text_size(icon, 2.2, 4)
    cv2.putText(frame, icon, ((w - iw) // 2, h // 2 - 16),
                FONT, 2.2, COLOR_WHITE, 4, cv2.LINE_AA)

    # Sub-label
    sw, _ = _text_size(sub, 0.9, 2)
    cv2.putText(frame, sub, ((w - sw) // 2, h // 2 + 52),
                FONT, 0.9, COLOR_WHITE, 2, cv2.LINE_AA)

    # Restart hint
    hint = "Press R to restart"
    hw, _ = _text_size(hint, 0.55, 1)
    _text(frame, hint, ((w - hw) // 2, h - 28), scale=0.55, color=COLOR_WHITE)
