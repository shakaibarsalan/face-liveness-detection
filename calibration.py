# calibration.py — Per-user baseline measurement at session start.

import numpy as np
from config import (
    CALIBRATION_FRAMES, EAR_BLINK_RATIO, MAR_OPEN_RATIO,
    LEFT_EYE_TOP_BOTTOM, RIGHT_EYE_TOP_BOTTOM,
    LEFT_EYE_LEFT_RIGHT, RIGHT_EYE_LEFT_RIGHT,
)
from landmark_utils import (
    eye_aspect_ratio, mouth_aspect_ratio, head_pose_ratios,
)


class Calibrator:
    """
    Collects CALIBRATION_FRAMES worth of landmark measurements while the
    user sits still, then derives personalised detection thresholds.

    Why calibrate per-user?
    -----------------------
    A fixed EAR blink threshold of, say, 0.25 works for most people but
    fails for users with naturally narrow eyes, glasses, or unusual camera
    angles. Measuring each user's neutral state and applying a *ratio* to
    that baseline is far more robust.
    """

    def __init__(self):
        self.ear_samples  : list[float] = []
        self.mar_samples  : list[float] = []
        self.pose_h_samples: list[float] = []
        self.pose_v_samples: list[float] = []
        self.is_done = False
        self.baseline: dict = {}

    # ── public ────────────────────────────────────────────────────────────

    def collect(self, face_landmarks, w: int, h: int) -> None:
        """
        Called once per frame while in CALIBRATING state.
        Accumulates measurements until CALIBRATION_FRAMES are reached,
        then automatically computes the baseline.
        """
        if self.is_done:
            return

        left_ear  = eye_aspect_ratio(face_landmarks,
                                     LEFT_EYE_TOP_BOTTOM[0],  LEFT_EYE_TOP_BOTTOM[1],
                                     LEFT_EYE_LEFT_RIGHT[0],  LEFT_EYE_LEFT_RIGHT[1],
                                     w, h)
        right_ear = eye_aspect_ratio(face_landmarks,
                                     RIGHT_EYE_TOP_BOTTOM[0], RIGHT_EYE_TOP_BOTTOM[1],
                                     RIGHT_EYE_LEFT_RIGHT[0], RIGHT_EYE_LEFT_RIGHT[1],
                                     w, h)
        mar  = mouth_aspect_ratio(face_landmarks, w, h)
        pose = head_pose_ratios(face_landmarks, w, h)

        avg_ear = (left_ear + right_ear) / 2.0
        self.ear_samples.append(avg_ear)
        self.mar_samples.append(mar)
        self.pose_h_samples.append(pose['horizontal'])
        self.pose_v_samples.append(pose['vertical'])

        if len(self.ear_samples) >= CALIBRATION_FRAMES:
            self._compute_baseline()

    def progress(self) -> float:
        """Return calibration completion as a fraction 0.0 → 1.0."""
        return min(len(self.ear_samples) / CALIBRATION_FRAMES, 1.0)

    # ── private ───────────────────────────────────────────────────────────

    def _compute_baseline(self) -> None:
        """
        Average each metric over all collected samples to get the user's
        neutral-face values, then derive action thresholds from those averages.
        """
        base_ear = float(np.mean(self.ear_samples))
        base_mar = float(np.mean(self.mar_samples))
        base_ph  = float(np.mean(self.pose_h_samples))
        base_pv  = float(np.mean(self.pose_v_samples))

        self.baseline = {
            'ear'             : base_ear,
            'mar'             : base_mar,
            'pose_h'          : base_ph,
            'pose_v'          : base_pv,
            # Thresholds derived from the user's own neutral values
            'blink_threshold' : base_ear * EAR_BLINK_RATIO,
            'mouth_threshold' : base_mar * MAR_OPEN_RATIO,
        }

        print("\n── Calibration complete ─────────────────────────────")
        print(f"   Baseline EAR          : {base_ear:.4f}")
        print(f"   Baseline MAR          : {base_mar:.4f}")
        print(f"   Blink threshold (EAR) : {self.baseline['blink_threshold']:.4f}")
        print(f"   Mouth threshold (MAR) : {self.baseline['mouth_threshold']:.4f}")
        print(f"   Head pose (h, v)      : ({base_ph:+.4f}, {base_pv:+.4f})")
        print("─────────────────────────────────────────────────────\n")

        self.is_done = True
