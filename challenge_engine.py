# challenge_engine.py — Random challenge generation and per-frame verification.

import random
import time
from collections import deque
from config import (
    CHALLENGE_COUNT, CHALLENGE_TIMEOUT,
    HEAD_TURN_THRESHOLD, HEAD_NOD_THRESHOLD,
    LEFT_EYE_TOP_BOTTOM, RIGHT_EYE_TOP_BOTTOM,
    LEFT_EYE_LEFT_RIGHT, RIGHT_EYE_LEFT_RIGHT,
)
from landmark_utils import (
    eye_aspect_ratio, mouth_aspect_ratio, head_pose_ratios,
)

# ── Challenge catalogue ────────────────────────────────────────────────────

CHALLENGES = ['BLINK', 'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN', 'OPEN_MOUTH']

CHALLENGE_MESSAGES = {
    'BLINK'      : 'Please BLINK your eyes',
    'TURN_LEFT'  : 'Turn your head LEFT',
    'TURN_RIGHT' : 'Turn your head RIGHT',
    'LOOK_UP'    : 'Look UP',
    'LOOK_DOWN'  : 'Look DOWN',
    'OPEN_MOUTH' : 'Open your MOUTH wide',
}


class ChallengeEngine:
    """
    Manages the full challenge-response lifecycle for one verification session.

    Design decisions
    ----------------
    * Challenges are drawn randomly without replacement — the same action
      can never appear twice, preventing simple replay attacks.
    * HOLD_FRAMES_REQUIRED consecutive detections are needed to count as a
      PASS, preventing a single-frame fluke (blink artefact etc.) from passing.
    * Timeout is wall-clock based — pausing or face loss doesn't reset the clock.
    """

    # Consecutive frames the gesture must be held to count as a PASS
    HOLD_FRAMES_REQUIRED = 8

    def __init__(self, baseline: dict, count: int = CHALLENGE_COUNT):
        """
        Parameters
        ----------
        baseline : dict   Personalised thresholds from Calibrator.
        count    : int    Number of challenges to issue this session.
        """
        self.baseline  = baseline
        self._selected = random.sample(CHALLENGES, min(count, len(CHALLENGES)))

        self.queue     = deque(self._selected)
        self.current   : str | None = None
        self.start_time: float      = 0.0
        self.completed : list[str]  = []
        self.failed    : list[str]  = []

        self._hold_counter: int = 0

        print(f"\n-- Challenges selected: {self._selected}")

    # ── public ────────────────────────────────────────────────────────────

    def next_challenge(self) -> None:
        """Pop the next challenge from the queue and start its timer."""
        if self.queue:
            self.current       = self.queue.popleft()
            self.start_time    = time.time()
            self._hold_counter = 0
            print(f"   Challenge: {self.current}")
        else:
            self.current = None   # all done

    def check(self, face_landmarks, w: int, h: int) -> str:
        """
        Evaluate the current challenge against live landmark data.

        Call this once per frame while in CHALLENGING state.

        Returns
        -------
        'PASS'    gesture held long enough — challenge complete
        'FAIL'    timeout exceeded
        'WAITING' still within window, keep going
        """
        if self.current is None:
            return 'WAITING'

        # ── Timeout check (wall clock — face loss doesn't pause it) ────────
        elapsed = time.time() - self.start_time
        if elapsed > CHALLENGE_TIMEOUT:
            self.failed.append(self.current)
            print(f"   FAIL (timeout): {self.current}")
            self.next_challenge()
            return 'FAIL'

        # ── Gesture detection ──────────────────────────────────────────────
        if self._detect(face_landmarks, w, h):
            self._hold_counter += 1
        else:
            self._hold_counter = 0   # must be held continuously

        if self._hold_counter >= self.HOLD_FRAMES_REQUIRED:
            self.completed.append(self.current)
            print(f"   PASS: {self.current}")
            self.next_challenge()
            return 'PASS'

        return 'WAITING'

    def is_finished(self) -> bool:
        """True when every challenge has been resolved (pass or fail)."""
        return self.current is None and len(self.queue) == 0

    def result(self) -> str:
        """'VERIFIED' if no failures, 'REJECTED' if any challenge timed out."""
        return 'VERIFIED' if len(self.failed) == 0 else 'REJECTED'

    def time_remaining(self) -> float:
        """Seconds remaining for the current challenge, clamped to 0."""
        if self.current is None:
            return 0.0
        return max(0.0, CHALLENGE_TIMEOUT - (time.time() - self.start_time))

    def selected_challenges(self) -> list:
        """The full ordered list of challenges chosen for this session."""
        return self._selected

    # ── private ───────────────────────────────────────────────────────────

    def _detect(self, face_landmarks, w: int, h: int) -> bool:
        """Return True if the current challenge gesture is present this frame."""
        c = self.current

        if c == 'BLINK':
            # Both eyes must close simultaneously (anti-wink spoof)
            left_ear  = eye_aspect_ratio(face_landmarks,
                                         LEFT_EYE_TOP_BOTTOM[0],  LEFT_EYE_TOP_BOTTOM[1],
                                         LEFT_EYE_LEFT_RIGHT[0],  LEFT_EYE_LEFT_RIGHT[1],
                                         w, h)
            right_ear = eye_aspect_ratio(face_landmarks,
                                         RIGHT_EYE_TOP_BOTTOM[0], RIGHT_EYE_TOP_BOTTOM[1],
                                         RIGHT_EYE_LEFT_RIGHT[0], RIGHT_EYE_LEFT_RIGHT[1],
                                         w, h)
            thr = self.baseline['blink_threshold']
            return left_ear < thr and right_ear < thr

        if c == 'OPEN_MOUTH':
            mar = mouth_aspect_ratio(face_landmarks, w, h)
            return mar > self.baseline['mouth_threshold']

        # All head-pose challenges share the same pose calculation
        pose = head_pose_ratios(face_landmarks, w, h)

        if c == 'TURN_LEFT':
            return pose['horizontal'] < -HEAD_TURN_THRESHOLD
        if c == 'TURN_RIGHT':
            return pose['horizontal'] >  HEAD_TURN_THRESHOLD
        if c == 'LOOK_UP':
            return pose['vertical']   < -HEAD_NOD_THRESHOLD
        if c == 'LOOK_DOWN':
            return pose['vertical']   >  HEAD_NOD_THRESHOLD

        return False   # unknown challenge — safe default
