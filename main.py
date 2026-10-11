# main.py — Liveness Detection Anti-Spoofing System
# ──────────────────────────────────────────────────
# States: CALIBRATING → CHALLENGING → CLASSIFYING → RESULT
#
# Controls
# --------
#   Q — quit at any time
#   R — restart from CALIBRATING (works in any state)

import time
import cv2
import mediapipe as mp

from config import (
    WINDOW_NAME, CHALLENGE_COUNT,
    COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_DARK, COLOR_WHITE,
)
from calibration      import Calibrator
from challenge_engine import ChallengeEngine, CHALLENGE_MESSAGES
from classifier       import SpoofClassifier
from landmark_utils   import extract_feature_vector
import ui


# ── Application state container ───────────────────────────────────────────

def make_state() -> dict:
    """Return a fresh application-state dict."""
    return {
        'phase'        : 'CALIBRATING',
        'calibrator'   : Calibrator(),
        'engine'       : None,
        'classifier'   : SpoofClassifier(),
        'final_result' : None,
        'result_time'  : 0.0,
        'prediction'   : None,
    }


# ── MediaPipe helper ──────────────────────────────────────────────────────

def build_face_mesh():
    """Construct and return a MediaPipe FaceMesh instance."""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# ── State handlers ────────────────────────────────────────────────────────

def handle_calibrating(frame, rgb, face_mesh, state: dict, h: int, w: int) -> None:
    """Collect baseline measurements; transition to CHALLENGING when done."""
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]
        state['calibrator'].collect(lm, w, h)
        ui.draw_landmarks(frame, lm, w, h)
    else:
        ui.draw_no_face_warning(frame)

    ui.draw_status_bar(frame, "Step 1 / 3  -  Calibrating  (hold still, look at camera)", COLOR_DARK)
    ui.draw_calibration_bar(frame, state['calibrator'].progress())

    if state['calibrator'].is_done:
        engine = ChallengeEngine(
            baseline=state['calibrator'].baseline,
            count=CHALLENGE_COUNT,
        )
        engine.next_challenge()
        state['engine'] = engine
        state['phase']  = 'CHALLENGING'
        print("[App] -> CHALLENGING")


def handle_challenging(frame, rgb, face_mesh, state: dict, h: int, w: int) -> None:
    """Issue random challenges; verify each via landmarks; collect SVM frames."""
    engine  = state['engine']
    results = face_mesh.process(rgb)

    # Challenge counter — clamp so it never exceeds CHALLENGE_COUNT
    done_count = len(engine.completed) + len(engine.failed)
    current_num = min(done_count + 1, CHALLENGE_COUNT)
    ui.draw_status_bar(
        frame,
        f"Step 2 / 3  -  Challenge {current_num} of {CHALLENGE_COUNT}",
        (40, 100, 20),
    )

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0]

        # Feed classifier buffer every frame
        fv = extract_feature_vector(lm, w, h)
        state['classifier'].add_frame(fv)

        # Evaluate current challenge
        engine.check(lm, w, h)

        ui.draw_landmarks(frame, lm, w, h)
        ui.draw_completed_challenges(frame, engine.completed, engine.failed)

        # Draw challenge instruction box while a challenge is active
        if engine.current:
            msg = CHALLENGE_MESSAGES.get(engine.current, engine.current)
            ui.draw_challenge_box(frame, msg, engine.time_remaining())

        # SVM live badge
        pred = state['classifier'].predict()
        if pred:
            state['prediction'] = pred
            ui.draw_classifier_badge(frame, pred)

        if engine.is_finished():
            state['phase'] = 'CLASSIFYING'
            print("[App] -> CLASSIFYING")
    else:
        ui.draw_no_face_warning(frame)
        # Keep showing the challenge box even when face is momentarily lost
        if engine.current:
            msg = CHALLENGE_MESSAGES.get(engine.current, engine.current)
            ui.draw_challenge_box(frame, msg, engine.time_remaining())
        ui.draw_completed_challenges(frame, engine.completed, engine.failed)


def handle_classifying(frame, state: dict) -> None:
    """Combine challenge result + SVM prediction → final verdict (instant)."""
    engine     = state['engine']
    prediction = state['prediction']

    challenge_passed = engine.result() == 'VERIFIED'

    if prediction is not None:
        ml_real = prediction['label'] == 'REAL'
        final   = 'VERIFIED' if (challenge_passed and ml_real) else 'REJECTED'
        print(f"[App] Challenge={engine.result()}  ML={prediction['label']}  -> {final}")
    else:
        # No model loaded — challenge result alone decides
        final = 'VERIFIED' if challenge_passed else 'REJECTED'
        print(f"[App] Challenge={engine.result()}  ML=N/A  -> {final}")

    state['final_result'] = final
    state['result_time']  = time.time()
    state['phase']        = 'RESULT'
    print(f"[App] -> RESULT ({final})")


def handle_result(frame, state: dict) -> None:
    """Display final result overlay."""
    ui.draw_result_screen(frame, state['final_result'])


# ── Main loop ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  Liveness Detection -- Anti-Spoofing System")
    print("=" * 62)
    print("  Controls:  Q = quit    R = restart anytime")
    print("=" * 62 + "\n")

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check that your camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    face_mesh = build_face_mesh()

    # ── Initial state ─────────────────────────────────────────────────────
    state = make_state()
    state['classifier'].load()   # graceful if model not found

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        frame = cv2.flip(frame, 1)           # mirror — feels natural to user
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # tiny perf win for MediaPipe

        phase = state['phase']

        # ── FSM dispatch ──────────────────────────────────────────────────
        if phase == 'CALIBRATING':
            handle_calibrating(frame, rgb, face_mesh, state, h, w)

        elif phase == 'CHALLENGING':
            handle_challenging(frame, rgb, face_mesh, state, h, w)

        elif phase == 'CLASSIFYING':
            # Classifying is instant — compute verdict then fall through to RESULT
            handle_classifying(frame, state)
            handle_result(frame, state)   # show result on this same frame

        elif phase == 'RESULT':
            handle_result(frame, state)

        # ── MUST show frame BEFORE waitKey on Windows ──────────────────────
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[App] Quit.")
            break

        if key == ord('r'):
            print("[App] Restarting...\n")
            old_clf = state['classifier']   # keep loaded model across restarts
            state   = make_state()
            state['classifier'] = old_clf

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
