# data_collector.py — Collect labelled training samples for the SVM classifier.
#
# Usage
# -----
#   python data_collector.py --label real
#   python data_collector.py --label spoof
#
# Controls
# --------
#   S  — save collected frames to a CSV file and reset the buffer
#   Q  — quit

import argparse
import csv
import os
import time
import cv2
import mediapipe as mp

from landmark_utils import extract_feature_vector
from config import WINDOW_NAME, COLOR_GREEN, COLOR_RED, COLOR_WHITE, COLOR_DARK

TARGET_SAMPLES = 150   # recommended minimum per class


def main():
    parser = argparse.ArgumentParser(description="Collect liveness training data")
    parser.add_argument("--label", choices=["real", "spoof"], required=True,
                        help="Class label for this recording session")
    args = parser.parse_args()

    label     = args.label
    save_dir  = os.path.join("data", label)
    os.makedirs(save_dir, exist_ok=True)

    color     = COLOR_GREEN if label == "real" else COLOR_RED
    mode_text = f"Mode: {label.upper()}"

    # ── MediaPipe setup ────────────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    collected: list[list[float]] = []
    total_saved = 0

    print(f"\n── Data Collector  (label = {label.upper()}) ─────────────────")
    print("   S = save buffer to CSV")
    print("   Q = quit")
    print(f"   Target: {TARGET_SAMPLES} samples per save\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_found = False
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            fv = extract_feature_vector(lm, w, h)
            collected.append(fv)
            face_found = True

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 38), COLOR_DARK, -1)
        cv2.putText(frame, mode_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        status = f"Collected: {len(collected)}  |  Saved total: {total_saved}"
        cv2.putText(frame, status, (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

        if not face_found:
            cv2.putText(frame, "No face — look at camera", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        progress = min(len(collected) / TARGET_SAMPLES, 1.0)
        bar_x2   = int(w * progress)
        cv2.rectangle(frame, (0, h - 8), (bar_x2, h), color, -1)

        cv2.imshow(WINDOW_NAME + " — Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and collected:
            ts       = int(time.time())
            filename = os.path.join(save_dir, f"{label}_{ts}.csv")
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(collected)
            total_saved += len(collected)
            print(f"   Saved {len(collected)} samples → {filename}")
            collected.clear()

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print(f"\n   Session ended. Total samples saved: {total_saved}")


if __name__ == "__main__":
    main()
