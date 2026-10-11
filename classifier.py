# classifier.py — Feature buffer and SVM-based spoof classifier.

import os
import joblib
import numpy as np
from collections import deque
from config import BUFFER_SIZE, MODEL_PATH, FEATURE_VECTOR_SIZE


class SpoofClassifier:
    """
    Thin wrapper around a pre-trained sklearn SVM that classifies whether
    the current face is REAL or SPOOF based on a rolling window of landmark
    feature vectors.

    Why a rolling buffer?
    ---------------------
    A single frame can be noisy (motion blur, partial blink). Averaging
    BUFFER_SIZE frames before predicting smooths out transient artefacts
    and gives a more stable classification.
    """

    def __init__(self):
        self.model        = None
        self.scaler       = None
        self.frame_buffer : deque = deque(maxlen=BUFFER_SIZE)
        self.is_loaded    = False

    # ── public ────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Load the saved model + scaler bundle from MODEL_PATH.

        Returns True on success, False if the file is missing or corrupt.
        The system degrades gracefully — challenge-response still works
        without the classifier.
        """
        if not os.path.exists(MODEL_PATH):
            print(f"[Classifier] Model not found at '{MODEL_PATH}'. "
                  "Running in rule-based mode only.")
            return False

        try:
            bundle       = joblib.load(MODEL_PATH)
            self.model   = bundle['model']
            self.scaler  = bundle['scaler']
            self.is_loaded = True
            print(f"[Classifier] Model loaded from '{MODEL_PATH}'.")
            return True
        except Exception as exc:
            print(f"[Classifier] Failed to load model: {exc}")
            return False

    def add_frame(self, feature_vector: list) -> None:
        """Append one frame's feature vector to the rolling buffer."""
        if len(feature_vector) == FEATURE_VECTOR_SIZE:
            self.frame_buffer.append(feature_vector)

    def predict(self) -> dict | None:
        """
        Average the buffered feature vectors and run the SVM.

        Returns
        -------
        dict with keys 'label', 'confidence', 'raw_proba'
        or None if the buffer is not yet full or the model is not loaded.
        """
        if not self.is_loaded or len(self.frame_buffer) < BUFFER_SIZE:
            return None

        try:
            avg_features = np.mean(list(self.frame_buffer), axis=0).reshape(1, -1)
            scaled       = self.scaler.transform(avg_features)
            proba        = self.model.predict_proba(scaled)[0]   # [spoof_p, real_p]

            # Convention: class 0 = SPOOF, class 1 = REAL
            real_prob  = float(proba[1])
            spoof_prob = float(proba[0])
            label      = 'REAL' if real_prob >= 0.5 else 'SPOOF'

            return {
                'label'     : label,
                'confidence': real_prob if label == 'REAL' else spoof_prob,
                'raw_proba' : [spoof_prob, real_prob],
            }
        except Exception as exc:
            print(f"[Classifier] Prediction error: {exc}")
            return None
