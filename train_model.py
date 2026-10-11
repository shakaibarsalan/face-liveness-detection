# train_model.py — Train and save the SVM spoof classifier.
#
# Usage
# -----
#   python train_model.py
#
# Requirements
# ------------
#   data/real/  — one or more CSV files collected with data_collector.py --label real
#   data/spoof/ — one or more CSV files collected with data_collector.py --label spoof

import os
import glob
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from config import MODEL_PATH


def load_class(directory: str, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Load all CSV files from *directory* and tag them with *label*."""
    pattern = os.path.join(directory, "*.csv")
    files   = glob.glob(pattern)
    if not files:
        return np.empty((0, 10)), np.empty((0,), dtype=int)

    rows = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values = [float(v) for v in line.split(",")]
                if len(values) == 10:
                    rows.append(values)

    X = np.array(rows)
    y = np.full(len(X), label, dtype=int)
    print(f"   Loaded {len(X):>5} samples from '{directory}' (label={label})")
    return X, y


def main():
    print("\n── Training Spoof Classifier ────────────────────────────────")

    X_real,  y_real  = load_class("data/real",  label=1)
    X_spoof, y_spoof = load_class("data/spoof", label=0)

    if len(X_real) == 0 or len(X_spoof) == 0:
        print("\n[ERROR] Need samples in both data/real/ and data/spoof/")
        print("  Run: python data_collector.py --label real")
        print("  Run: python data_collector.py --label spoof")
        return

    X = np.vstack([X_real, X_spoof])
    y = np.hstack([y_real, y_spoof])

    print(f"\n   Total samples : {len(X)}  (real={len(X_real)}, spoof={len(X_spoof)})")

    # ── Train / test split (stratified) ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Train / test  : {len(X_train)} / {len(X_test)}")

    # ── Scaling ────────────────────────────────────────────────────────────
    # Fit ONLY on training data — never let test data influence scaling
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── SVM with RBF kernel ────────────────────────────────────────────────
    print("\n   Training SVM (RBF kernel) …")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred = svm.predict(X_test)

    print("\n── Evaluation ───────────────────────────────────────────────")
    print(f"   Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"   F1-score  : {f1_score(y_test, y_pred):.4f}")

    print("\n── Classification Report ─────────────────────────────────────")
    print(classification_report(y_test, y_pred,
                                target_names=["SPOOF", "REAL"]))

    print("── Confusion Matrix ──────────────────────────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Pred SPOOF  Pred REAL")
    print(f"  True SPOOF     {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  True REAL      {cm[1][0]:>5}      {cm[1][1]:>5}")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({'model': svm, 'scaler': scaler}, MODEL_PATH)
    print(f"\n   Model saved → {MODEL_PATH}")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
