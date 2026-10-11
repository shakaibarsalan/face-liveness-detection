# config.py — All constants and settings. No magic numbers anywhere else.

# ── MediaPipe landmark indices ──────────────────────────────────────────────
LEFT_EYE_TOP_BOTTOM  = [386, 374]   # top / bottom lid (left eye)
RIGHT_EYE_TOP_BOTTOM = [159, 145]   # top / bottom lid (right eye)
LEFT_EYE_LEFT_RIGHT  = [263, 362]   # outer / inner corner (left eye)
RIGHT_EYE_LEFT_RIGHT = [133, 33]    # outer / inner corner (right eye)

MOUTH_TOP_BOTTOM = [13, 14]         # upper / lower lip centre
MOUTH_LEFT_RIGHT = [78, 308]        # left / right mouth corners

NOSE_TIP    = 1
CHIN        = 199
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454
FOREHEAD    = 10

# ── Detection thresholds (relative — calibrated per user at runtime) ────────
EAR_BLINK_RATIO   = 0.75   # blink_threshold  = baseline_EAR * this
MAR_OPEN_RATIO    = 1.60   # mouth_threshold  = baseline_MAR * this
HEAD_TURN_THRESHOLD = 0.15  # |horizontal ratio| that counts as a head turn
HEAD_NOD_THRESHOLD  = 0.12  # |vertical ratio|   that counts as a nod

# ── Timing ──────────────────────────────────────────────────────────────────
CALIBRATION_FRAMES = 60    # frames collected for baseline (~2 s at 30 fps)
CHALLENGE_TIMEOUT  = 6     # seconds the user has to complete each challenge
CHALLENGE_COUNT    = 3     # how many challenges are issued per session
BUFFER_SIZE        = 15    # rolling frame buffer used by the SVM classifier

# ── Classifier ──────────────────────────────────────────────────────────────
FEATURE_VECTOR_SIZE = 10
MODEL_PATH          = "models/spoof_classifier.pkl"

# ── UI colours (BGR for OpenCV) ─────────────────────────────────────────────
COLOR_GREEN  = (0, 220, 100)
COLOR_RED    = (0, 60, 220)
COLOR_YELLOW = (0, 200, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_DARK   = (20, 20, 20)
COLOR_BLUE   = (200, 100, 0)

# ── Window ───────────────────────────────────────────────────────────────────
WINDOW_NAME = "Liveness Detection — Anti-Spoofing"
