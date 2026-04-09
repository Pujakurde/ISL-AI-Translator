from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import pickle
from pathlib import Path
import io
try:
    import mediapipe as mp
    _mp_hands_module = mp.solutions.hands
except Exception:
    try:
        from mediapipe.python.solutions import hands as _mp_hands_module
    except Exception as _mp_err:
        _mp_hands_module = None
        _mp_import_error = _mp_err

# ---------- Config ----------
MODELS_DIR = Path("models")
MODEL_PATHS = {
    "number": MODELS_DIR / "isl_number_model.h5",
    "onehand": MODELS_DIR / "isl_onehand_model.h5",
    "twohand": MODELS_DIR / "isl_twohand_model.h5",
}
LABEL_PATHS = {
    "number": MODELS_DIR / "isl_number_labels.json",
    "onehand": MODELS_DIR / "isl_onehand_labels.json",
    "twohand": MODELS_DIR / "isl_twohand_labels.json",
}
LABEL_PKL_FALLBACK = MODELS_DIR / "label_encoder.pkl"

# ---------- App ----------
app = FastAPI(title="ISL Translator API", version="1.0.0")

# CORS (allow all during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load Models + Labels ----------
MODELS = {}
LABELS = {}
INPUT_SIZES = {}

def load_labels(path: Path):
    """Supports both {"A": 0} and {"0": "A"} formats."""
    with open(path, "r") as f:
        data = json.load(f)
    if data and isinstance(list(data.values())[0], int):
        # {"A": 0} -> {0: "A"}
        return {v: k for k, v in data.items()}
    # {"0": "A"} -> {0: "A"}
    return {int(k): v for k, v in data.items()}

def load_labels_from_pickle(path: Path):
    with open(path, "rb") as f:
        le = pickle.load(f)
    return {i: label for i, label in enumerate(le.classes_)}

print("\n=== Loading ISL Models ===")
for key, model_path in MODEL_PATHS.items():
    if not model_path.exists():
        print(f"[WARN] Missing model: {model_path}")
        continue
    try:
        MODELS[key] = tf.keras.models.load_model(str(model_path))
        if LABEL_PATHS[key].exists():
            LABELS[key] = load_labels(LABEL_PATHS[key])
        elif LABEL_PKL_FALLBACK.exists() and key == "twohand":
            LABELS[key] = load_labels_from_pickle(LABEL_PKL_FALLBACK)
        else:
            LABELS[key] = {}
        INPUT_SIZES[key] = MODELS[key].input_shape
        print(f"[OK] Loaded {key} model + labels")
    except Exception as e:
        print(f"[ERROR] Failed to load {key}: {e}")
print("=== Model Loading Complete ===\n")

# ---------- Helpers ----------
def is_landmark_model(model_type: str) -> bool:
    shape = INPUT_SIZES.get(model_type)
    return shape is not None and len(shape) == 2

def expected_feature_len(model_type: str) -> int:
    shape = INPUT_SIZES.get(model_type)
    if not shape or len(shape) < 2:
        return 0
    return int(shape[1])

def expected_image_size(model_type: str) -> tuple[int, int]:
    shape = INPUT_SIZES.get(model_type)
    if shape and len(shape) == 4:
        h, w = int(shape[1]), int(shape[2])
        return (w, h)
    return (64, 64)

def preprocess_image(image_bytes: bytes, model_type: str) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(expected_image_size(model_type))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

if _mp_hands_module is None:
    raise RuntimeError(
        f"MediaPipe import failed: {_mp_import_error}. "
        "Reinstall mediapipe or ensure no local mediapipe.py shadows the package."
    )

mp_hands = _mp_hands_module.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def extract_landmarks(image_bytes: bytes, model_type: str) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image)
    results = mp_hands.process(image_np)

    if not results.multi_hand_landmarks:
        raise ValueError("No hands detected")

    # Sort hands by confidence (if available)
    hand_items = list(zip(
        results.multi_hand_landmarks,
        results.multi_handedness or [None] * len(results.multi_hand_landmarks)
    ))
    hand_items.sort(
        key=lambda x: (x[1].classification[0].score if x[1] else 0),
        reverse=True
    )

    coords = []
    for hand_lms, _ in hand_items:
        for lm in hand_lms.landmark:
            coords.extend([lm.x, lm.y])

    coords = np.array(coords, dtype=np.float32)

    # Normalize only for twohand model (based on training reference)
    if model_type == "twohand" and coords.size >= 4:
        lm = coords.reshape(-1, 2)
        center = np.mean(lm, axis=0)
        lm = lm - center
        max_val = np.max(np.linalg.norm(lm, axis=1))
        if max_val > 0:
            lm = lm / max_val
        coords = lm.flatten()

    expected_len = expected_feature_len(model_type)
    if expected_len:
        if coords.size < expected_len:
            coords = np.pad(coords, (0, expected_len - coords.size))
        elif coords.size > expected_len:
            coords = coords[:expected_len]

    return np.expand_dims(coords, axis=0)

# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "ISL Translator API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys())
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...)
):
    if model_type not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Invalid model_type. Use: number, onehand, twohand")
    if model_type not in MODELS:
        raise HTTPException(status_code=503, detail=f"Model '{model_type}' not loaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        if is_landmark_model(model_type):
            img = extract_landmarks(image_bytes, model_type)
        else:
            img = preprocess_image(image_bytes, model_type)
        preds = MODELS[model_type].predict(img, verbose=0)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        label = LABELS[model_type].get(idx, str(idx))
        return {"prediction": label, "confidence": round(conf, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
