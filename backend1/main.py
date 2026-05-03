from __future__ import annotations

import io
import json
import os
import string
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow.keras.models import load_model

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
DATASET_DIR = API_DIR / "dataset"
INDIAN_DATASET_DIR = DATASET_DIR / "Indian"
MODELS_DIR = API_DIR / "models"
FRONTEND_DIST_DIR = PROJECT_ROOT / "ISL-Converter" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"

CAMERA_PADDING = 24
CONFIRMATION_DELAY = 1.2
CAMERA_CONFIDENCE_THRESHOLDS = {
    "number": 0.75,
    "onehand": 0.75,
    "twohand": 0.75,
}
CAMERA_MARGIN_THRESHOLD = 0.15
FULL_FRAME_NUMBER_CONFIDENCE_THRESHOLD = 0.95
FULL_FRAME_NUMBER_MARGIN_THRESHOLD = 0.2
ROUTER_CONFIDENCE_THRESHOLD = 0.6
FULL_FRAME_FALLBACK_CONFIDENCE_THRESHOLD = 0.95
FULL_FRAME_FALLBACK_MARGIN_THRESHOLD = 0.2
FULL_FRAME_OVERRIDE_CONFIDENCE_THRESHOLD = 0.99
ONEHAND_CLASS_SET = {"C", "I", "L", "O", "U", "V"}
NUMBER_CLASS_SET = set(string.digits)
VALID_MODEL_TYPES = {"number", "onehand", "twohand"}
LIVE_ONEHAND_LABELS = ["C", "I", "J", "L", "O", "U", "V"]
LIVE_ONEHAND_CLASS_SET = set(LIVE_ONEHAND_LABELS)
LIVE_TWOHAND_LABELS = [
    "A",
    "B",
    "D",
    "E",
    "F",
    "G",
    "H",
    "K",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "W",
    "X",
    "Y",
    "Z",
]
LIVE_TWOHAND_CLASS_SET = set(LIVE_TWOHAND_LABELS)
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class PredictionUpdate(BaseModel):
    text: str = ""


def parse_allowed_origins() -> list[str]:
    origins = list(DEFAULT_ALLOWED_ORIGINS)
    raw_value = os.getenv("ALLOWED_ORIGINS", "")
    if raw_value.strip():
        for origin in raw_value.split(","):
            normalized = origin.strip()
            if normalized and normalized not in origins:
                origins.append(normalized)
    return origins


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "None of the expected paths exist: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_labels(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data and isinstance(next(iter(data.values())), int):
        return {value: key for key, value in data.items()}
    return {int(key): value for key, value in data.items()}


MODEL_PATHS = {
    "number": resolve_existing_path(MODELS_DIR / "isl_number_model.h5"),
    "onehand": resolve_existing_path(MODELS_DIR / "isl_onehand_model.h5"),
    "twohand": resolve_existing_path(MODELS_DIR / "isl_twohand_model.h5"),
}
LABEL_PATHS = {
    "number": resolve_existing_path(MODELS_DIR / "isl_number_labels.json"),
    "onehand": resolve_existing_path(MODELS_DIR / "isl_onehand_labels.json"),
    "twohand": resolve_existing_path(MODELS_DIR / "isl_twohand_labels.json"),
}
ROUTER_MODEL_PATH = resolve_existing_path(
    API_DIR / "image_hand_gesture_img_model.h5",
    MODELS_DIR / "image_hand_gesture_model.h5",
)
ROUTER_MAPPING_PATH = resolve_existing_path(
    DATASET_DIR / "class_mapping.npy",
    API_DIR / "class_mapping.npy",
)
LIVE_MODEL_PATHS = {
    "number": resolve_existing_path(MODELS_DIR / "number_gesture_model.h5"),
    "onehand": resolve_existing_path(MODELS_DIR / "onehand_gesture_model.h5"),
    "twohand": resolve_existing_path(MODELS_DIR / "twohand_gesture_model.h5"),
}
LIVE_LABELS = {
    "number": {index: label for index, label in enumerate(sorted(NUMBER_CLASS_SET))},
    "onehand": {index: label for index, label in enumerate(LIVE_ONEHAND_LABELS)},
    "twohand": {index: label for index, label in enumerate(LIVE_TWOHAND_LABELS)},
}
MODELS: dict[str, Any] = {}
LABELS: dict[str, dict[int, str]] = {}
ROUTER_MODEL: Any = None
LIVE_MODELS: dict[str, Any] = {}
IMAGE_ONEHAND_CLASS_SET: set[str] = set()
IMAGE_TWOHAND_CLASS_SET: set[str] = set()
ROUTER_LABELS: dict[int, str] = {}
INPUT_SIZES: dict[str, tuple[int, int]] = {}
LIVE_HANDS: Any = None
MODELS_READY = False
MODEL_LOAD_ERROR = ""
MODEL_LOAD_STARTED = False
MODEL_LOAD_LOCK = threading.Lock()

cap: Optional[cv2.VideoCapture] = None
sign_state = {
    "current_word": "",
    "mode": "alphabet",
    "running": True,
    "active_model": "",
    "active_confidence": 0.0,
}
pending_prediction = {"label": "", "start_time": 0.0}
LIVE_HANDS_LOCK = threading.Lock()


def load_runtime_assets() -> None:
    global MODELS
    global LABELS
    global ROUTER_MODEL
    global LIVE_MODELS
    global IMAGE_ONEHAND_CLASS_SET
    global IMAGE_TWOHAND_CLASS_SET
    global ROUTER_LABELS
    global INPUT_SIZES
    global LIVE_HANDS
    global MODELS_READY
    global MODEL_LOAD_ERROR

    try:
        models = {model_type: load_model(str(path)) for model_type, path in MODEL_PATHS.items()}
        labels = {model_type: load_labels(path) for model_type, path in LABEL_PATHS.items()}
        router_model = load_model(str(ROUTER_MODEL_PATH))
        live_models = {model_type: load_model(str(path)) for model_type, path in LIVE_MODEL_PATHS.items()}
        router_labels = {
            value: key
            for key, value in np.load(str(ROUTER_MAPPING_PATH), allow_pickle=True).item().items()
        }
        input_sizes = {
            model_type: (
                int(model.input_shape[2]),
                int(model.input_shape[1]),
            )
            for model_type, model in models.items()
        }
        live_hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2,
        )
    except Exception as exc:
        with MODEL_LOAD_LOCK:
            MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            MODELS_READY = False
        print(f"Failed to load runtime assets: {MODEL_LOAD_ERROR}")
        return

    with MODEL_LOAD_LOCK:
        MODELS = models
        LABELS = labels
        ROUTER_MODEL = router_model
        LIVE_MODELS = live_models
        IMAGE_ONEHAND_CLASS_SET = set(labels["onehand"].values())
        IMAGE_TWOHAND_CLASS_SET = set(labels["twohand"].values())
        ROUTER_LABELS = router_labels
        INPUT_SIZES = input_sizes
        LIVE_HANDS = live_hands
        MODEL_LOAD_ERROR = ""
        MODELS_READY = True


def start_runtime_asset_load() -> None:
    global MODEL_LOAD_STARTED

    with MODEL_LOAD_LOCK:
        if MODEL_LOAD_STARTED or MODELS_READY:
            return
        MODEL_LOAD_STARTED = True

    threading.Thread(target=load_runtime_assets, name="runtime-asset-loader", daemon=True).start()


def require_models_ready() -> None:
    start_runtime_asset_load()
    if MODEL_LOAD_ERROR:
        raise HTTPException(status_code=503, detail=f"Runtime assets failed to load: {MODEL_LOAD_ERROR}")
    if not MODELS_READY:
        raise HTTPException(status_code=503, detail="Runtime assets are still loading. Please retry shortly.")

app = FastAPI(title="ISL Translator Backend", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
if FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="frontend-assets")


@app.on_event("startup")
def load_runtime_assets_on_startup() -> None:
    start_runtime_asset_load()

BASE_PATH = resolve_existing_path(INDIAN_DATASET_DIR)
alphabet_dict: dict[str, str] = {}
for ch in string.ascii_uppercase + string.digits:
    jpg_path = BASE_PATH / ch / "4.jpg"
    png_path = BASE_PATH / ch / "4.png"
    if jpg_path.exists():
        alphabet_dict[ch] = str(jpg_path)
    elif png_path.exists():
        alphabet_dict[ch] = str(png_path)

word_list: list[str] = []
try:
    import nltk
    from nltk.corpus import words

    try:
        word_list = [word.lower() for word in words.words()]
    except Exception:
        nltk.download("words", quiet=True)
        word_list = [word.lower() for word in words.words()]
except Exception:
    fallback_path = PROJECT_ROOT / "words.txt"
    if fallback_path.exists():
        with fallback_path.open("r", encoding="utf-8", errors="ignore") as file:
            word_list = [word.strip().lower() for word in file if word.strip()]


def suggest_words(prefix: str, max_suggestions: int = 6) -> list[str]:
    if not prefix or not word_list:
        return []
    normalized_prefix = prefix.lower()
    return [word for word in word_list if word.startswith(normalized_prefix)][:max_suggestions]


def open_capture(index: int) -> Optional[cv2.VideoCapture]:
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        try:
            capture = cv2.VideoCapture(index, backend)
        except Exception:
            capture = cv2.VideoCapture(index)
        if capture is not None and capture.isOpened():
            print(f"Opened camera index {index} backend {backend}")
            return capture
        if capture is not None:
            capture.release()
    return None


def get_video_capture() -> Optional[cv2.VideoCapture]:
    global cap
    if cap is not None and cap.isOpened():
        return cap
    for index in range(5):
        capture = open_capture(index)
        if capture is not None:
            cap = capture
            return cap
    print("Error: Camera not accessible. Tried indexes 0-4.")
    return None


def release_video_capture() -> None:
    global cap
    if cap is not None:
        cap.release()
        cap = None


def preprocess_rgb_image(rgb_image: np.ndarray, model_type: str) -> np.ndarray:
    width, height = INPUT_SIZES.get(model_type, (64, 64))
    resized = cv2.resize(rgb_image, (width, height))
    image_array = resized.astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def load_rgb_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def preprocess_image_bytes(image_bytes: bytes, model_type: str) -> np.ndarray:
    return preprocess_rgb_image(load_rgb_image_from_bytes(image_bytes), model_type)


def predict_batch(image_array: np.ndarray, model_type: str) -> tuple[str, float, float]:
    predictions = MODELS[model_type].predict(image_array, verbose=0)[0]
    ranked = np.argsort(predictions)
    top_index = int(ranked[-1])
    top_confidence = float(predictions[top_index])
    second_confidence = float(predictions[ranked[-2]]) if len(ranked) > 1 else 0.0
    predicted_label = LABELS[model_type].get(top_index, str(top_index))
    return predicted_label, top_confidence, top_confidence - second_confidence


def predict_rgb_image(rgb_image: np.ndarray, model_type: str) -> tuple[str, float, float]:
    return predict_batch(preprocess_rgb_image(rgb_image, model_type), model_type)


def predict_router_label(rgb_image: np.ndarray) -> tuple[str, float]:
    predictions = ROUTER_MODEL.predict(preprocess_rgb_image(rgb_image, "number"), verbose=0)[0]
    index = int(np.argmax(predictions))
    return ROUTER_LABELS.get(index, str(index)), float(np.max(predictions))


def resolve_label_model_type(label: str) -> Optional[str]:
    if label in NUMBER_CLASS_SET:
        return "number"
    if label in IMAGE_ONEHAND_CLASS_SET:
        return "onehand"
    if label in IMAGE_TWOHAND_CLASS_SET:
        return "twohand"
    return None


def predict_full_frame_fallback(rgb_image: np.ndarray) -> tuple[str, float, str]:
    router_label, router_confidence = predict_router_label(rgb_image)
    model_type = resolve_label_model_type(router_label)
    if model_type is None:
        return "", router_confidence, ""

    image_label, image_confidence, image_margin = predict_rgb_image(rgb_image, model_type)
    if (
        image_label == router_label
        and router_confidence >= FULL_FRAME_FALLBACK_CONFIDENCE_THRESHOLD
        and image_confidence >= FULL_FRAME_FALLBACK_CONFIDENCE_THRESHOLD
        and image_margin >= FULL_FRAME_FALLBACK_MARGIN_THRESHOLD
    ):
        return image_label, max(router_confidence, image_confidence), f"{model_type}-image-fallback"

    return "", max(router_confidence, image_confidence), f"{model_type}-image-fallback"


def should_prefer_full_frame_fallback(
    live_label: str,
    fallback_label: str,
    fallback_confidence: float,
) -> bool:
    return (
        bool(fallback_label)
        and fallback_label != live_label
        and fallback_confidence >= FULL_FRAME_OVERRIDE_CONFIDENCE_THRESHOLD
    )


def decode_prediction(
    predictions: np.ndarray,
    label_map: dict[int, str],
) -> tuple[str, float, float]:
    ranked = np.argsort(predictions)
    top_index = int(ranked[-1])
    top_confidence = float(predictions[top_index])
    second_confidence = float(predictions[ranked[-2]]) if len(ranked) > 1 else 0.0
    predicted_label = label_map.get(top_index, str(top_index))
    return predicted_label, top_confidence, top_confidence - second_confidence


def predict_live_features(features: np.ndarray, model_type: str) -> tuple[str, float, float]:
    predictions = LIVE_MODELS[model_type].predict(features, verbose=0)[0]
    return decode_prediction(predictions, LIVE_LABELS[model_type])


def normalize_box(box: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x_min, y_min, x_max, y_max = box
    x_min = max(0, min(width - 1, x_min))
    y_min = max(0, min(height - 1, y_min))
    x_max = max(x_min + 1, min(width, x_max))
    y_max = max(y_min + 1, min(height, y_max))
    return x_min, y_min, x_max, y_max


def collect_hand_candidates(results, frame_shape: tuple[int, ...]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if not results.multi_hand_landmarks:
        return candidates

    height, width = frame_shape[:2]
    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x * width for lm in hand_landmarks.landmark]
        y_coords = [lm.y * height for lm in hand_landmarks.landmark]
        box = normalize_box(
            (
                int(min(x_coords)) - CAMERA_PADDING,
                int(min(y_coords)) - CAMERA_PADDING,
                int(max(x_coords)) + CAMERA_PADDING,
                int(max(y_coords)) + CAMERA_PADDING,
            ),
            frame_shape,
        )
        candidates.append(
            {
                "box": box,
                "landmarks": hand_landmarks,
                "area": box_area(box),
            }
        )

    candidates.sort(key=lambda item: (item["box"][0], item["box"][1]))
    return candidates


def extract_hand_boxes(results, frame_shape: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
    return [candidate["box"] for candidate in collect_hand_candidates(results, frame_shape)]


def box_area(box: tuple[int, int, int, int]) -> int:
    x_min, y_min, x_max, y_max = box
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def largest_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return max(boxes, key=box_area)


def extract_single_hand_features(
    results,
    frame_shape: tuple[int, ...],
) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
    candidates = collect_hand_candidates(results, frame_shape)
    if not candidates:
        return None, None

    target = max(candidates, key=lambda item: item["area"])
    target_box = target["box"]
    hand_landmarks = target["landmarks"]
    coords: list[float] = []
    for landmark in hand_landmarks.landmark:
        coords.extend([landmark.x, landmark.y])
    return np.expand_dims(np.array(coords, dtype=np.float32), axis=0), target_box


def normalize_twohand_features(features: list[float]) -> np.ndarray:
    landmarks = np.array(features, dtype=np.float32).reshape(-1, 2)
    center = np.mean(landmarks, axis=0)
    landmarks = landmarks - center
    max_value = float(np.max(np.linalg.norm(landmarks, axis=1)))
    if max_value > 0:
        landmarks = landmarks / max_value
    return landmarks.flatten()


def extract_twohand_feature_variants(
    results,
    frame_shape: tuple[int, ...],
) -> tuple[list[np.ndarray], Optional[tuple[int, int, int, int]]]:
    candidates = collect_hand_candidates(results, frame_shape)
    if len(candidates) < 2:
        return [], None

    top_candidates = sorted(candidates, key=lambda item: item["area"], reverse=True)[:2]
    target_box = merge_boxes([item["box"] for item in top_candidates], frame_shape)
    base_features: list[list[float]] = []

    for candidate in top_candidates:
        coords: list[float] = []
        for landmark in candidate["landmarks"].landmark:
            coords.extend([landmark.x, landmark.y])
        base_features.append(coords)

    variants: list[np.ndarray] = []
    for ordered in (base_features, list(reversed(base_features))):
        raw = np.array(ordered[0] + ordered[1], dtype=np.float32)
        normalized = normalize_twohand_features(ordered[0] + ordered[1])
        variants.append(np.expand_dims(raw, axis=0))
        variants.append(np.expand_dims(normalized, axis=0))

    return variants, target_box


def merge_boxes(boxes: list[tuple[int, int, int, int]], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)
    return normalize_box((x_min, y_min, x_max, y_max), shape)


def expand_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, ...],
    factor: float,
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = (x_max - x_min) * factor
    height = (y_max - y_min) * factor
    expanded = (
        int(center_x - (width / 2)),
        int(center_y - (height / 2)),
        int(center_x + (width / 2)),
        int(center_y + (height / 2)),
    )
    return normalize_box(expanded, shape)


def crop_frame(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = box
    return frame[y_min:y_max, x_min:x_max]


def is_confident_camera_prediction(model_type: str, confidence: float, margin: float) -> bool:
    return (
        confidence >= CAMERA_CONFIDENCE_THRESHOLDS[model_type]
        and margin >= CAMERA_MARGIN_THRESHOLD
    )


def choose_camera_prediction(
    frame: np.ndarray,
    results,
    mode: str,
) -> tuple[str, float, str, Optional[tuple[int, int, int, int]]]:
    full_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = extract_hand_boxes(results, frame.shape)
    fallback_label, fallback_confidence, fallback_source = predict_full_frame_fallback(full_frame_rgb)
    alpha_fallback_label = fallback_label if fallback_label in string.ascii_uppercase else ""
    numeric_fallback_label = fallback_label if fallback_label in NUMBER_CLASS_SET else ""

    if not boxes:
        if mode == "numeric" and numeric_fallback_label:
            return numeric_fallback_label, fallback_confidence, fallback_source, None
        if mode != "numeric" and alpha_fallback_label:
            return alpha_fallback_label, fallback_confidence, fallback_source, None
        return "", fallback_confidence, fallback_source, None

    if mode == "numeric":
        features, active_box = extract_single_hand_features(results, frame.shape)
        if features is None:
            if numeric_fallback_label:
                return numeric_fallback_label, fallback_confidence, fallback_source, largest_box(boxes)
            return "", 0.0, "number-landmark", largest_box(boxes)
        label, confidence, margin = predict_live_features(features, "number")
        if label in NUMBER_CLASS_SET and is_confident_camera_prediction("number", confidence, margin):
            if should_prefer_full_frame_fallback(
                label,
                numeric_fallback_label,
                fallback_confidence,
            ):
                return numeric_fallback_label, fallback_confidence, fallback_source, active_box
            return label, confidence, "number-landmark", active_box
        if numeric_fallback_label:
            return numeric_fallback_label, fallback_confidence, fallback_source, active_box
        return "", confidence, "number-landmark", active_box

    feature_variants, active_box = extract_twohand_feature_variants(results, frame.shape)
    if feature_variants:
        best_prediction = max(
            (predict_live_features(features, "twohand") for features in feature_variants),
            key=lambda item: (item[1], item[2]),
        )
        label, confidence, margin = best_prediction
        if label in LIVE_TWOHAND_CLASS_SET and is_confident_camera_prediction("twohand", confidence, margin):
            if should_prefer_full_frame_fallback(
                label,
                alpha_fallback_label,
                fallback_confidence,
            ):
                return alpha_fallback_label, fallback_confidence, fallback_source, active_box
            return label, confidence, "twohand-landmark", active_box
        if alpha_fallback_label:
            return alpha_fallback_label, fallback_confidence, fallback_source, active_box
        return "", confidence, "twohand-landmark", active_box

    active_box = largest_box(boxes)
    if alpha_fallback_label in IMAGE_TWOHAND_CLASS_SET and fallback_confidence >= ROUTER_CONFIDENCE_THRESHOLD:
        return alpha_fallback_label, fallback_confidence, fallback_source, active_box

    features, active_box = extract_single_hand_features(results, frame.shape)
    if features is None:
        if alpha_fallback_label:
            return alpha_fallback_label, fallback_confidence, fallback_source, active_box
        return "", 0.0, "onehand-landmark", active_box

    label, confidence, margin = predict_live_features(features, "onehand")
    if label in LIVE_ONEHAND_CLASS_SET and is_confident_camera_prediction("onehand", confidence, margin):
        if should_prefer_full_frame_fallback(
            label,
            alpha_fallback_label,
            fallback_confidence,
        ):
            return alpha_fallback_label, fallback_confidence, fallback_source, active_box
        return label, confidence, "onehand-landmark", active_box

    if alpha_fallback_label:
        return alpha_fallback_label, fallback_confidence, fallback_source, active_box
    return "", confidence, "onehand-landmark", active_box


def reset_pending_prediction() -> None:
    pending_prediction["label"] = ""
    pending_prediction["start_time"] = 0.0


def update_live_prediction_state(
    label: str,
    confidence: float,
    active_model: str,
    mode: str,
) -> dict[str, object]:
    sign_state["mode"] = mode
    sign_state["active_model"] = active_model
    sign_state["active_confidence"] = round(confidence, 4)

    accepted_prediction = ""
    if label:
        now = time.time()
        if label != pending_prediction["label"]:
            pending_prediction["label"] = label
            pending_prediction["start_time"] = now
        elif now - pending_prediction["start_time"] >= CONFIRMATION_DELAY:
            accepted_prediction = label
            if sign_state["mode"] == "alphabet":
                sign_state["current_word"] += label.lower()
            else:
                sign_state["current_word"] += label
            sign_state["current_word"] = sign_state["current_word"][-30:]
            reset_pending_prediction()
    else:
        reset_pending_prediction()

    return {
        "prediction": label,
        "accepted_prediction": accepted_prediction,
        "current_word": sign_state["current_word"],
        "mode": sign_state["mode"],
        "active_model": sign_state["active_model"],
        "confidence": sign_state["active_confidence"],
        "pending_prediction": pending_prediction["label"],
    }


def predict_camera_frame(frame: np.ndarray, mode: str) -> dict[str, object]:
    require_models_ready()
    normalized_mode = mode if mode in {"alphabet", "numeric"} else sign_state["mode"]
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with LIVE_HANDS_LOCK:
        results = LIVE_HANDS.process(rgb_frame)

    label, confidence, active_model, _ = choose_camera_prediction(
        frame,
        results,
        normalized_mode,
    )
    return update_live_prediction_state(label, confidence, active_model, normalized_mode)


def gen_sign_to_text() -> Iterator[bytes]:
    require_models_ready()
    capture = get_video_capture()
    if capture is None:
        raise HTTPException(status_code=503, detail="Camera not accessible.")

    try:
        while sign_state.get("running", True):
            ret, frame = capture.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with LIVE_HANDS_LOCK:
                results = LIVE_HANDS.process(rgb_frame)

            label, confidence, active_model, active_box = choose_camera_prediction(
                frame,
                results,
                sign_state["mode"],
            )
            update_live_prediction_state(label, confidence, active_model, sign_state["mode"])

            if active_box is not None:
                x_min, y_min, x_max, y_max = active_box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (214, 167, 167), 2)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

            cv2.putText(
                frame,
                f"Mode: {sign_state['mode']}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Model: {active_model or 'waiting'}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (214, 167, 167),
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (176, 106, 115),
                2,
            )
            cv2.putText(
                frame,
                f"Text: {sign_state['current_word']}",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    finally:
        release_video_capture()


def build_text_to_sign_image(text: str) -> Optional[np.ndarray]:
    words = text.strip().split()
    line_images = []
    max_width = 0

    for word in words:
        images = []
        for char in word:
            if char.isalpha() or char.isdigit():
                img_path = alphabet_dict.get(char.upper())
                if img_path and os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    if image is not None:
                        images.append(image)
        if images:
            word_image = np.hstack(images)
            line_images.append(word_image)
            max_width = max(max_width, word_image.shape[1])

    if not line_images:
        return None

    padded_images = []
    for line_image in line_images:
        padded = np.zeros((line_image.shape[0], max_width, 3), dtype=np.uint8)
        padded[:, : line_image.shape[1], :] = line_image
        padded_images.append(padded)

    return np.vstack(padded_images)


def resolve_requested_model_type(model_type: str, filename: Optional[str] = None) -> str:
    if model_type in VALID_MODEL_TYPES:
        return model_type

    if model_type != "auto":
        raise HTTPException(
            status_code=400,
            detail="Invalid model_type. Use: number, onehand, twohand",
        )

    if filename:
        stem = Path(filename).stem.strip().upper()
        if stem in NUMBER_CLASS_SET:
            return "number"
        if stem in ONEHAND_CLASS_SET:
            return "onehand"
        if stem in TWOHAND_CLASS_SET:
            return "twohand"

    return "twohand"


async def predict_uploaded_file(
    file: Optional[UploadFile],
    model_type: str = "auto",
) -> dict[str, object]:
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    requested_model_type = resolve_requested_model_type(model_type, file.filename)

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    image_array = preprocess_image_bytes(file_bytes, requested_model_type)
    label, confidence, _ = predict_batch(image_array, requested_model_type)
    return {
        "prediction": label,
        "predicted_class": label,
        "confidence": round(confidence, 4),
        "model_type": requested_model_type,
    }


def resolve_frontend_file(path_name: str) -> Optional[Path]:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if not index_path.exists():
        return None

    target = (FRONTEND_DIST_DIR / path_name).resolve()
    frontend_root = FRONTEND_DIST_DIR.resolve()
    try:
        target.relative_to(frontend_root)
    except ValueError:
        return None

    if target.exists() and target.is_file():
        return target
    return None


def serve_frontend_index() -> Response:
    index_path = resolve_frontend_file("index.html")
    if index_path is not None:
        return FileResponse(index_path)
    return HTMLResponse(content="Backend running. Use frontend on localhost:5173")


@app.api_route("/", methods=["GET", "HEAD"])
def index() -> Response:
    return serve_frontend_index()


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    start_runtime_asset_load()

    if MODEL_LOAD_ERROR:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "detail": MODEL_LOAD_ERROR,
                "loaded_models": sorted(MODELS.keys()),
                "mode": sign_state["mode"],
            },
        )

    if not MODELS_READY:
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "loaded_models": sorted(MODELS.keys()),
                "mode": sign_state["mode"],
            },
        )

    return {
        "status": "ok",
        "loaded_models": sorted(MODELS.keys()),
        "mode": sign_state["mode"],
    }


@app.get("/video_feed")
def video_feed():
    require_models_ready()
    if get_video_capture() is None:
        raise HTTPException(status_code=503, detail="Camera not accessible.")
    sign_state["running"] = True
    return StreamingResponse(
        gen_sign_to_text(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/stop-camera")
def stop_camera():
    sign_state["running"] = False
    sign_state["active_model"] = ""
    sign_state["active_confidence"] = 0.0
    reset_pending_prediction()
    release_video_capture()
    return {"status": "stopped"}


@app.get("/set-mode/{mode}")
def set_mode(mode: str):
    if mode not in {"alphabet", "numeric"}:
        return JSONResponse(status_code=400, content={"error": "invalid mode"})
    sign_state["mode"] = mode
    sign_state["active_model"] = ""
    sign_state["active_confidence"] = 0.0
    reset_pending_prediction()
    return {"status": "ok", "mode": mode}


@app.post("/predict-live-frame")
async def predict_live_frame_route(
    file: Optional[UploadFile] = File(default=None),
    mode: str = Form(default="alphabet"),
):
    require_models_ready()
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    if mode not in {"alphabet", "numeric"}:
        raise HTTPException(status_code=400, detail="invalid mode")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    rgb_image = load_rgb_image_from_bytes(file_bytes)
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return predict_camera_frame(frame, mode)


@app.get("/last-prediction/sign")
def last_sign_prediction():
    return {"prediction": sign_state.get("current_word", "")}


@app.post("/last-prediction/sign")
def set_last_sign_prediction(payload: PredictionUpdate):
    sign_state["current_word"] = payload.text[-30:]
    reset_pending_prediction()
    return {"status": "updated", "prediction": sign_state["current_word"]}


@app.delete("/last-prediction/sign")
def clear_last_sign_prediction():
    sign_state["current_word"] = ""
    reset_pending_prediction()
    return {"status": "cleared"}


@app.delete("/last-prediction/sign/last")
def remove_last_sign_prediction():
    sign_state["current_word"] = sign_state.get("current_word", "")[:-1]
    reset_pending_prediction()
    return {"status": "updated", "prediction": sign_state["current_word"]}


@app.get("/suggestions")
def suggestions_route(prefix: str = Query(default="")):
    return {"suggestions": suggest_words(prefix)}


@app.post("/predict-image")
async def predict_image_route(
    file: Optional[UploadFile] = File(default=None),
    model_type: str = Form(default="auto"),
):
    require_models_ready()
    return await predict_uploaded_file(file, model_type)


@app.post("/predict")
async def predict_route(
    file: Optional[UploadFile] = File(default=None),
    model_type: str = Form(default="auto"),
):
    require_models_ready()
    return await predict_uploaded_file(file, model_type)


@app.get("/text-to-sign/{text:path}")
def text_to_sign_feed(text: str):
    image = build_text_to_sign_image(text)
    if image is None:
        raise HTTPException(status_code=400, detail="No valid characters")
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
def frontend_routes(full_path: str) -> Response:
    frontend_file = resolve_frontend_file(full_path)
    if frontend_file is not None:
        return FileResponse(frontend_file)
    return serve_frontend_index()


if __name__ == "__main__":
    import uvicorn

    print("Starting ISL Backend on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
