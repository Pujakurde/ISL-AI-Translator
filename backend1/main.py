from __future__ import annotations

import os
import string
import tempfile
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
MODEL_DIR = API_DIR / "models"
DATASET_DIR = API_DIR / "dataset"
INDIAN_DATASET_DIR = DATASET_DIR / "Indian"
TEMP_DIR = PROJECT_ROOT / "temp"


class PredictionUpdate(BaseModel):
    text: str = ""


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "None of the expected paths exist: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


app = FastAPI(title="ISL Translator Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_model = load_model(
    str(
        resolve_existing_path(
            API_DIR / "image_hand_gesture_img_model.h5",
            MODEL_DIR / "image_hand_gesture_img_model.h5",
            MODEL_DIR / "image_hand_gesture_model.h5",
        )
    )
)

class_indices = np.load(
    str(resolve_existing_path(DATASET_DIR / "class_mapping.npy")),
    allow_pickle=True,
).item()
class_mapping = {value: key for key, value in class_indices.items()}

cap: Optional[cv2.VideoCapture] = None
sign_state = {"current_word": "", "mode": "alphabet", "running": True}
pending_prediction = {"label": "", "start_time": 0.0}
CONFIRMATION_DELAY = 1.2

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
        word_list = [w.lower() for w in words.words()]
    except Exception:
        nltk.download("words", quiet=True)
        word_list = [w.lower() for w in words.words()]
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
            print(f"✓ Camera opened: index {index}, backend {backend}")
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
    print("✗ Error: Camera not accessible on any index 0-4")
    return None


def release_video_capture() -> None:
    global cap
    if cap is not None:
        cap.release()
        cap = None


def gen_sign_to_text() -> Iterator[bytes]:
    capture = get_video_capture()
    if capture is None:
        raise HTTPException(status_code=503, detail="Camera not accessible.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=2,
    )

    try:
        while sign_state.get("running", True):
            ret, frame = capture.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            label = ""
            try:
                if results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))

                        padding = 20
                        x_min = max(0, x_min - padding)
                        x_max = min(w, x_max + padding)
                        y_min = max(0, y_min - padding)
                        y_max = min(h, y_max + padding)

                        hand_crop = frame[y_min:y_max, x_min:x_max]
                        if hand_crop.size == 0:
                            continue

                        hand_crop = cv2.resize(hand_crop, (64, 64))
                        hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                        img_array = img_to_array(hand_crop) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        pred = image_model.predict(img_array, verbose=0)
                        idx = int(np.argmax(pred))
                        confidence = float(pred[0][idx])
                        predicted_class = class_mapping.get(idx, "")

                        if confidence > 0.35:
                            if sign_state["mode"] == "alphabet" and predicted_class and predicted_class[0].isalpha():
                                label = predicted_class[0].upper()
                                break
                            if sign_state["mode"] == "numeric" and predicted_class and predicted_class.isdigit():
                                label = predicted_class
                                break
            except Exception as ex:
                print(f"Prediction error: {ex}")
                label = ""

            if label:
                now = time.time()
                if label != pending_prediction["label"]:
                    pending_prediction["label"] = label
                    pending_prediction["start_time"] = now
                if now - pending_prediction["start_time"] >= CONFIRMATION_DELAY:
                    if sign_state["mode"] == "alphabet":
                        sign_state["current_word"] += label[0].lower()
                    else:
                        sign_state["current_word"] += label
                    sign_state["current_word"] = sign_state["current_word"][-30:]
                    pending_prediction["label"] = ""
                    pending_prediction["start_time"] = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

            cv2.putText(
                frame,
                f"Mode: {sign_state['mode']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
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

            ret2, buffer = cv2.imencode(".jpg", frame)
            if not ret2:
                continue

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    finally:
        hands.close()
        release_video_capture()


def predict_image(img_path: Path, target_size: tuple[int, int] = (64, 64)) -> tuple[str, float]:
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = image_model.predict(img_array, verbose=0)
    idx = int(np.argmax(predictions[0]))
    return class_mapping.get(idx, "Unknown"), float(predictions[0][idx])


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
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
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


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(content="Backend running. Use frontend on localhost:5173")


@app.get("/video_feed")
def video_feed():
    if get_video_capture() is None:
        raise HTTPException(status_code=503, detail="Camera not accessible.")
    sign_state["running"] = True
    return StreamingResponse(gen_sign_to_text(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/stop-camera")
def stop_camera():
    sign_state["running"] = False
    pending_prediction["label"] = ""
    pending_prediction["start_time"] = 0.0
    release_video_capture()
    return {"status": "stopped"}


@app.get("/set-mode/{mode}")
def set_mode(mode: str):
    if mode not in {"alphabet", "numeric"}:
        return JSONResponse(status_code=400, content={"error": "invalid mode"})
    sign_state["mode"] = mode
    pending_prediction["label"] = ""
    pending_prediction["start_time"] = 0.0
    return {"status": "ok", "mode": mode}


@app.get("/last-prediction/sign")
def last_sign_prediction():
    return {"prediction": sign_state.get("current_word", "")}


@app.post("/last-prediction/sign")
def set_last_sign_prediction(payload: PredictionUpdate):
    sign_state["current_word"] = payload.text[-30:]
    pending_prediction["label"] = ""
    pending_prediction["start_time"] = 0.0
    return {"status": "updated", "prediction": sign_state["current_word"]}


@app.delete("/last-prediction/sign")
def clear_last_sign_prediction():
    sign_state["current_word"] = ""
    pending_prediction["label"] = ""
    pending_prediction["start_time"] = 0.0
    return {"status": "cleared"}


@app.delete("/last-prediction/sign/last")
def remove_last_sign_prediction():
    sign_state["current_word"] = sign_state.get("current_word", "")[:-1]
    pending_prediction["label"] = ""
    pending_prediction["start_time"] = 0.0
    return {"status": "updated", "prediction": sign_state["current_word"]}


@app.get("/suggestions")
def suggestions_route(prefix: str = Query(default="")):
    return {"suggestions": suggest_words(prefix)}


async def predict_uploaded_file(file: Optional[UploadFile]) -> dict[str, object]:
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    TEMP_DIR.mkdir(exist_ok=True)
    suffix = Path(file.filename).suffix or ".jpg"
    temp_path: Optional[Path] = None
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file")
        with tempfile.NamedTemporaryFile(delete=False, dir=TEMP_DIR, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = Path(temp_file.name)
        predicted_class, confidence = predict_image(temp_path)
        return {
            "prediction": predicted_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
        }
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


@app.post("/predict-image")
async def predict_image_route(file: Optional[UploadFile] = File(default=None)):
    return await predict_uploaded_file(file)


@app.post("/predict")
async def predict_route(
    file: Optional[UploadFile] = File(default=None),
    model_type: str = Form(default="auto"),
):
    result = await predict_uploaded_file(file)
    result["model_type"] = model_type
    return result


@app.get("/text-to-sign/{text:path}")
def text_to_sign_feed(text: str):
    image = build_text_to_sign_image(text)
    if image is None:
        raise HTTPException(status_code=400, detail="No valid characters")
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    print("Starting ISL Backend on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
