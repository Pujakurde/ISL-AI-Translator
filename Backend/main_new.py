# ISL Translator - FastAPI Backend
# This backend serves pre-trained CNN models for Indian Sign Language recognition
# It accepts images and returns predictions for numbers, one-hand, and two-hand signs

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import io
import os

# Initialize FastAPI app
app = FastAPI(
    title="ISL Translator API",
    description="API for Indian Sign Language recognition using CNN models",
    version="1.0.0"
)

# ---- Mode and label rules (strict separation) ----
ONEHAND_CLASSES = {"C", "I", "L", "O", "U", "V"}
ALPHABETS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TWOHAND_CLASSES = ALPHABETS - ONEHAND_CLASSES
NUMBER_CLASSES = set("0123456789")

EXPECTED_LABELS = {
    "number": NUMBER_CLASSES,
    "onehand": ONEHAND_CLASSES,
    "twohand": TWOHAND_CLASSES,
}

# Selection dataset folders (frontend grid)
DATASET_DIR = Path("dataset/Indian")
SELECT_DIRS = {
    "number": DATASET_DIR / "numbers_select",
    "onehand": DATASET_DIR / "onehand_select",
    "twohand": DATASET_DIR / "twohand_select",
}

# CORS setup (production friendly)
def _get_cors_origins():
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]

_cors_origins = _get_cors_origins()
_allow_credentials = False if "*" in _cors_origins else True

# Add CORS middleware to allow frontend (React) connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,  # In production, set CORS_ORIGINS env var
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths to models and labels
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

# Global variables to store models and labels
models = {}
labels = {}

def load_labels(path: Path, expected_labels=None):
    """Load labels and return index->label mapping. Supports two JSON formats."""
    with open(path, "r") as f:
        data = json.load(f)

    if data and isinstance(list(data.values())[0], int):
        # {"A": 0} -> {0: "A"}
        label_set = set(data.keys())
        idx_to_label = {v: k for k, v in data.items()}
    else:
        # {"0": "A"} -> {0: "A"}
        label_set = set(data.values())
        idx_to_label = {int(k): v for k, v in data.items()}

    if expected_labels is not None and label_set != expected_labels:
        missing = expected_labels - label_set
        extra = label_set - expected_labels
        print(
            "[WARN] Label mismatch for",
            path,
            "missing:",
            sorted(missing),
            "extra:",
            sorted(extra),
        )
        print("[WARN] Consider retraining this model.")

    return idx_to_label

def load_model_and_labels(model_type: str):
    """Load a specific model and its corresponding labels"""
    try:
        # Load the Keras model
        models[model_type] = tf.keras.models.load_model(str(MODEL_PATHS[model_type]))
        print(f"✓ Loaded {model_type} model")

        # Load labels from JSON file
        labels[model_type] = load_labels(
            LABEL_PATHS[model_type],
            EXPECTED_LABELS.get(model_type),
        )
        print(f"✓ Loaded {model_type} labels")

        return True
    except Exception as e:
        print(f"✗ Error loading {model_type}: {str(e)}")
        return False

# Load all models and labels at startup
print("\n=== Loading ISL Models ===")
for model_type in MODEL_PATHS:
    load_model_and_labels(model_type)
print("=== Model Loading Complete ===\n")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess the uploaded image for model prediction
    Steps:
    1. Convert bytes to PIL Image
    2. Resize to 64x64 (model input size)
    3. Convert to RGB (if not already)
    4. Convert to numpy array and normalize to [0, 1]
    5. Add batch dimension
    """
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to 64x64 (model input size)
    image = image.resize((64, 64))

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Add batch dimension (model expects shape: [batch, height, width, channels])
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

def get_prediction(image_array: np.ndarray, model_type: str) -> tuple:
    """
    Run prediction on preprocessed image
    Returns: (predicted_label, confidence_score)
    """
    # Get the appropriate model and labels
    model = models[model_type]
    label_map = labels[model_type]

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    # Get the index with highest probability
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    # Convert index to label using the label map
    predicted_label = label_map.get(predicted_index, str(predicted_index))

    return predicted_label, confidence

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "ISL Translator API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/select/list")
async def list_select(mode: str):
    """Return available selection labels for the given mode"""
    if mode not in SELECT_DIRS:
        raise HTTPException(status_code=400, detail="Invalid mode. Use: number, onehand, twohand")
    folder = SELECT_DIRS[mode]
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Selection folder not found: {folder}")
    labels_list = sorted([p.stem for p in folder.iterdir() if p.is_file()])
    return {"mode": mode, "labels": labels_list}

@app.get("/select/image/{mode}/{label}")
async def get_select_image(mode: str, label: str):
    """Serve selection image for frontend grid"""
    if mode not in SELECT_DIRS:
        raise HTTPException(status_code=400, detail="Invalid mode. Use: number, onehand, twohand")
    folder = SELECT_DIRS[mode]
    image_path = folder / f"{label}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loaded_models = [k for k in models if models[k] is not None]
    return {
        "status": "healthy",
        "loaded_models": loaded_models,
        "total_models": len(MODEL_PATHS)
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file to predict"),
    model_type: str = Form(..., description="Model type: number, onehand, or twohand")
):
    """
    Predict ISL sign from uploaded image

    Parameters:
    - file: Image file (multipart/form-data)
    - model_type: Type of model to use (number/onehand/twohand)

    Returns:
    - prediction: Predicted sign label
    - confidence: Confidence score (0-1)
    """
    # Validate model_type
    if model_type not in MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type. Must be one of: {', '.join(MODEL_PATHS.keys())}"
        )

    # Check if model is loaded
    if model_type not in models or models[model_type] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_type}' is not available"
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Preprocess the image
        image_array = preprocess_image(image_bytes)

        # Get prediction
        prediction, confidence = get_prediction(image_array, model_type)

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "model_type": model_type
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
