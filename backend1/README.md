# Backend1

FastAPI backend for the ISL translator project.

## Run locally

```bash
cd backend1
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API starts on `http://127.0.0.1:8000`.

## Important files

- `main.py` - main FastAPI app used by deployment
- `main_new.py` - small runner that imports `main:app`
- `models/` - image and live camera model files
- `dataset/Indian/*/4.jpg` and `4.png` - sample sign images used by the UI

## Notes

- Keep generated folders like `extract_landmarks/`, `numerical_landmarks/`, and `onehand_landmarks/` out of Git.
- Deployment is configured from the repo root `render.yaml`.
