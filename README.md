# ISL AI Translator

This repo is set up to run as two separate Render services:

- `isl-ai-translator-1`: a static site built from [`ISL-Converter`](./ISL-Converter)
- `isl-backend`: a FastAPI web service built from [`backend1`](./backend1)

## Deploy On Render

The recommended path is to deploy with the repo root [`render.yaml`](./render.yaml).

1. Push the repository to GitHub.
2. In the Render dashboard, choose `New` -> `Blueprint`.
3. Select this repository and branch.
4. Review the two services Render detects from `render.yaml`.
5. Create the Blueprint and wait for both deploys to finish.

The Blueprint wires the services together automatically:

- `VITE_API_BASE` on the frontend is set from the backend service's `RENDER_EXTERNAL_URL`
- `ALLOWED_ORIGINS` on the backend is set from the frontend service's `RENDER_EXTERNAL_URL`

If you already have services on Render, keep the service names in `render.yaml` aligned with those existing services. Render uses the `name` field to decide whether it should update an existing service or create a new one.

## Render Service Settings

### Frontend

- Service type: `Static Site`
- Root directory: `ISL-Converter`
- Build command: `npm ci && npm run build`
- Publish directory: `dist`

### Backend

- Service type: `Web Service`
- Runtime: `Python`
- Root directory: `backend1`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

## Local Development

- Frontend: `cd ISL-Converter && npm install && npm run dev`
- Backend: `cd backend1 && pip install -r requirements.txt && uvicorn main:app --reload`

The backend keeps local Vite origins enabled by default, and the frontend can target the backend with `VITE_API_BASE` when needed.
