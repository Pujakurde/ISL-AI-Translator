# Backend Compatibility Shim

This directory exists so older Render services configured with `Root Directory = Backend`
can still deploy after the backend was renamed to `backend1/`.

For new deployments, prefer the repo root `render.yaml`, which points Render to `backend1/`.
