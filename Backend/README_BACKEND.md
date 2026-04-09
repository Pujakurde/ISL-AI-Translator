# ISL Translator Backend - FastAPI

A clean, beginner-friendly FastAPI backend for Indian Sign Language (ISL) recognition using pre-trained CNN models.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 📁 Project Structure

```
Backend/
├── models/                          # Directory for model files
│   ├── isl_number_model.h5         # Model for number recognition (0-9)
│   ├── isl_onehand_model.h5        # Model for one-hand alphabet recognition
│   ├── isl_twohand_model.h5        # Model for two-hand alphabet recognition
│   ├── isl_number_labels.json      # Labels for number model
│   ├── isl_onehand_labels.json     # Labels for one-hand model
│   └── isl_twohand_labels.json     # Labels for two-hand model
├── main_new.py                      # Main FastAPI application
├── requirements.txt                 # Python dependencies
└── README_BACKEND.md               # This file
```

## 🚀 Installation Steps

### 1. Navigate to the Backend Directory

```bash
cd d:\Folder D\ISL\Backend
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```bash
pip install fastapi uvicorn[standard] python-multipart tensorflow numpy pillow
```

### 4. Verify Model Files

Make sure the following files exist in the `models/` directory:
- `isl_number_model.h5`
- `isl_onehand_model.h5`
- `isl_twohand_model.h5`
- `isl_number_labels.json`
- `isl_onehand_labels.json`
- `isl_twohand_labels.json`

## 🏃 Running the Backend

### Start the FastAPI Server

```bash
uvicorn main_new:app --reload
```

This will start the server at:
- **Local**: http://127.0.0.1:8000
- **Network**: http://0.0.0.0:8000

### Alternative: Run with Python

```bash
python main_new.py
```

## 🧪 Testing the API

### 1. Interactive API Documentation

Open your browser and navigate to:
```
http://127.0.0.1:8000/docs
```

This provides an interactive Swagger UI where you can:
- View all available endpoints
- Test the API directly from your browser
- See request/response formats

### 2. Alternative Documentation (ReDoc)

```
http://127.0.0.1:8000/redoc
```

### 3. Testing the Prediction Endpoint

Using the Swagger UI at `/docs`:

1. Click on the `/predict` endpoint
2. Click "Try it out"
3. Fill in the parameters:
   - **file**: Upload an image file
   - **model_type**: Enter "number", "onehand", or "twohand"
4. Click "Execute"
5. View the response

### 4. Testing with curl

```bash
# Number prediction
curl -X POST "http://127.0.0.1:8000/predict"   -F "file=@path/to/your/image.jpg"   -F "model_type=number"

# One-hand prediction
curl -X POST "http://127.0.0.1:8000/predict"   -F "file=@path/to/your/image.jpg"   -F "model_type=onehand"

# Two-hand prediction
curl -X POST "http://127.0.0.1:8000/predict"   -F "file=@path/to/your/image.jpg"   -F "model_type=twohand"
```

## 📡 API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Health Check
```
GET /health
```
Returns the health status and loaded models.

### 3. Prediction Endpoint
```
POST /predict
```
**Parameters:**
- `file` (UploadFile, required): Image file to predict
- `model_type` (str, required): Type of model to use ("number", "onehand", or "twohand")

**Response:**
```json
{
  "prediction": "A",
  "confidence": 0.9532,
  "model_type": "onehand"
}
```

## 🔧 Configuration

### CORS Settings

The backend is configured to accept requests from any origin. In production, update the CORS middleware in `main_new.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Port Configuration

To run on a different port:

```bash
uvicorn main_new:app --reload --port 8080
```

## 🐛 Troubleshooting

### Model Not Found Error
- Ensure all `.h5` model files are in the `models/` directory
- Check that file names match exactly (case-sensitive)

### Import Errors
- Make sure you've activated your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

### TensorFlow/GPU Issues
- If you don't have a GPU, TensorFlow will use CPU (this is fine)
- For GPU support, install CUDA and cuDNN according to TensorFlow requirements

### Port Already in Use
- Change the port: `uvicorn main_new:app --reload --port 8080`
- Or stop the process using port 8000

## 📝 Example Response

Successful prediction:
```json
{
  "prediction": "5",
  "confidence": 0.9876,
  "model_type": "number"
}
```

Error response:
```json
{
  "detail": "Invalid model_type. Must be one of: number, onehand, twohand"
}
```

## 🔐 Security Notes for Production

1. **CORS**: Restrict `allow_origins` to your frontend domain only
2. **File Size**: Add file size limits to prevent large uploads
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **HTTPS**: Use HTTPS in production
5. **Environment Variables**: Store sensitive data in environment variables

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Keras Documentation](https://www.tensorflow.org/guide/keras)
- [Uvicorn Documentation](https://www.uvicorn.org/)

## 🤝 Connecting with React Frontend

Your React frontend can make requests like this:

```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('model_type', 'onehand');

const response = await fetch('http://127.0.0.1:8000/predict', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(data.prediction, data.confidence);
```

## 📄 License

This project is part of the ISL Translator system.
