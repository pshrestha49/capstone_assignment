# capstone_assignment

# README.md

## 📦 Unified ML Inference API: Heart Disease & CIFAR-10

This project is a unified FastAPI-based ML inference server that serves predictions for:
- **Heart Disease Prediction** (tabular clinical data)
- **CIFAR-10 Image Classification** (RGB 32x32 image data)

Both models are exposed via RESTful API endpoints. The project is containerized using Docker for ease of deployment.

---

## 🔧 Project Structure

```
ml_project/
├── main.py                     # FastAPI entry point
├── models/
│   ├── heart/
│   │   ├── arch.py            # Model architecture
│   │   ├── service.py         # FastAPI route logic
│   │   └── heart_model.pth    # Trained model
│   └── cifar10/
│       ├── arch.py            # CNN architecture
│       ├── service.py         # FastAPI route logic
│       └── cifar10_cnn.pth    # Trained model
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd ml_project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the FastAPI server
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

Now access it at: [http://localhost:8080](http://localhost:8080)

---

## 🚀 Docker Deployment

### 1. Build the Docker image
```bash
docker build -t unified-ml-api .
```

### 2. Run the container
```bash
docker run -d -p 8080:8080 unified-ml-api
```

---

## 📡 API Endpoints

### Health Check
```
GET /
```
Returns:
```json
{"status": "Unified API is running"}
```

### Heart Disease Prediction
```
POST /heart/predict
```
**Payload:**
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```
**Response:**
```json
{
  "prediction": 1,
  "interpreted": "Patient is likely to have heart disease"
}
```

### CIFAR-10 Image Classification
```
POST /cifar10/predict
```
**Payload:** `multipart/form-data`
- Upload a `.jpg` or `.png` image

**Response:**
```json
{
  "prediction": 3,
  "interpreted": "Cat"
}
```

---

## 🧪 Example CURL Requests

### Heart Disease
```bash
curl -X POST http://localhost:8080/heart/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 60, "sex": 1, "cp": 2, "trestbps": 140, "chol": 289, "fbs": 0, "restecg": 1, "thalach": 172, "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2}'
```

### CIFAR-10
```bash
curl -X POST http://localhost:8080/cifar10/predict \
  -F file=@sample_cat.jpg
```

---

## 📋 Notes
- Ensure the `.pth` model files are in the correct paths as referenced in `service.py` files.
- All endpoints are open (no token/auth needed).
- The project works best in environments with `torch`, `fastapi`, `uvicorn`, `pillow`, `scikit-learn`, etc.

---
