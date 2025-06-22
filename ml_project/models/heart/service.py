from fastapi import APIRouter, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
import joblib

router = APIRouter()
API_KEY = "heart123"
model = joblib.load("models/heart/heart_disease_model.joblib")

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@router.post("/predict")
def predict_heart(data: HeartInput):
    # if token != API_KEY:
    #     raise HTTPException(status_code=403, detail="Invalid Token")
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    if int(pred) == 1:
        result = 'heart disease detected'
    else:
        result = 'no heart disease detected'
    return {
        "prediction_index": int(pred),
        "prediction_label":result
        }
