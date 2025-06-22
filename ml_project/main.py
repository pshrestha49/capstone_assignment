from fastapi import FastAPI
from models.heart.service import router as heart_router
from models.cifar10.service import router as cifar_router

app = FastAPI(title="Unified ML Inference API")

app.include_router(heart_router, prefix="/heart")
app.include_router(cifar_router, prefix="/cifar10")

@app.get("/")
def root():
    return {"status": "Unified API is running"}
