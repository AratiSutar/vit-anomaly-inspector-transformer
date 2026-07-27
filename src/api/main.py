import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from src.api.inference import InferenceEngine

app = FastAPI(title="ViT Anomaly Inspector")

engine = InferenceEngine(
    category='bottle',
    save_path='outputs'
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = engine.predict_from_bytes(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
