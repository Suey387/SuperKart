from fastapi import FastAPI, HTTPException, Header
from typing import Optional
import os

app = FastAPI(title="SuperKart model API")

# Path to your model file inside the repo or an absolute path in the container
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pt")
# API key must be set in the environment; server will refuse to start otherwise
API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    # Fail early so the operator knows to set a secret and avoid accidentally exposing the model
    raise RuntimeError("API_KEY environment variable must be set to a non-empty secret value before starting the server")

model = None


def load_model():
    global model
    try:
        # Example PyTorch loader — replace with your real loader if using TF or HF
        import torch
        model = torch.load(MODEL_PATH, map_location="cpu")
        model.eval()
        print("Model loaded from", MODEL_PATH)
    except Exception as e:
        # Keep model as None if loading fails; startup still succeeds so you can inspect logs
        print("Model load failed (this is OK if you haven't added a model file yet):", e)
        model = None


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/predict")
async def predict(q: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    # Require API key on every request
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded; check server logs or MODEL_PATH")

    try:
        # Replace the following with your model's inference code
        prediction = {"message": f"Received input of length {len(q or '')}"}
        return {"input": q, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
