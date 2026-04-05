# app.py
# Experiment 4: FastAPI-based ML Inference API
# Offline: uvicorn app:app --reload  → http://127.0.0.1:8000
# Online:  Deployed to Azure App Service via GitHub Actions

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import json
import os

# ── Base directory — always relative to this file ─────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH   = os.path.join(BASE_DIR, "models", "metadata.json")
UI_PATH     = os.path.join(BASE_DIR, "ui.html")

app = FastAPI(
    title="Placement Prediction API",
    description="ML inference API for student placement prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = [
    "CGPA", "Internships", "Projects",
    "AptitudeTestScore", "SoftSkillsRating",
    "SSC_Marks", "HSC_Marks"
]

# ── Safe lazy loader ───────────────────────────────────────────────────────
_model    = None
_metadata = {}

def load_model():
    global _model, _metadata
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run: python train.py first."
            )
        _model = joblib.load(MODEL_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                _metadata = json.load(f)
    return _model

# Load metadata at startup so /model/info works immediately
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        _metadata = json.load(f)

# ── Schemas ────────────────────────────────────────────────────────────────
class Student(BaseModel):
    CGPA:              float = Field(..., ge=0, le=10,  example=8.5)
    Internships:       int   = Field(..., ge=0, le=10,  example=2)
    Projects:          int   = Field(..., ge=0, le=20,  example=3)
    AptitudeTestScore: int   = Field(..., ge=0, le=100, example=80)
    SoftSkillsRating:  float = Field(..., ge=0, le=5,   example=4.2)
    SSC_Marks:         int   = Field(..., ge=0, le=100, example=78)
    HSC_Marks:         int   = Field(..., ge=0, le=100, example=75)

class BatchRequest(BaseModel):
    students: list[Student]

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/robots933456.txt", include_in_schema=False)
def robots():
    return {"status": "healthy"}
@app.get("/", include_in_schema=False)
def serve_ui():
    if os.path.exists(UI_PATH):
        return FileResponse(UI_PATH)
    return {"message": "Placement Prediction API running", "docs": "/docs"}

@app.get("/health", tags=["System"])
def health():
    return {
        "status":      "healthy",
        "model_ready": os.path.exists(MODEL_PATH),
        "version":     "1.0.0"
    }

@app.get("/model/info", tags=["Model"])
def model_info():
    if not os.path.exists(META_PATH):
        return {"status": "not_trained", "message": "Run train.py first"}
    with open(META_PATH) as f:
        meta = json.load(f)
    return {"status": "loaded", **meta}

@app.post("/predict", tags=["Inference"])
def predict(student: Student):
    model = load_model()
    X = pd.DataFrame([[
        student.CGPA, student.Internships, student.Projects,
        student.AptitudeTestScore, student.SoftSkillsRating,
        student.SSC_Marks, student.HSC_Marks,
    ]], columns=FEATURES)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {
        "placement_status":   "Placed" if pred == 1 else "Not Placed",
        "probability_placed": round(float(proba), 4),
        "confidence":         "High" if proba > 0.75 or proba < 0.25 else "Medium",
        "model_type":         _metadata.get("model_type", ""),
        "run_id":             _metadata.get("run_id", ""),
    }

@app.post("/predict/batch", tags=["Inference"])
def predict_batch(batch: BatchRequest):
    model = load_model()
    results = []
    for s in batch.students:
        X = pd.DataFrame([[
            s.CGPA, s.Internships, s.Projects,
            s.AptitudeTestScore, s.SoftSkillsRating,
            s.HSC_Marks, s.SSC_Marks,
        ]], columns=FEATURES)
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        results.append({
            "placement_status":   "Placed" if pred == 1 else "Not Placed",
            "probability_placed": round(float(proba), 4),
        })
    return {"count": len(results), "predictions": results}