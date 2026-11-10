"""REST API serving the champion MLflow model.

Run locally from the project root:
    uvicorn app.main:app --reload --port 8000
Then open http://127.0.0.1:8000 (form) or http://127.0.0.1:8000/docs (Swagger UI).
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR.parent / "model"))
FLOAT_COLUMNS = ["Cholesterol", "RestingBP", "Age", "FastingBS", "MaxHR", "Oldpeak"]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    state["model"] = mlflow.sklearn.load_model(str(MODEL_DIR))
    info_file = MODEL_DIR / "model_info.json"
    state["info"] = json.loads(info_file.read_text()) if info_file.exists() else {}
    yield
    state.clear()


app = FastAPI(
    title="Heart disease prediction API",
    description="Serves the MLflow-registered champion model (sklearn pipeline: raw features in, prediction out).",
    version="1.0.0",
    lifespan=lifespan,
)


class Patient(BaseModel):
    """Input schema. Pydantic rejects invalid values with a clear 422 error before they reach the model."""
    Age: int = Field(..., ge=1, le=120, description="Age in years")
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"] = Field(..., description="TA typical angina, ATA atypical angina, NAP non-anginal pain, ASY asymptomatic")
    RestingBP: float = Field(..., ge=0, le=250, description="Resting blood pressure (mm Hg); 0 = unknown")
    Cholesterol: float = Field(..., ge=0, le=700, description="Serum cholesterol (mg/dl); 0 = unknown")
    FastingBS: Literal[0, 1] = Field(..., description="1 if fasting blood sugar > 120 mg/dl")
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(..., ge=50, le=230, description="Maximum heart rate achieved")
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(..., ge=-5, le=10, description="ST depression")
    ST_Slope: Literal["Up", "Flat", "Down"]

    model_config = {"json_schema_extra": {"examples": [{
        "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 239,
        "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 160, "ExerciseAngina": "N",
        "Oldpeak": 1.2, "ST_Slope": "Flat"}]}}


class Prediction(BaseModel):
    prediction: int
    label: str
    probability: float
    model_version: int | None = None


@app.get("/", include_in_schema=False)
def form():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/info")
def info():
    """Which registered model version is being served, and how it scored."""
    return state.get("info", {})


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = pd.DataFrame([patient.model_dump()]).astype({c: "float64" for c in FLOAT_COLUMNS})
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return Prediction(
        prediction=pred,
        label="Heart disease likely" if pred else "Heart disease unlikely",
        probability=round(proba, 4),
        model_version=state.get("info", {}).get("version"),
    )
