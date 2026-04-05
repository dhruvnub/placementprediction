# app.py (SIMPLE VERSION)

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model once
model = joblib.load("models/model.pkl")

FEATURES = [
    "CGPA", "Internships", "Projects",
    "AptitudeTestScore", "SoftSkillsRating",
    "SSC_Marks", "HSC_Marks"
]

# Input schema
class Student(BaseModel):
    CGPA: float
    Internships: int
    Projects: int
    AptitudeTestScore: int
    SoftSkillsRating: float
    SSC_Marks: int
    HSC_Marks: int

# Home route
@app.get("/")
def home():
    return {"message": "Placement Prediction API is running"}

# Prediction route
@app.post("/predict")
def predict(student: Student):
    data = pd.DataFrame([[ 
        student.CGPA,
        student.Internships,
        student.Projects,
        student.AptitudeTestScore,
        student.SoftSkillsRating,
        student.SSC_Marks,
        student.HSC_Marks
    ]], columns=FEATURES)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "placement": "Placed" if prediction == 1 else "Not Placed",
        "probability": round(float(probability), 4)
    }