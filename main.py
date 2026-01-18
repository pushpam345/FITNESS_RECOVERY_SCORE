import pandas as pd
import joblib
from fastapi import FastAPI


from pydantic import BaseModel, Field


class RecoveryMetrics(BaseModel):
    sleep_hours: float = Field(..., ge=2, le=9, description="Hours of sleep")
    sleep_quality: int = Field(..., ge=1, le=5)
    fatigue_level: int = Field(..., ge=1, le=5)
    muscle_soreness: int = Field(..., ge=1, le=5)
    prev_day_intensity: float = Field(..., ge=0, le=1)
    workout_streak: int = Field(..., ge=0, le=21)
    resting_hr: int = Field(..., ge=45, le=100,
                            description="Resting heart rate")


app = FastAPI()
model = joblib.load("fit_model.pkl")


@app.post("/predict")
async def predict_recovery(data: RecoveryMetrics):

    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)

    return {"predicted_recovery_score": float(prediction[0])}
