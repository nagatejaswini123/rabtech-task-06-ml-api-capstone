from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Titanic ML Prediction API",
    description="FastAPI service for Titanic survival prediction",
    version="1.0.0"
)

model = joblib.load("Model/champion_model.joblib")


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@app.get("/")
def home():
    return {
        "message": "Titanic ML Prediction API is running"
    }


@app.post("/predict")
def predict(passenger: Passenger):

    data = pd.DataFrame([{
        "Pclass": passenger.Pclass,
        "Sex": passenger.Sex,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare,
        "Embarked": passenger.Embarked
    }])

    probability = float(
        model.predict_proba(data)[0][1]
    )

    prediction = int(
        model.predict(data)[0]
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }
