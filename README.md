# Titanic ML Prediction API

## Project Overview

This project packages a trained Titanic survival classification model into a FastAPI REST API.

The model predicts whether a passenger is likely to survive based on passenger information.

## Machine Learning Model

The trained champion model was developed using a Scikit-Learn preprocessing pipeline and classification model.

Features used:

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

## API Endpoint

### GET /

Checks whether the API is running.

### POST /predict

Accepts passenger information and returns a survival prediction and probability.

Example request:

```json
{
  "Pclass": 3,
  "Sex": "female",
  "Age": 25,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 20.5,
  "Embarked": "S"
}
