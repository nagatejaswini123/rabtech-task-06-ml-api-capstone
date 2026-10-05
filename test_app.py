from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    data = {
        "Pclass": 3,
        "Sex": "female",
        "Age": 25,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 20.5,
        "Embarked": "S"
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "probability" in result

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1
