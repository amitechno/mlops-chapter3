# test_main.py

import pytest
from fastapi.testclient import TestClient
from main import app
from pydantic import ValidationError

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API for model inference!"}

def test_valid_prediction():
    data = {
        "age": 30,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
        "salary": ">50K",
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": 1}

def test_invalid_prediction():
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 96372,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Divorced",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States",
        "salary": "<=50K",
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": 0}

def test_invalid_data():
    # This test case checks for invalid data where age and education_num are negative values
    data = {
        "age": -30,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": -13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
        "salary": ">50K",
    }
    with pytest.raises(ValidationError):
        client.post("/predict/", json=data)

