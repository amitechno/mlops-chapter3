# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
from sklearn.linear_model import LogisticRegression

# Create FastAPI app
app = FastAPI()

# Pydantic model to ingest the body for POST request
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    salary: str

    @validator('age', 'fnlgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week')
    def check_non_negative(cls, v):
        if v < 0:
            raise ValueError(f"{v} must be a non-negative value")
        return v

    @validator('education_num')
    def check_education_num(cls, v, values):
        education = values.get('education')
        if education == 'Preschool' and v != 1:
            raise ValueError(f"Educational number for Preschool should be 1")
        elif education == 'Doctorate' and v != 16:
            raise ValueError(f"Educational number for Doctorate should be 16")
        return v

# Load the trained model
model = LogisticRegression()

# Define the GET endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the API for model inference!"}

# Define the POST endpoint for model inference
@app.post("/predict/")
def predict(input_data: InputData):
    X = np.array([[input_data.age, input_data.workclass, input_data.fnlgt, input_data.education,
                   input_data.education_num, input_data.marital_status, input_data.occupation,
                   input_data.relationship, input_data.race, input_data.sex, input_data.capital_gain,
                   input_data.capital_loss, input_data.hours_per_week, input_data.native_country]])
    preds = model.predict(X)
    return {"prediction": int(preds[0])}
