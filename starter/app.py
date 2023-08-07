import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, ValidationError, validator
import joblib
from starter.starter.ml.model import inference

app = FastAPI()

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class Item(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Married-civ-spouse")
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")

    @validator("age", "fnlgt", "education_num", "capital_gain", "capital_loss", "hours_per_week")
    def check_non_negative_int(cls, v):
        if v < 0:
            raise ValueError("Integer values should not be negative.")
        return v

def online_inference(row_dict, model_path, cat_features):
    # Load the model from `model_path`
    model, encoder, lb = joblib.load(model_path)

    row_transformed = list()
    X_categorical = list()
    X_continuous = list()

    for key, value in row_dict.items():
        mod_key = key.replace('_', '-')
        if mod_key in cat_features:
            X_categorical.append(value)
        else:
            X_continuous.append(value)

    y_cat = encoder.transform([X_categorical])
    y_conts = np.asarray([X_continuous])
    row_transformed = np.concatenate([y_conts, y_cat], axis=1)

    # Get inference from model
    preds = inference(model=model, X=row_transformed)

    return '>50K' if preds[0] else '<=50K'


@app.get("/")
def home():
    return {"Hello": "Welcome to project 3!"}

@app.post('/predict')
async def predict_income(inputrow: Item):
    model_path = 'starter/model/census_model.pkl'
    try:
        prediction = online_inference(inputrow.dict(), model_path, CAT_FEATURES)
        return {"income class": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
