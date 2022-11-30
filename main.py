"""
Module responsable for running the API with the model prediction flow

Author: Gabriel
Date: 11/2022

"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter import inference, process_data

file_path = os.path.dirname(os.path.abspath(__file__))

encoder = joblib.load(f'{file_path}/model/encoder.pkl')
model = joblib.load(f'{file_path}/model/model.pkl')
lb = joblib.load(f'{file_path}/model/lb.pkl')


class Data(BaseModel):
    age:            int
    workclass:      str
    fnlgt:          int
    education:      str
    education_num:  int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation:     str
    relationship:   str
    race:           str
    sex:            str
    capital_gain:   int = Field(alias="capital-gain")
    capital_loss:   int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    
    class Config:
        schema_extra = {
            "example": {
                'age': 45,
                'workclass': 'State-gov',
                'fnlgt': 50567,
                'education': 'HS-grad',
                'education-num': 9,
                'marital-status': 'Married-civ-spouse',
                'occupation': 'Exec-managerial',
                'relationship': 'Wife',
                'race': 'White',
                'sex': 'Female',
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }


app = FastAPI()


@app.get("/")
def root():
    return {"Welcome to my API"}


@app.post("/model_inference")
async def model_inference(data: Data):

    # Getting BaseModel as Dictionary and then as DataFrame
    df_input = pd.DataFrame(data.dict(by_alias=True), index=[0])

    # Processing data
    X_input, _y, _encoder, _lb = process_data(
        df_input,
        categorical_features=encoder['features'],
        encoder=encoder['encoder'],
        lb=lb,
        training=False
    )

    # Getting inference
    pred = inference(model, X_input)

    if pred == 1:
        pred = "Salary > 50k"
    else:
        pred = "Salary <= 50k"

    return {"Result": pred}
