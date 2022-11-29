import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter import inference, process_data

encoder = joblib.load('model/encoder.pkl')
model = joblib.load('model/model.pkl')
lb = joblib.load('model/lb.pkl')

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


app = FastAPI()

@app.get("/")
def root():
    return {"Welcome to my API"}

@app.post("/model_inference")
async def model_inference(data: Data):

    # Getting BaseModel as Dictionary and then as DataFrame
    df = pd.DataFrame(data.dict(by_alias=True), index=[0])

    # Processing data
    X, _y, _encoder, _lb = process_data(
        df,
        categorical_features=encoder['features'],
        encoder=encoder['encoder'],
        lb=lb,
        training=False
    )

    # Getting inference
    pred = inference(model, X)

    if pred[0] == 1:
        pred = "Salary > 50k"
    else:
        pred = "Salary <= 50k"
        
    return {"Result": pred}
