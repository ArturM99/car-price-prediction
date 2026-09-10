import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()
with open('Model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: int
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: int

class Prediction(BaseModel):
    ID: int
    Pred: str
    Price: int


@app.get("/status")
def status():
    return "I am OK"

@app.get("/version")
def version():
    return model['metadata']

@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {
        'ID': form.id,
        'Pred': y[0],
        'Price': form.price
    }


