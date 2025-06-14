from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load model parameters
with open("model.pkl", "rb") as f:
    w0, w = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

with open("categories.pkl", "rb") as f:
    categories = pickle.load(f)

# Base numeric features
base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

app = FastAPI()

class CarInput(BaseModel):
    make: str
    engine_cylinders: float
    year: int
    engine_fuel_type: str
    engine_hp: float
    transmission_type: str
    driven_wheels: str
    number_of_doors: float
    market_category: str
    vehicle_size: str
    vehicle_style: str
    highway_mpg: float
    city_mpg: float
    popularity: int

def prepare_X(df):
    df = df.copy()
    feats = base.copy()
    df['age'] = 2017 - df.year
    feats.append('age')

    for v in [2, 3, 4]:
        df[f'num_doors_{v}'] = (df.number_of_doors == v).astype(int)
        feats.append(f'num_doors_{v}')

    for c, values in categories.items():
        for v in values:
            df[f'{c}_{v}'] = (df[c] == v).astype(int)
            feats.append(f'{c}_{v}')

    df_num = df[feats]
    df_num = df_num.fillna(0)
    return df_num.values

@app.get("/")
def home():
    return {"message": "🚗 Welcome to Car Price Prediction API!"}

@app.post("/predict")
def predict_price(car: CarInput):
    car_dict = car.dict()
    df = pd.DataFrame([car_dict])
    X = prepare_X(df)
    y_pred_log = w0 + X.dot(w)
    print("Log price prediction:", y_pred_log)
    y_pred = np.expm1(y_pred_log[0])  # convert log(price+1) back to price
    return {"predicted_price": float(y_pred)}

