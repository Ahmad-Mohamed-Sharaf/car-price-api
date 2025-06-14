import pickle
import numpy as np
import pandas as pd

# Load the trained model (intercept w0 and weights w) from the pickle file
with open('model.pkl', 'rb') as f_in:
    w0, w = pickle.load(f_in)
# Load the categories used during training
with open('categories.pkl', 'rb') as f_in:
    categories = pickle.load(f_in)

# Load the features and categories used during training
with open('features.pkl', 'rb') as f_in:
    features = pickle.load(f_in)

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
# Same prepare_X function as used in main.py to transform raw input data
def prepare_X(df):
    df = df.copy()
    features = base.copy()
    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        df[f'num_doors_{v}'] = (df.number_of_doors == v).astype(int)
        features.append(f'num_doors_{v}')

    for c, values in categories.items():
        for v in values:
            df[f'{c}_{v}'] = (df[c] == v).astype(int)
            features.append(f'{c}_{v}')

    df_num = df[features]
    df_num = df_num.fillna(0)
    return df_num.values

# Example car input (same format as training data)
car = {
    'engine_hp': 200,
    'engine_cylinders': 4,
    'highway_mpg': 30,
    'city_mpg': 25,
    'popularity': 2000,
    'year': 2015,
    'number_of_doors': 4,
    'make': 'toyota',
    'engine_fuel_type': 'regular_unleaded',
    'transmission_type': 'automatic',
    'driven_wheels': 'front_wheel_drive',
    'market_category': 'crossover',
    'vehicle_size': 'compact',
    'vehicle_style': 'sedan'
}
# Convert the car dictionary to a DataFrame
df = pd.DataFrame([car])
# Prepare the input features
X = prepare_X(df)
# Make the prediction using the linear model: y = w0 + X·w
y_pred = w0 + X.dot(w)
# Output the predicted log(price + 1)
print("Predicted log(price + 1):", y_pred[0])
# Convert the log prediction back to original price scale
print("Predicted price:", np.expm1(y_pred[0]))
