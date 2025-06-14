import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.read_csv('data.csv')
print(df.dtypes)
df.columns = df.columns.str.lower().str.replace(' ', '_')
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
plt.figure(figsize=(6, 4))
sns.histplot(df.msrp, bins=40, color='black', alpha=1)
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.title('Distribution of prices')
plt.show()
plt.figure(figsize=(6, 4))
sns.histplot(df.msrp[df.msrp < 100000], bins=40, color='black', alpha=1)
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.title('Distribution of prices')
plt.show()
log_price = np.log1p(df.msrp)
plt.figure(figsize=(6, 4))
sns.histplot(log_price, bins=40, color='black', alpha=1)
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Distribution of prices after log transformation')
plt.show()
np.random.seed(2)
n = len(df)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)
idx = np.arange(n)
np.random.shuffle(idx)
df_shuffled = df.iloc[idx]
df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]
base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
plt.figure(figsize=(6, 4))
sns.histplot(y_train, label='target', color='#222222', alpha=0.6, bins=40)
sns.histplot(y_pred, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)
plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')
plt.show()
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
X_val = prepare_X(df_val)
y_pred_val = w_0 + X_val.dot(w)
print("Train RMSE:", rmse(y_train, y_pred))
print("Validation RMSE:", rmse(y_val, y_pred_val))
base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
def prepare_X(df):
    df = df.copy()
    features = base.copy()
    df['age'] = 2017 - df.year
    features.append('age')
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
X_val = prepare_X(df_val)
y_pred_val = w_0 + X_val.dot(w)
print("Train RMSE:", rmse(y_train, y_pred))
print("Validation RMSE:", rmse(y_val, y_pred_val))
plt.figure(figsize=(6, 4))
sns.histplot(y_val, label='target', color='#222222', alpha=0.6, bins=40)
sns.histplot(y_pred_val, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)
plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')
plt.show()
df['make'].value_counts()
df['number_of_doors'].value_counts()
def prepare_X(df):
    df = df.copy()
    features = base.copy()
    df['age'] = 2017 - df.year
    features.append('age')
    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)
    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
df.columns
print(df.dtypes)
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
X_val = prepare_X(df_val)
y_pred_val = w_0 + X_val.dot(w)
print("Train RMSE:", rmse(y_train, y_pred))
print("Validation RMSE:", rmse(y_val, y_pred_val))
plt.figure(figsize=(6, 4))
sns.histplot(y_val, label='target', color='#222222', alpha=0.6, bins=40)
sns.histplot(y_pred_val, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)
plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')
plt.show()
df['engine_fuel_type'].value_counts()
def prepare_X(df):
    df = df.copy()
    features = base.copy()
    df['age'] = 2017 - df.year
    features.append('age')
    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)
    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)
    for v in ['regular_unleaded', 'premium_unleaded_(required)',
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
X_val = prepare_X(df_val)
y_pred_val = w_0 + X_val.dot(w)
print("Train RMSE:", rmse(y_train, y_pred))
print("Validation RMSE:", rmse(y_val, y_pred_val))
df['transmission_type'].value_counts()
def prepare_X(df):
    df = df.copy()
    features = base.copy()
    df['age'] = 2017 - df.year
    features.append('age')
    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)
    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)
    for v in ['regular_unleaded', 'premium_unleaded_(required)',
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)
    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
X_val = prepare_X(df_val)
y_pred_val = w_0 + X_val.dot(w)
print("Train RMSE:", rmse(y_train, y_pred))
print("Validation RMSE:", rmse(y_val, y_pred_val))
df['driven_wheels'].value_counts()
df['market_category'].value_counts().head(5)
df['vehicle_size'].value_counts().head(5)
df['vehicle_style'].value_counts().head(5)
def prepare_X(df):
    df = df.copy()
    features = base.copy()
    df['age'] = 2017 - df.year
    features.append('age')
    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)
    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)
    for v in ['regular_unleaded', 'premium_unleaded_(required)',
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)
    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)
    for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
        feature = 'is_driven_wheens_%s' % v
        df[feature] = (df['driven_wheels'] == v).astype(int)
        features.append(feature)
    for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
        feature = 'is_mc_%s' % v
        df[feature] = (df['market_category'] == v).astype(int)
        features.append(feature)
    for v in ['compact', 'midsize', 'large']:
        feature = 'is_size_%s' % v
        df[feature] = (df['vehicle_size'] == v).astype(int)
        features.append(feature)
    for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
        feature = 'is_style_%s' % v
        df[feature] = (df['vehicle_style'] == v).astype(int)
        features.append(feature)
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
X_val = prepare_X(df_val)
y_pred_val = w_0 + X_val.dot(w)
print("Train RMSE:", rmse(y_train, y_pred))
print("Validation RMSE:", rmse(y_val, y_pred_val))
categorical_variables = [
    'make', 'engine_fuel_type', 'transmission_type', 'driven_wheels',
    'market_category', 'vehicle_size', 'vehicle_style'
]
categories = {}
for c in categorical_variables:
    categories[c] = list(df[c].value_counts().head().index)
def prepare_X(df):
    df = df.copy()
    features = base.copy()
    df['age'] = 2017 - df.year
    features.append('age')
    for v in [2, 3, 4]:
        df['num_doors_%s' % v] = (df.number_of_doors == v).astype('int')
        features.append('num_doors_%s' % v)
    for c, values in categories.items():
        for v in values:
            df['%s_%s' % (c, v)] = (df[c] == v).astype('int')
            features.append('%s_%s' % (c, v))
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X, features

X_train, features = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)
y_pred = w_0 + X_train.dot(w)
X_val, features = prepare_X(df_val)
y_pred_val = w_0 + X_val.dot(w)
print("Train RMSE:", rmse(y_train, y_pred))
print("Validation RMSE:", rmse(y_val, y_pred_val))
plt.figure(figsize=(6, 4))
sns.histplot(y_val, label='target', color='#222222', alpha=0.6, bins=40)
sns.histplot(y_pred_val, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)
plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')
plt.show()
w_0
w
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]
X_train, features = prepare_X(df_train)
for r in [0, 0.001, 0.01, 0.1, 1, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    print('%5s, %.2f, %.2f, %.2f' % (r, w_0, w[13], w[21]))
X_train, features = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0)
y_pred = w_0 + X_train.dot(w)
print('train', rmse(y_train, y_pred))
X_val , features = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('val', rmse(y_val, y_pred))
X_train , features = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)
y_pred = w_0 + X_train.dot(w)
print('train', rmse(y_train, y_pred))
X_val , features = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('val', rmse(y_val, y_pred))
X_train , features = prepare_X(df_train)
X_val , features = prepare_X(df_val)
for r in [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w_0 + X_val.dot(w)
    print('%6s' %r, rmse(y_val, y_pred))
X_train , features = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)
y_pred_train = w_0 + X_train.dot(w)
print('train:', rmse(y_train, y_pred_train))
X_val , features = prepare_X(df_val)
y_pred_val = w_0 + X_val.dot(w)
print('validation:', rmse(y_val, y_pred_val))
X_test , features = prepare_X(df_test)
y_pred_test = w_0 + X_test.dot(w)
print('test:', rmse(y_test, y_pred_test))
w_0
w
df_test.iloc[20]
car = df_test.iloc[20].to_dict()
car
df_small = pd.DataFrame([car])
df_small
X_small , features = prepare_X(df_small)
X_small
y_pred = w_0 + X_small.dot(w)
y_pred = y_pred[0]
y_pred
np.expm1(y_pred)
df_full_train = pd.concat([df_train, df_val])
y_full_train = np.concatenate([y_train, y_val])
y_full_train
df_full_train
df_full_train = df_full_train.reset_index(drop=True)
df_full_train
X_full_train,features = prepare_X(df_full_train)
X_full_train
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.0)
w0, w
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=.001)
w0, w
X_test , features = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
print("rmse:", score)
X_train , features = prepare_X(df_full_train)
y_p = w0 + X_train.dot(w)
score = rmse(y_full_train, y_p)
print("rmse:", score)
df_test.iloc[20]
car = df_test.iloc[20].to_dict()
car
df_small = pd.DataFrame([car])
df_small
X_small , features = prepare_X(df_full_train)
X_small
y_pred = w0 + X_small.dot(w)
y_pred = y_pred[0]
y_pred
np.expm1(y_pred)
np.expm1(y_test[20])
import pickle
output = (w0, w)
with open('model.pkl', 'wb') as f_out:
    pickle.dump(output, f_out)
with open('features.pkl', 'wb') as f_out:
    pickle.dump(features, f_out)
with open('categories.pkl', 'wb') as f_out:
    pickle.dump(categories, f_out)
