import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import seaborn as sns
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(script_dir, '../Data Historis JKSE_raw.csv')
df = pd.read_csv(file_dir, parse_dates=['Tanggal'])

df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce', infer_datetime_format=True)
df = df.sort_values(by="Tanggal", ascending=True).reset_index(drop=True)

# Terlihat kolom masih bertipe object karena separator "," instead of "." maka dibersihkan
numeric_cols = ['Terakhir','Pembukaan','Tertinggi','Terendah']
for col in numeric_cols:
    # Replace ',' with '.' and remove any extra spaces
    df[col] = df[col].str.replace('.', '', regex=False)  # remove thousands sep if exists
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Membersihkan "B"(stands for billion) pada kolom Vol.
def parse_volume(x):
    if pd.isna(x):
        return None
    x = x.replace('.', '').replace(',', '.').upper()
    if 'B' in x:
        return float(x.replace('B','')) * 1e9
    elif 'M' in x: # in case not all of them are "B"
        return float(x.replace('M','')) * 1e6
    elif 'K' in x:
        return float(x.replace('K','')) * 1e3
    else:
        return float(x)
df['Vol.'] = df['Vol.'].apply(parse_volume)

df['Perubahan%'] = df['Perubahan%'].str.replace('%','').str.replace(',','.', regex=False)
df['Perubahan%'] = pd.to_numeric(df['Perubahan%'], errors='coerce')

data = df[["Terakhir", "Terendah", "Tertinggi"]].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

np.save("scaled_stock_data.npy", scaled_data)
joblib.dump(scaler, 'output/scaler.pkl')

def create_sequences(data, time_steps=10, target_col_index=0):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])   # previous 60 days (all features)
        y.append(data[i, target_col_index])  # next day's Close
    X = np.array(X)
    y = np.array(y)
    np.save(f"output/X_timestep_{time_steps}", X)
    np.save(f"output/y_timestep_{time_steps}", y)
    print("Data saved successfully")

time_steps = int(input("Enter the number of time steps(eg 10, 20, 30): "))
create_sequences(scaled_data, time_steps, 0)