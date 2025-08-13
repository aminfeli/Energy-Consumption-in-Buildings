import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scipy.stats import zscore

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ============================= بارگذاری داده =============================
df = pd.read_csv("D:/third semester/Casystudy2/2-Dataset/electricity_consumption-2.csv")

# ============================= پیش‌پردازش مشابه =============================
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df.dropna(subset=['Timestamp'], inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=[np.number])
z_scores = np.abs(zscore(numeric_cols))
df = df[(z_scores < 3).all(axis=1)]

df.sort_values(by='Timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================= آماده‌سازی برای Prophet =============================
df_prophet = df[['Timestamp', 'Energy Consumption (kWh)']].copy()
df_prophet.rename(columns={'Timestamp': 'ds', 'Energy Consumption (kWh)': 'y'}, inplace=True)

df_prophet = df_prophet.set_index('ds').asfreq('h')
df_prophet['y'] = df_prophet['y'].interpolate(method='linear')
df_prophet.reset_index(inplace=True)

# ============================= تقسیم داده =============================
train_size = int(len(df_prophet) * 0.8)
train_prophet = df_prophet.iloc[:train_size]
test_prophet = df_prophet.iloc[train_size:]

# ========== مدل Prophet ==========
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
model.add_seasonality(name='hourly', period=24, fourier_order=10)
model.fit(train_prophet)

future = model.make_future_dataframe(periods=len(test_prophet), freq='h')
forecast = model.predict(future)

df_eval_prophet = forecast[['ds', 'yhat']].merge(test_prophet[['ds', 'y']], on='ds', how='inner')

# ============================= آماده‌سازی داده برای LSTM =============================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_prophet[['y']].values)

def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 24  # 24 ساعت پنجره
X, y = create_sequences(data_scaled, seq_length)

train_size_lstm = int(len(X) * 0.8)
X_train, y_train = X[:train_size_lstm], y[:train_size_lstm]
X_test, y_test = X[train_size_lstm:], y[train_size_lstm:]

# ============================= ساخت مدل LSTM =============================
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# ============================= پیش‌بینی با LSTM =============================
y_pred_lstm_scaled = lstm_model.predict(X_test)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)

# تنظیم شاخص‌های زمانی برای داده تست LSTM (چون دنباله است)
time_index_test = df_prophet['ds'].iloc[seq_length + train_size_lstm : seq_length + train_size_lstm + len(y_pred_lstm)]

# ============================= آماده‌سازی داده‌ها برای ترکیب =============================
# فقط بخش تست Prophet که با LSTM برابر باشد:
df_eval_prophet_cut = df_eval_prophet[df_eval_prophet['ds'].isin(time_index_test)]

# مرتب‌سازی بر اساس زمان
df_eval_prophet_cut = df_eval_prophet_cut.sort_values('ds').reset_index(drop=True)
time_index_test = time_index_test.reset_index(drop=True)

# ============================= ترکیب پیش‌بینی‌ها (میانگین ساده) =============================
combined_pred = (df_eval_prophet_cut['yhat'].values + y_pred_lstm.flatten()) / 2

# ============================= ارزیابی مدل‌ها =============================
y_true = df_eval_prophet_cut['y'].values
y_pred_prophet = df_eval_prophet_cut['yhat'].values
y_pred_lstm = y_pred_lstm.flatten()

print("\nارزیابی مدل Prophet:")
print(f"MAE: {mean_absolute_error(y_true, y_pred_prophet):.4f}")
print(f"R2: {r2_score(y_true, y_pred_prophet):.4f}")

print("\nارزیابی مدل LSTM:")
print(f"MAE: {mean_absolute_error(y_true, y_pred_lstm):.4f}")
print(f"R2: {r2_score(y_true, y_pred_lstm):.4f}")

print("\nارزیابی مدل ترکیبی (میانگین ساده):")
print(f"MAE: {mean_absolute_error(y_true, combined_pred):.4f}")
print(f"R2: {r2_score(y_true, combined_pred):.4f}")

# ============================= ترسیم نمودار =============================
plt.figure(figsize=(14,6))
plt.plot(time_index_test, y_true, label='Actual')
plt.plot(time_index_test, y_pred_prophet, label='Prophet Prediction')
plt.plot(time_index_test, y_pred_lstm, label='LSTM Prediction')
plt.plot(time_index_test, combined_pred, label='Combined Prediction', linestyle='--')
plt.legend()
plt.title('Comparison of Prophet, LSTM, and Combined Predictions')
plt.xlabel('Timestamp')
plt.ylabel('Energy Consumption (kWh)')
plt.grid()
plt.show()
