import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.stats import zscore

# ======================== بارگذاری و پاکسازی داده ==========================
df = pd.read_csv("D:/third semester/Casystudy2/2-Dataset/electricity_consumption-2.csv")

# تبدیل فرمت ستون Timestamp و تنظیم ایندکس زمانی
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df.dropna(subset=['Timestamp'], inplace=True)
df.sort_values('Timestamp', inplace=True)
df.set_index('Timestamp', inplace=True)      # این خط خیلی مهم است
df = df.asfreq('H')  # تعیین فرکانس داده ها به صورت ساعتی

# حذف داده‌های تکراری
df = df.drop_duplicates()

# حذف داده‌های پرت با z-score فقط روی ستون‌های عددی
numeric_df = df.select_dtypes(include=[np.number])
z_scores = np.abs(zscore(numeric_df))
df = df[(z_scores < 3).all(axis=1)]

# ======================== تعیین متغیر هدف و ویژگی‌ها ==========================
target_col = 'Energy Consumption (kWh)'

# تمام ستون‌های عددی به جز هدف به عنوان ویژگی (exogenous variables)
features = df.select_dtypes(include=[np.number]).columns.drop(target_col)

X = df[features]
y = df[target_col]

# تقسیم داده به آموزش و تست (براساس زمان، 80 درصد آموزش)
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# ======================== انتخاب بهترین مدل SARIMA با Auto ARIMA ==========================
auto_model = auto_arima(
    y_train,
    exogenous=X_train,
    seasonal=True,
    m=24,                # دوره فصلی 24 ساعته برای داده‌های ساعتی
    stepwise=True,
    suppress_warnings=True,
    trace=True
)

print(f"Best SARIMAX order: {auto_model.order}, seasonal_order: {auto_model.seasonal_order}")

# ======================== آموزش مدل SARIMAX ==========================
sarima_model = SARIMAX(
    y_train,
    exog=X_train,
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order
)
sarima_result = sarima_model.fit(disp=False)

# ======================== پیش‌بینی مدل SARIMAX ==========================
sarima_forecast = sarima_result.predict(
    start=len(y_train),
    end=len(y_train) + len(y_test) - 1,
    exog=X_test
)

# ======================== محاسبه باقیمانده‌ها (Residuals) ==========================
residuals = y_test.values - sarima_forecast.values

# ======================== آموزش مدل LSTM روی Residuals ==========================
scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals.reshape(-1, 1))

def create_sequences(data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 10
X_lstm, y_lstm = create_sequences(residuals_scaled, seq_length)

split_lstm = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split_lstm], X_lstm[split_lstm:]
y_train_lstm, y_test_lstm = y_lstm[:split_lstm], y_lstm[split_lstm:]

model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, verbose=1)

lstm_pred_scaled = model_lstm.predict(X_test_lstm)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

# ======================== ترکیب پیش‌بینی‌ها ==========================
sarima_aligned = sarima_forecast[-len(lstm_pred):].values
final_forecast = sarima_aligned + lstm_pred.flatten()

# ======================== ارزیابی مدل ==========================
true_values = y_test[-len(lstm_pred):].values
mae = mean_absolute_error(true_values, final_forecast)
mse = mean_squared_error(true_values, final_forecast)
r2 = r2_score(true_values, final_forecast)

print("\n🔁 Hybrid SARIMA + LSTM Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# ======================== ترسیم نمودار ==========================
plt.figure(figsize=(14,6))
plt.plot(true_values, label='Actual')
plt.plot(sarima_aligned, label='SARIMA Forecast')
plt.plot(final_forecast, label='Hybrid Forecast (SARIMA + LSTM)', color='red')
plt.legend()
plt.title("Hybrid SARIMA + LSTM Forecasting")
plt.xlabel("Time Steps")
plt.ylabel("Energy Consumption (kWh)")
plt.show()
