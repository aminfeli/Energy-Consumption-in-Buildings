import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ============================= بارگذاری داده =============================
df = pd.read_csv("D:/third semester/Casystudy2/2-Dataset/electricity_consumption-2.csv")

# ============================= تبدیل فرمت تاریخ =============================
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df.dropna(subset=['Timestamp'], inplace=True)

# ============================= پاک‌سازی داده‌ها =============================
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# ============================= مرتب‌سازی =============================
df.sort_values('Timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================= کدگذاری ستون‌های غیرعددی =============================
categorical_cols = [
    'Building Type',
    'Occupancy Schedule',
    'Building Orientation',
    'Peak Demand Reduction Indicator',
    'Carbon Emission Reduction Category'
]

df = pd.get_dummies(df, columns=categorical_cols)

# ============================= تنظیم ایندکس و پرکردن گپ‌های زمانی =============================
df.set_index('Timestamp', inplace=True)
df = df.asfreq('h')  # فواصل زمانی ساعتی
df.interpolate(method='linear', inplace=True)  # پر کردن گپ‌ها

# ============================= جدا کردن ستون هدف =============================
target_col = 'Energy Consumption (kWh)'
features = df.drop(columns=[target_col])
target = df[[target_col]]

# ============================= مقیاس‌بندی =============================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

features_scaled = scaler_X.fit_transform(features)
target_scaled = scaler_y.fit_transform(target)

# ============================= ساخت داده‌های توالی برای LSTM =============================
def create_dataset_multivariate(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps), :])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 24
X, y = create_dataset_multivariate(features_scaled, target_scaled, time_steps)

# ============================= تقسیم داده به آموزش و تست =============================
train_size = int(len(X) * 0.9)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ============================= ساخت مدل LSTM =============================
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# ============================= آموزش مدل =============================
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# ============================= پیش‌بینی =============================
y_pred_scaled = model.predict(X_test)

# ============================= برگرداندن مقیاس به حالت اولیه =============================
y_test_orig = scaler_y.inverse_transform(y_test)
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)

# ============================= ارزیابی مدل =============================
mae = mean_absolute_error(y_test_orig, y_pred_orig)
mse = mean_squared_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)

print("\n📈 ارزیابی مدل LSTM با تمام ویژگی‌ها:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# ============================= ترسیم نمودار =============================
plt.figure(figsize=(14, 6))
plt.plot(df.index[-len(y_test_orig):], y_test_orig, label='Actual')
plt.plot(df.index[-len(y_test_orig):], y_pred_orig, label='LSTM Forecast', color='red')
plt.legend()
plt.title("LSTM Forecast vs Actual Energy Consumption (Hourly) with All Features")
plt.xlabel("Date")
plt.ylabel("Energy Consumption (kWh)")
plt.grid()
plt.show()
