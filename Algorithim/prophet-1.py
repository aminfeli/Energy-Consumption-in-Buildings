import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scipy.stats import zscore

# ============================= بارگذاری داده =============================
df = pd.read_csv("D:/third semester/Casystudy2/2-Dataset/electricity_consumption-2.csv")

# ============================= تبدیل فرمت تاریخ =============================
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# حذف تاریخ‌های نامعتبر
df.dropna(subset=['Timestamp'], inplace=True)

# ============================= پاک‌سازی داده‌ها =============================
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# حذف داده‌های پرت با Z-score
numeric_cols = df.select_dtypes(include=[np.number])
z_scores = np.abs(zscore(numeric_cols))
df = df[(z_scores < 3).all(axis=1)]

# ============================= مرتب‌سازی زمانی =============================
df.sort_values(by='Timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================= آماده‌سازی داده برای Prophet =============================
df_prophet = df[['Timestamp', 'Energy Consumption (kWh)']].copy()
df_prophet.rename(columns={'Timestamp': 'ds', 'Energy Consumption (kWh)': 'y'}, inplace=True)

# ============================= تکمیل فواصل زمانی ساعتی =============================
df_prophet = df_prophet.set_index('ds').asfreq('h')  # توجه به h کوچک برای pandas جدید
df_prophet['y'] = df_prophet['y'].interpolate(method='linear')  # پر کردن گپ‌ها
df_prophet.reset_index(inplace=True)

print("\n📊 داده آماده برای Prophet بعد از تکمیل ساعتی:")
print(df_prophet.head(10))

# ============================= تقسیم آموزش و تست =============================
train_size = int(len(df_prophet) * 0.8)
train = df_prophet.iloc[:train_size]
test = df_prophet.iloc[train_size:]

# ============================= ساخت مدل Prophet =============================
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
model.add_seasonality(name='hourly', period=24, fourier_order=10)


model.fit(train)

# ============================= ساخت future با متد Prophet =============================
future = model.make_future_dataframe(periods=len(test), freq='h')

# ============================= پیش‌بینی =============================
forecast = model.predict(future)

# ============================= آماده‌سازی داده برای ارزیابی =============================
df_eval = forecast[['ds', 'yhat']].merge(test[['ds', 'y']], on='ds', how='inner')

y_true = df_eval['y'].values
y_pred = df_eval['yhat'].values

# ============================= ارزیابی مدل =============================
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n📈 ارزیابی مدل Prophet:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# ============================= ترسیم نمودار =============================
plt.figure(figsize=(14, 6))
plt.plot(df_eval['ds'], y_true, label='Actual')
plt.plot(df_eval['ds'], y_pred, label='Prophet Forecast', color='red')
plt.legend()
plt.title("Prophet Forecast vs Actual Energy Consumption (Hourly)")
plt.xlabel("Date")
plt.ylabel("Energy Consumption (kWh)")
plt.grid()
plt.show()

# نمودارهای داخلی Prophet
model.plot(forecast)
plt.show()

model.plot_components(forecast)
plt.show()
