import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# ============================= بارگذاری داده =============================
df = pd.read_csv("D:/third semester/Casystudy2/2-Dataset/electricity_consumption-2.csv")

# ============================= تبدیل فرمت تاریخ =============================
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# حذف تاریخ‌های نامعتبر
df.dropna(subset=['Timestamp'], inplace=True)

# ============================= پاک‌سازی داده‌ها =============================
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# ============================= مرتب‌سازی زمانی =============================
df.sort_values(by='Timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================= آماده‌سازی داده برای Prophet =============================
df_prophet = df[['Timestamp', 'Energy Consumption (kWh)']].copy()
df_prophet.rename(columns={'Timestamp': 'ds', 'Energy Consumption (kWh)': 'y'}, inplace=True)

# ============================= تکمیل فواصل زمانی ساعتی =============================
df_prophet = df_prophet.set_index('ds').asfreq('h')
df_prophet['y'] = df_prophet['y'].interpolate(method='linear')
df_prophet.reset_index(inplace=True)

# ============================= Cross-Validation با TimeSeriesSplit =============================
tscv = TimeSeriesSplit(n_splits=5)

mae_list = []
mse_list = []
r2_list = []

for fold, (train_index, test_index) in enumerate(tscv.split(df_prophet)):
    train = df_prophet.iloc[train_index]
    test = df_prophet.iloc[test_index]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.add_seasonality(name='hourly', period=24, fourier_order=10)

    model.fit(train)

    future = test[['ds']]
    forecast = model.predict(future)

    y_true = test['y'].values
    y_pred = forecast['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)

    print(f"Fold {fold + 1}: MAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}")

print("\n📊 میانگین ارزیابی‌ها روی همه Fold ها:")
print(f"Mean MAE: {np.mean(mae_list):.4f}")
print(f"Mean MSE: {np.mean(mse_list):.4f}")
print(f"Mean R²: {np.mean(r2_list):.4f}")

# اگر خواستی می‌تونی نمودار پیش‌بینی آخرین Fold رو ترسیم کنی
plt.figure(figsize=(14,6))
plt.plot(test['ds'], y_true, label='Actual')
plt.plot(test['ds'], y_pred, label='Predicted', color='red')
plt.legend()
plt.title('Prophet Forecast vs Actual (Last Fold)')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.grid()
plt.show()
