import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

# ============================= تنظیم ایندکس =============================
df.set_index('Timestamp', inplace=True)
df = df.asfreq('h')  # hourly frequency
df['Energy Consumption (kWh)'].interpolate(method='linear', inplace=True)  # fill missing

# ============================= ACF & PACF Plots =============================
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plot_acf(df['Energy Consumption (kWh)'], lags=48, zero=False, ax=plt.gca())
plt.title("Auto-correlation Function (ACF)\nEnergy Consumption (kWh)")

plt.subplot(1,2,2)
plot_pacf(df['Energy Consumption (kWh)'], lags=48, zero=False, method='ywm', ax=plt.gca())
plt.title("Partial Auto-correlation Function (PACF)\nEnergy Consumption (kWh)")

plt.tight_layout()
plt.show()
