
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# بارگذاری داده‌ها
df = pd.read_csv("D:/third semester/Casystudy2/2-Dataset/electricity_consumption.csv")
print(df.info())
print(df.head())
print(df.describe())

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Series
series = df['Energy Consumption (kWh)'].copy()

# Step 2: EWM mean
ewm_mean = series.ewm(span=10, adjust=False).mean()

# Step 3: MAD
mad = np.median(np.abs(series - np.median(series)))

# Step 4: Robust Z-Score
z_rehws = (series - ewm_mean) / (1.4826 * mad)

# Step 5: Detect outliers
threshold = 3
outliers_rehws = abs(z_rehws) > threshold
print(f"[REHWS] Number of outliers: {outliers_rehws.sum()}")


# Step 6: Plot
plt.figure(figsize=(15,5))
plt.plot(df['Timestamp'], series, label='Actual')
plt.plot(df['Timestamp'][outliers_rehws], series[outliers_rehws], 'ro', label='Outliers')
plt.title('Outlier Detection using REHWS')
plt.xlabel('Timestamp')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.show()  # ← this makes the plot appear


