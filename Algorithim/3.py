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
df = pd.read_csv("D:/third semester/Casystudy2/2-Dataset/electricity_consumption-2.csv")
print(df.info())
print(df.head())
print(df.describe())

# بررسی مقدارهای گمشده
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

total_missing = missing_values.sum()
if total_missing > 0:
    print(f"Total missing values: {total_missing}")
    df = df.dropna()
    print("Missing values deleted.")
else:
    print("No missing values in the dataset.")

# بررسی مقادیر تکراری
duplicates_count = df.duplicated().sum()
if duplicates_count > 0:
    print(f"Duplicates are present. Total duplicate rows: {duplicates_count}")
    df = df.drop_duplicates()
    print("Duplicates deleted.")
else:
    print("No duplicates are present in the dataset.")

##################################
import matplotlib.pyplot as plt
import seaborn as sns

# انتخاب فقط ستون‌های عددی برای محاسبه همبستگی
numeric_df = df.select_dtypes(include=[np.number])

# حذف داده‌های پرت با Z-Score
from scipy.stats import zscore
z_scores = np.abs(zscore(numeric_df))
threshold = 3
non_outliers_mask = (z_scores < threshold).all(axis=1)
print(f"تعداد داده‌های پرت حذف‌شده: {(~non_outliers_mask).sum()}")
df = df[non_outliers_mask]
numeric_df = df.select_dtypes(include=[np.number])  # بازتعریف پس از حذف

# محاسبه ماتریس همبستگی پیرسون
corr_matrix = numeric_df.corr()

# چاپ ماتریس همبستگی (اختیاری)
print("Correlation matrix:")
print(corr_matrix)

# ترسیم heatmap همبستگی
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# برای شناسایی جفت‌های همبستگی خیلی بالا (مثلاً > 0.85)
threshold = 0.85
high_corr = []

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname_i = corr_matrix.columns[i]
            colname_j = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            high_corr.append((colname_i, colname_j, corr_value))

print(f"\nPairs with correlation higher than {threshold}:")
for pair in high_corr:
    print(f"{pair[0]} <--> {pair[1]} : correlation = {pair[2]:.2f}")
#####################################################################3XGBOOST Algorirhim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# هدف ما: پیش‌بینی 'Energy Consumption (kWh)'
target = 'Energy Consumption (kWh)'

# انتخاب ویژگی‌ها (تمام ستون‌های عددی به جز هدف)
features = numeric_df.drop(columns=[target]).columns
X = df[features]
y = df[target]

# تقسیم داده به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تعریف و آموزش مدل XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.2, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# پیش‌بینی
y_pred = model.predict(X_test)

# ارزیابی عملکرد مدل
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation Metrics:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  R2 Score: {r2:.4f}")
##########################################SHAP
import shap

# مقداردهی اولیه توضیح‌دهنده SHAP برای XGBoost
explainer = shap.Explainer(model)

# گرفتن مقادیر shap برای داده‌های تست
shap_values = explainer(X_test)

# نمودار خلاصه تأثیر ویژگی‌ها
shap.summary_plot(shap_values, X_test)

# نمودار تأثیر یک ویژگی خاص (مثلاً 'HVAC Consumption (kWh)')
# shap.dependence_plot('HVAC Consumption (kWh)', shap_values.values, X_test)

###################################################################################Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# انتخاب ویژگی‌ها و هدف
target = 'Energy Consumption (kWh)'
features = numeric_df.drop(columns=[target]).columns
X = df[features]
y = df[target]

# تقسیم داده به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تعریف و آموزش مدل Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# پیش‌بینی
y_pred = rf_model.predict(X_test)

# ارزیابی عملکرد مدل
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🔍 Random Forest Evaluation Metrics:")
print(f"  MAE: {mae:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  R² Score: {r2:.4f}")

#########################################Shap - random forest

import shap

# مقداردهی اولیه Explainer برای مدل Random Forest
explainer = shap.Explainer(rf_model, X_test)

# محاسبه مقادیر shap
shap_values = explainer(X_test)

# نمودار summary برای تحلیل کلی تأثیر ویژگی‌ها
shap.summary_plot(shap_values, X_test)

# [اختیاری] نمودار تأثیر ویژگی خاص:
# shap.dependence_plot("HVAC Consumption (kWh)", shap_values.values, X_test)
