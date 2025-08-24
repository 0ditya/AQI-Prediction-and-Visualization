# model_training.py

from pycaret.regression import *
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("merged_aqi_weather.csv")

data = data.dropna(subset=['PM2.5 (ug/m3)', 'PM10 (ug/m3)'])
data = data.drop(columns=['From Date', 'DATE', 'pm10_imputed'], errors='ignore')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# PM2.5 MODEL
print("\n--- Training PM2.5 model ---")
setup(data=train_data, target='PM2.5 (ug/m3)', session_id=42, use_gpu=True, verbose=False)
best_pm25_model = compare_models()
final_pm25_model = finalize_model(best_pm25_model)
pred_pm25 = predict_model(final_pm25_model, data=test_data)
save_model(final_pm25_model, 'best_model_pm25')

# PM10 MODEL
print("\n--- Training PM10 model ---")
setup(data=train_data, target='PM10 (ug/m3)', session_id=42, use_gpu=True, verbose=False)
best_pm10_model = compare_models()
final_pm10_model = finalize_model(best_pm10_model)
pred_pm10 = predict_model(final_pm10_model, data=test_data)
save_model(final_pm10_model, 'best_model_pm10')

print("✅ Models trained and saved.")
