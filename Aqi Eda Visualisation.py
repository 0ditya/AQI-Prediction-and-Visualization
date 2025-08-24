# eda_visualisation.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load merged data
df = pd.read_csv("merged_aqi_weather.csv")

# Correlation matrix
plt.figure(figsize=(14, 10))
corr = df[[
    'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO (ug/m3)', 'NO2 (ug/m3)', 'NOx (ppb)', 'SO2 (ug/m3)',
    'CO (mg/m3)', 'Ozone (ug/m3)', 'temp', 'humidity', 'windspeed', 'sealevelpressure', 'precip']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Weather vs Pollutants")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

# Time series plot
plt.figure(figsize=(12, 6))
df.plot(x='From Date', y=['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'temp', 'humidity'], figsize=(16,6))
plt.title("AQI vs Weather Parameters Over Time")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("aqi_weather_timeseries.png")
print("✅ EDA visualizations saved.")
