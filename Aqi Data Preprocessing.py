
import pandas as pd

aqi_df = pd.read_csv("DL001.csv")
weather_df = pd.read_csv("kaggel_weather_2013_to_2024.csv")

aqi_df['From Date'] = pd.to_datetime(aqi_df['From Date'])
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])

aqi_numeric = aqi_df.select_dtypes(include='number')
aqi_numeric['From Date'] = aqi_df['From Date']
aqi_daily = aqi_numeric.resample('D', on='From Date').mean().reset_index()

merged_df = pd.merge(aqi_daily, weather_df, left_on='From Date', right_on='DATE', how='inner')

# Impute PM10 using Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

features = ['PM2.5 (ug/m3)', 'NO (ug/m3)', 'NO2 (ug/m3)', 'NOx (ppb)', 'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)', 'temp', 'humidity', 'windspeed']
complete_pm10 = merged_df[merged_df['PM10 (ug/m3)'].notnull()]
missing_pm10 = merged_df[merged_df['PM10 (ug/m3)'].isnull()]

X = complete_pm10[features]
y = complete_pm10['PM10 (ug/m3)']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
model = RandomForestRegressor(random_state=42)
model.fit(X_imputed, y)

X_missing = imputer.transform(missing_pm10[features])
predicted_pm10 = model.predict(X_missing)
merged_df.loc[missing_pm10.index, 'PM10 (ug/m3)'] = predicted_pm10
merged_df['pm10_imputed'] = merged_df.index.isin(missing_pm10.index)

# Save preprocessed file
merged_df.to_csv("merged_aqi_weather.csv", index=False)
print("✅ Data preprocessing complete. File saved as merged_aqi_weather.csv")
