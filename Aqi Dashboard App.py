# app.py (Streamlit Dashboard)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Delhi AQI Dashboard", layout="wide")
st.title("🌫️ Delhi AQI Analysis & Prediction Dashboard")

df = pd.read_csv("merged_aqi_weather.csv")

# Correlation heatmap
st.subheader("📊 Correlation Between AQI & Weather")
corr = df[[
    'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'temp', 'humidity', 'windspeed',
    'sealevelpressure', 'precip']].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Custom CSV for prediction
st.subheader("🔮 Predict PM2.5 or PM10 from Weather Data")
model_option = st.selectbox("Select Model", ["PM2.5", "PM10"])
model_path = "best_model_pm25" if model_option == "PM2.5" else "best_model_pm10"
model = load_model(model_path)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # Drop target if present
    if model_option == "PM2.5" and 'PM2.5 (ug/m3)' in input_df.columns:
        input_df.drop(columns=['PM2.5 (ug/m3)'], inplace=True)
    if model_option == "PM10" and 'PM10 (ug/m3)' in input_df.columns:
        input_df.drop(columns=['PM10 (ug/m3)'], inplace=True)

    # Check model input compatibility
    required = model.feature_names_in_
    if not all(col in input_df.columns for col in required):
        st.error("Uploaded data is missing some required columns.")
    else:
        result = predict_model(model, data=input_df)
        st.success(f"Prediction using {model_option} model completed.")
        st.dataframe(result[['Label']])
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, file_name=f"{model_option}_predictions.csv")
