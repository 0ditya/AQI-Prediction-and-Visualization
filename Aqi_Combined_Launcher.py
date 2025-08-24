import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Delhi AQI Dashboard", layout="wide")
st.title("🌫️ Delhi AQI Analysis & Prediction Dashboard")

df = pd.read_csv("merged_aqi_weather.csv")

# Sidebar for menu
menu = st.sidebar.selectbox("Choose a section", [
    "📈 AQI vs Weather Over Time",
    "📊 Correlation Heatmap",
    "🔮 AQI Prediction"
])

# AQI vs Weather Time Series
if menu == "📈 AQI vs Weather Over Time":
    st.subheader("📈 AQI & Weather Parameter Trends")
    variable = st.selectbox("Select weather parameter", ["temp", "humidity", "windspeed", "precip", "sealevelpressure"])
    target = st.selectbox("Select AQI pollutant", ["PM2.5 (ug/m3)", "PM10 (ug/m3)"])

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_xlabel("Date")
    ax1.set_ylabel(target, color='tab:red')
    ax1.plot(pd.to_datetime(df['From Date']), df[target], color='tab:red', label=target)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel(variable, color='tab:blue')
    ax2.plot(pd.to_datetime(df['From Date']), df[variable], color='tab:blue', label=variable)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    st.pyplot(fig)

# Correlation Heatmap
elif menu == "📊 Correlation Heatmap":
    st.subheader("📊 Correlation Between AQI & Weather")
    corr = df[[
        'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'temp', 'humidity', 'windspeed',
        'sealevelpressure', 'precip']].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# AQI Prediction
elif menu == "🔮 AQI Prediction":
    st.subheader("🔮 Predict PM2.5 or PM10 from Weather Data")
    model_option = st.selectbox("Select Model", ["PM2.5", "PM10"])
    model_path = "best_model_pm25" if model_option == "PM2.5" else "best_model_pm10"
    model = load_model(model_path)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)

        if model_option == "PM2.5" and 'PM2.5 (ug/m3)' in input_df.columns:
            input_df.drop(columns=['PM2.5 (ug/m3)'], inplace=True)
        if model_option == "PM10" and 'PM10 (ug/m3)' in input_df.columns:
            input_df.drop(columns=['PM10 (ug/m3)'], inplace=True)

        required = model.feature_names_in_
        if not all(col in input_df.columns for col in required):
            st.error("Uploaded data is missing some required columns.")
        else:
            result = predict_model(model, data=input_df)
            st.success(f"Prediction using {model_option} model completed.")
            st.dataframe(result[['Label']])
            csv = result.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, file_name=f"{model_option}_predictions.csv")
