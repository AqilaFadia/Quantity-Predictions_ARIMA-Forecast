import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your daily_data time series here
daily_data = pd.read_csv('./dataset/Transaction.csv', sep=';')  

# Mengubah kolom 'Date' menjadi tipe data datetime
daily_data['Date'] = pd.to_datetime(daily_data['Date'], format='%d/%m/%Y')

st.title('Daily Total Quantity Predictions')
st.subheader('How to predict Daily Total quantity Product selling')

# Sidebar for ARIMA Model Order Selection
st.sidebar.header("ARIMA Model Order Selection")
p = st.sidebar.slider("p (AR Order)", 0, 10, 2)
d = st.sidebar.slider("d (Differencing Order)", 0, 2, 1)
q = st.sidebar.slider("q (MA Order)", 0, 10, 1)

# Sidebar for Forecast Steps
forecast_steps = st.sidebar.slider("Number of Forecast Steps", 1, 365, 30)

def update_forecast(p, d, q, forecast_steps):
    # Build ARIMA Model
    model = ARIMA(daily_data['Qty'], order=(p, d, q))
    model_fit = model.fit()

    # Perform Forecasts
    forecast = model_fit.get_forecast(steps=forecast_steps).predicted_mean
    conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()

    # Calculate MAE and RMSE
    actual_data = daily_data['Qty'].values[-forecast_steps:]
    mae = mean_absolute_error(actual_data, forecast)
    rmse = np.sqrt(mean_squared_error(actual_data, forecast))

    # Display the forecast result using Seaborn with a black background
    st.header("ARIMA Forecast")
    st.write("Using ARIMA order (p, d, q):", (p, d, q))
    st.write("Mean Absolute Error (MAE):", mae)
    st.write("Root Mean Squared Error (RMSE):", rmse)

    # Set Matplotlib theme to a dark background
    plt.style.use('dark_background')

    # Create a Seaborn plot with a dark background
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=daily_data['Date'], y=daily_data['Qty'], label='Actual')
    forecast_dates = pd.date_range(start=daily_data['Date'].iloc[-1], periods=forecast_steps, freq='D')
    sns.lineplot(x=forecast_dates, y=forecast, label='Forecast', color='red')
    plt.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='r', alpha=0.2, label='Confidence Interval')
    plt.title('Actual vs Forecast')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    st.pyplot(plt)

# Call the function to initially display the forecast
update_forecast(p, d, q, forecast_steps)
