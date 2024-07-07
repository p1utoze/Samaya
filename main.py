import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from nixtla import NixtlaClient
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import asyncio
from dotenv import load_dotenv
import os
load_dotenv()

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


# Function to calculate SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Handle division by zero
    mask = denominator != 0
    valid_entries = numerator[mask] / denominator[mask]

    return np.mean(valid_entries) * 100

# Function to calculate MASE
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    if d == 0:
        d = 1e-6

    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

# Function to run Prophet model
def run_prophet(df, forecast_horizon):
    st.session_state.prophet.fit(df)
    future = st.session_state.prophet.make_future_dataframe(periods=forecast_horizon)
    forecast = st.session_state.prophet.predict(future)
    return forecast


async def run_timegpt(df, **kwargs):
    forecast = st.session_state.timegpt.forecast(df, **kwargs)
    return forecast


# Function to evaluate models
def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    rmae = np.sqrt(mae)
    smape = symmetric_mean_absolute_percentage_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    mase = mean_absolute_scaled_error(actual, predicted, actual[:-len(predicted)])
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase,
        "RMAE": rmae,
        'SMAPE': smape
    }


# Streamlit app
def main():
    st.set_page_config(
        page_title='Time Series Forecasting App',
        page_icon='📈',
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    st.title('Time Series Forecasting App')

    if "timegpt" not in st.session_state:
        st.session_state.timegpt = NixtlaClient(api_key=os.getenv('NIXTLA_API_KEY'))
        st.session_state.timegpt.validate_api_key()

    if "prophet" not in st.session_state:
        st.session_state.prophet = Prophet()

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        # Select date and value columns
        date_col = st.selectbox('Select Date Column', df.columns)
        value_col = st.selectbox('Select Value Column', df.columns)

        # Prepare data for models
        df['ds'] = pd.to_datetime(df[date_col])
        df['y'] = df[value_col]
        df = df[['ds', 'y']]

        # Plot original time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Original Data'))
        fig.update_layout(title='Time Series Plot', xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig)

        # Forecast horizon
        forecast_horizon = st.slider('Select Forecast Horizon', min_value=1, max_value=365, value=30)
        model_name = "timegpt-1"
        if forecast_horizon > 12:
            model_name = "timegpt-1-long-horizon"
        if st.button('Run Forecast'):
            # Run models
            with st.spinner('Running Prophet...'):
                prophet_forecast = run_prophet(df, forecast_horizon)

            with st.spinner('Running TimeGPT...'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                timegpt_forecast = loop.run_until_complete(run_timegpt(df, h=forecast_horizon, model=model_name))
                plot = st.session_state.timegpt.plot(df, timegpt_forecast, engine='plotly')

            st.plotly_chart(plot)
            st.plotly_chart(st.session_state.prophet.plot(prophet_forecast))

            # Evaluate models
            timegpt_forecast['unique_id'] = "OT"
            prophet_eval = evaluate_model(df['y'][-forecast_horizon:], prophet_forecast['yhat'][-forecast_horizon:])
            timegpt_eval = evaluate_model(df['y'][-forecast_horizon:], timegpt_forecast['TimeGPT'][-forecast_horizon:])

            # Display evaluation metrics
            st.subheader('Evaluation Metrics')
            metrics_df = pd.DataFrame({
                'Prophet': prophet_eval,
                'TimeGPT': timegpt_eval
            }).T
            st.table(metrics_df)


if __name__ == '__main__':
    main()
