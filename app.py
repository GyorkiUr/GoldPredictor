import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the trained model
model_path = "lstm_gold_price_model.keras"
model = load_model(model_path)

# Define a function to fetch data from Polygon.io
def fetch_data(api_key, start_date, end_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/C:XAUUSD/range/1/day/{start_date}/{end_date}?adjusted=false&sort=asc&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.rename(columns={'c': 'Close'}, inplace=True)
            df = df[['Date', 'Close']]
            return df
        else:
            st.error("No data available for the given range.")
            return None
    else:
        st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
        return None

# Streamlit app
st.title("Gold Price Prediction")

# API key input
api_key = st.text_input("Enter your Polygon.io API Key:")

if api_key:
    # Date inputs
    start_date = st.date_input("Start Date", datetime(2024, 9, 20))
    end_date = st.date_input("End Date", datetime.today() - timedelta(days=1))

    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
    elif st.button("Fetch and Predict"):
        # Adjust start date for 120-day window
        days_needed = 120  # Minimum days for the prediction window
        adjusted_start_date = start_date - timedelta(days=days_needed)

        st.info(f"Fetching data from {adjusted_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} to ensure sufficient data for predictions.")
        data = fetch_data(api_key, adjusted_start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if data is not None:
            st.write(f"Fetched {len(data)} days of data.")

            if len(data) < 120:
                st.error("Not enough data even after adjusting the start date. Please choose a longer date range.")
            else:
                # Prepare data for prediction
                data['Scaled_Close'] = data['Close'] / data['Close'].max()

                # Prepare sequences for the entire fetched data
                sequence = []
                for i in range(120, len(data)):
                    sequence.append(data['Scaled_Close'].iloc[i-120:i].values)
                X = np.array(sequence).reshape(len(sequence), 120, 1)

                st.write(f"Input shape for prediction: {X.shape}")

                if X.size > 0:
                    try:
                        # Make predictions
                        predictions = model.predict(X)
                        predictions = predictions.flatten() * data['Close'].max()

                        # Add predictions to data
                        prediction_start_index = 120  # The first prediction corresponds to the 120th day
                        data.loc[data.index[prediction_start_index:], 'Predicted_Close'] = predictions

                        # Filter data for the selected date range
                        filtered_data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

                        st.write("### Actual vs Predicted", filtered_data[['Date', 'Close', 'Predicted_Close']])

                        # Calculate metrics for the selected date range
                        mse = mean_squared_error(filtered_data['Close'], filtered_data['Predicted_Close'])
                        mape = mean_absolute_percentage_error(filtered_data['Close'], filtered_data['Predicted_Close'])
                        accuracy = 1 - mape

                        st.write("### Metrics")
                        st.metric("Mean Squared Error (MSE)", round(mse, 2))
                        st.metric("Mean Absolute Percentage Error (MAPE)", round(mape * 100, 2))
                        st.metric("Accuracy", round(accuracy * 100, 2))

                        # Plot actual vs predicted
                        st.write("### Plot: Actual vs Predicted")
                        plt.figure(figsize=(10, 6))
                        plt.plot(filtered_data['Date'], filtered_data['Close'], label='Actual', marker='o')
                        plt.plot(filtered_data['Date'], filtered_data['Predicted_Close'], label='Predicted', marker='x')
                        plt.legend()
                        plt.title("Actual vs Predicted Closing Prices")
                        plt.xlabel("Date")
                        plt.ylabel("Price")
                        plt.grid(True)
                        st.pyplot(plt)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                else:
                    st.error("Input data for prediction is empty. Please check your data preparation.")
