import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
import time
import math
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import threading


# Function to calculate Black-Scholes Gamma
def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    return gamma


# Function to fetch options data and calculate GEX
def fetch_gex(ticker="SPY", r=0.01):
    try:
        stock = yf.Ticker(ticker)
        S = stock.history(period="1d")["Close"][0]
        expirations = stock.options
        gex = 0

        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            T = (exp_date - datetime.now()).days / 365.25
            if T <= 0:
                continue
            opt = stock.option_chain(exp)
            for option in opt.calls.itertuples():
                K = option.strike
                sigma = option.impliedVolatility
                if sigma is None or np.isnan(sigma):
                    continue
                gamma = calculate_gamma(S, K, T, r, sigma)
                gex += gamma * option.openInterest

            for option in opt.puts.itertuples():
                K = option.strike
                sigma = option.impliedVolatility
                if sigma is None or np.isnan(sigma):
                    continue
                gamma = calculate_gamma(S, K, T, r, sigma)
                gex += gamma * option.openInterest

        return gex
    except Exception as e:
        print(f"Error fetching GEX: {e}")
        return None


# Initialize Plotly figure
fig = go.Figure()
fig.update_layout(
    title="Live Gamma Exposure Index (GEX) for SPY with ML Predictions",
    xaxis_title="Time",
    yaxis_title="GEX",
    template="plotly_dark",
)

# Lists to store time, GEX values, and predicted GEX
times = []
gex_values = []
predicted_gex = []
signals = []

# Placeholder for ML model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model_trained = False


# Function to update the plot
def update_plot(fig, times, gex_values, predicted_gex, signals):
    fig.data = []  # Clear existing data
    fig.add_trace(
        go.Scatter(
            x=times,
            y=gex_values,
            mode="lines+markers",
            name="Actual GEX",
            line=dict(color="cyan"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=predicted_gex,
            mode="lines",
            name="Predicted GEX",
            line=dict(color="orange"),
        )
    )

    # Highlight buy/sell signals
    buy_times = [t for t, s in zip(times, signals) if s == "Buy"]
    buy_values = [g for g, s in zip(gex_values, signals) if s == "Buy"]
    sell_times = [t for t, s in zip(times, signals) if s == "Sell"]
    sell_values = [g for g, s in zip(gex_values, signals) if s == "Sell"]

    fig.add_trace(
        go.Scatter(
            x=buy_times,
            y=buy_values,
            mode="markers",
            marker=dict(color="green", size=10, symbol="triangle-up"),
            name="Buy Signal",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sell_times,
            y=sell_values,
            mode="markers",
            marker=dict(color="red", size=10, symbol="triangle-down"),
            name="Sell Signal",
        )
    )

    fig.update_layout(
        xaxis=dict(
            range=[
                min(times) if times else 0,
                max(times) + timedelta(minutes=1) if times else 1,
            ]
        )
    )
    fig.show(renderer="browser")


# Function to train ML model
def train_ml_model(data):
    global model, model_trained
    if len(data) < 50:
        # Not enough data to train
        return

    df = pd.DataFrame(data, columns=["GEX"])
    df["GEX_prev"] = df["GEX"].shift(1)
    df["GEX_diff"] = df["GEX"] - df["GEX_prev"]
    df = df.dropna()

    X = df[["GEX_prev"]]
    y = df["GEX"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"ML Model trained. MSE on test set: {mse:.2f}")
    model_trained = True


# Function to generate trading signals based on predictions
def generate_signal(actual, predicted):
    if not model_trained:
        return None
    if predicted > actual:
        return "Buy"
    elif predicted < actual:
        return "Sell"
    else:
        return None


# Main loop for live updates
def live_update():
    global times, gex_values, predicted_gex, signals, model_trained
    data_history = []

    while True:
        current_time = datetime.now()
        gex = fetch_gex()

        if gex is not None:
            times.append(current_time)
            gex_values.append(gex)
            data_history.append(gex)
            print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} - GEX: {gex:.2f}")

            # Train ML model with historical data
            train_ml_model(data_history)

            # Predict next GEX
            if model_trained and len(data_history) >= 2:
                last_gex = data_history[-1]
                predicted = model.predict([[last_gex]])[0]
                predicted_gex.append(predicted)

                # Generate trading signal
                signal = generate_signal(last_gex, predicted)
                signals.append(signal)
            else:
                predicted_gex.append(None)
                signals.append(None)

            # Keep only the last 100 data points for clarity
            if len(times) > 100:
                times = times[-100:]
                gex_values = gex_values[-100:]
                predicted_gex = predicted_gex[-100:]
                signals = signals[-100:]

            update_plot(fig, times, gex_values, predicted_gex, signals)
        else:
            print("Failed to fetch GEX data.")

        # Wait for 60 seconds before next update
        time.sleep(60)


if __name__ == "__main__":
    # Run live update in a separate thread
    thread = threading.Thread(target=live_update, daemon=True)
    thread.start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Live GEX with ML plotting stopped.")
