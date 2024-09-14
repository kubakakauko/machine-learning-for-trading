import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
import time
import math
from scipy.stats import norm
import threading


# Function to calculate Black-Scholes Gamma
def calculate_gamma(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes Gamma of an option.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Implied volatility

    Returns:
    float: Gamma value
    """
    if T <= 0 or sigma <= 0:
        return 0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    return gamma


# Function to fetch options data and calculate GEX
def fetch_gex(ticker="SPY", r=0.01):
    """
    Fetch options data for the given ticker and calculate GEX.

    Parameters:
    ticker (str): Ticker symbol (default 'SPY')
    r (float): Risk-free interest rate (default 1%)

    Returns:
    float: Calculated GEX
    """
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
    title="Live Gamma Exposure Index (GEX) for SPY",
    xaxis_title="Time",
    yaxis_title="GEX",
    template="plotly_dark",
)

# Lists to store time and GEX values
times = []
gex_values = []


# Function to update the plot
def update_plot(fig, times, gex_values):
    fig.data = []  # Clear existing data
    fig.add_trace(
        go.Scatter(
            x=times,
            y=gex_values,
            mode="lines+markers",
            name="GEX",
            line=dict(color="cyan"),
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


# Main loop for live updates
def live_update():
    global times, gex_values, fig
    while True:
        current_time = datetime.now()
        gex = fetch_gex()

        if gex is not None:
            times.append(current_time)
            gex_values.append(gex)
            print(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} - GEX: {gex:.2f}")

            # Keep only the last 100 data points for clarity
            if len(times) > 100:
                times = times[-100:]
                gex_values = gex_values[-100:]

            update_plot(fig, times, gex_values)
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
        print("Live GEX plotting stopped.")
