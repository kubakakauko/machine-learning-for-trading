# %%
# %% [markdown]
# # Nadaraya-Watson Envelope with EMA and Trailing Stop / TP for Live and Historical Data

# %% [markdown]
# ## 1. Imports

import re
import sys
import threading
import time
from datetime import datetime, timedelta
import logging

import dash_bootstrap_components as dbc
import numpy as np

# %%
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import pytz
import requests
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots
from statsmodels.nonparametric.kernel_regression import KernelReg

# %% [markdown]
# ## 2. Configuration

# %%
# Configuration Parameters
SYMBOL = "BTCUSDT"
INTERVAL = "1m"  # Fetch 1-minute candles
AGG_INTERVAL_MINUTES = 9  # Desired timeframe to aggregate to
BANDWIDTH = 7  # Bandwidth for Nadaraya-Watson
EMA_SLOW = 50
EMA_FAST = 40
ATR_LENGTH = 14
BACKCANDLES = 7
SL_COEF = 1.5
TP_SL_RATIO = 2.0
SLIPPAGE = 0.0005
TRAIL_PERCENT = 0.00618
HISTORICAL_DAYS = 10

# Account and Risk Management
ACCOUNT_BALANCE = 10000  # Example starting balance in USD
RISK_PER_TRADE = 0.01  # Risk 1% of account per trade

MAX_TRADE_DURATION = timedelta(hours=4)  # Close trades after 4 hours

# Dash Configuration
PLOT_UPDATE_INTERVAL = 60  # seconds
DASH_APP_NAME = "Live Trading Dashboard"

# Timezone Configuration
LOCAL_TIMEZONE = pytz.timezone(
    "Europe/London"
)  # Adjust to your local timezone if different

# Configure logging
logging.basicConfig(
    filename="trading_log.log",
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
)

# Global variables and lock for synchronization
df = None
trades = None
current_trade = None
data_lock = threading.Lock()


# Function to align datetime to the nearest previous interval
def align_to_interval(dt, interval_minutes):
    """
    Align datetime to the nearest previous interval_minutes
    """
    discard = timedelta(
        minutes=dt.minute % interval_minutes,
        seconds=dt.second,
        microseconds=dt.microsecond,
    )
    return dt - discard


# %% [markdown]
# ## 3. Fetch Historical Data


# %%
def fetch_historical_data(
    symbol=SYMBOL,
    agg_interval_minutes=AGG_INTERVAL_MINUTES,
    bandwidth=BANDWIDTH,
    ema_slow=EMA_SLOW,
    ema_fast=EMA_FAST,
    atr_length=ATR_LENGTH,
):
    """
    Fetch historical data from Binance API, aggregate it into specified interval, and calculate indicators.
    """
    base_url = "https://api.binance.com"
    endpoint = f"/api/v3/klines"

    # Calculate end_time as the last complete aggregated candle
    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    end_time = align_to_interval(now, agg_interval_minutes)

    # Start time is HISTORICAL_DAYS ago, aligned to aggregation interval
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    start_time = align_to_interval(start_time, agg_interval_minutes)

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    all_candles = []
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": "1m",  # Fetch 1-minute candles
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000,  # Maximum allowed by Binance
        }

        response = requests.get(f"{base_url}{endpoint}", params=params)
        if response.status_code != 200:
            logging.error(f"Error fetching data: {response.text}")
            time.sleep(60)  # Wait before retrying
            continue

        data = response.json()

        if not data:
            break

        all_candles.extend(data)
        if len(data) < 1000:
            break

        start_ts = data[-1][0] + 1  # Start from the next candle

    df_hist_1m = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    df_hist_1m["timestamp"] = pd.to_datetime(
        df_hist_1m["timestamp"], unit="ms", utc=True
    )
    df_hist_1m.set_index("timestamp", inplace=True)

    df_hist_1m[["Open", "High", "Low", "Close", "Volume"]] = df_hist_1m[
        ["Open", "High", "Low", "Close", "Volume"]
    ].astype(float)

    # Aggregate 1-minute candles into desired interval
    df_hist = (
        df_hist_1m.resample(f"{agg_interval_minutes}T")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    # Calculate indicators
    df_hist = calculate_indicators(
        df_hist, ema_slow=ema_slow, ema_fast=ema_fast, atr_length=atr_length
    )
    df_hist = calculate_nadaraya_watson(df_hist, bandwidth=bandwidth)
    df_hist = ema_signal(df_hist, backcandles=BACKCANDLES)
    df_hist = total_signal(df_hist)

    return df_hist


# %% [markdown]
# ## 4. Fetch Candles Between Two Times


# %%
def fetch_candles(symbol, interval, start_time, end_time):
    """
    Fetch candles from Binance API between two times.
    """
    base_url = "https://api.binance.com"
    endpoint = f"/api/v3/klines"

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    all_candles = []
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000,
        }

        response = requests.get(f"{base_url}{endpoint}", params=params)
        if response.status_code != 200:
            logging.error(f"Error fetching candles: {response.text}")
            time.sleep(60)  # Wait before retrying
            continue

        data = response.json()

        if not data:
            break

        all_candles.extend(data)
        if len(data) < 1000:
            break

        start_ts = data[-1][0] + 1  # Start from the next candle

    if not all_candles:
        return None

    df_candles = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )

    df_candles["timestamp"] = pd.to_datetime(
        df_candles["timestamp"], unit="ms", utc=True
    )
    df_candles.set_index("timestamp", inplace=True)

    df_candles[["Open", "High", "Low", "Close", "Volume"]] = df_candles[
        ["Open", "High", "Low", "Close", "Volume"]
    ].astype(float)

    return df_candles


# %% [markdown]
# ## 5. Calculate Indicators: Nadaraya-Watson Envelope and EMAs


# %%
def calculate_indicators(
    df, ema_slow=EMA_SLOW, ema_fast=EMA_FAST, atr_length=ATR_LENGTH
):
    """
    Calculate technical indicators: EMA and ATR.
    """
    df["EMA_slow"] = ta.ema(df["Close"], length=ema_slow)
    df["EMA_fast"] = ta.ema(df["Close"], length=ema_fast)
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=atr_length)
    return df


def calculate_nadaraya_watson(df, bandwidth=BANDWIDTH):
    """
    Calculate Nadaraya-Watson envelopes.
    """
    df = df.copy()

    # Convert datetime index to numerical values
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values

    # Perform Nadaraya-Watson kernel regression
    model = KernelReg(endog=y, exog=X, var_type="c", bw=[bandwidth])
    fitted_values, _ = model.fit(X)

    # Store the fitted values
    df["NW_Fitted"] = fitted_values

    # Calculate the residuals
    residuals = df["Close"] - df["NW_Fitted"]

    # Calculate the standard deviation of the residuals
    std_dev = np.std(residuals)

    # Create the envelopes
    df["Upper_Envelope"] = df["NW_Fitted"] + 2 * std_dev
    df["Lower_Envelope"] = df["NW_Fitted"] - 2 * std_dev

    return df


def ema_signal(df, backcandles=BACKCANDLES):
    """
    Generate EMA crossover signals.
    """
    above = df["EMA_fast"] > df["EMA_slow"]
    below = df["EMA_fast"] < df["EMA_slow"]

    above_all = (
        above.rolling(window=backcandles)
        .apply(lambda x: x.all(), raw=True)
        .fillna(0)
        .astype(bool)
    )
    below_all = (
        below.rolling(window=backcandles)
        .apply(lambda x: x.all(), raw=True)
        .fillna(0)
        .astype(bool)
    )

    df["EMASignal"] = 0
    df.loc[above_all, "EMASignal"] = 2
    df.loc[below_all, "EMASignal"] = 1

    return df


def total_signal(df):
    """
    Generate total trading signals based on EMA and Nadaraya-Watson envelopes.
    """
    condition_buy = (df["EMASignal"] == 2) & (df["Close"] <= df["Lower_Envelope"])
    condition_sell = (df["EMASignal"] == 1) & (df["Close"] >= df["Upper_Envelope"])

    df["Total_Signal"] = 0
    df.loc[condition_buy, "Total_Signal"] = 2
    df.loc[condition_sell, "Total_Signal"] = 1

    return df


# %% [markdown]
# ## 6. Trade Tracking


# %%
class Trade:
    def __init__(self, entry_time, entry_price, trade_type, sl, tp, position_size):
        self.EntryTime = entry_time
        self.EntryPrice = float(entry_price)
        self.Type = trade_type  # 'Long' or 'Short'
        self.InitialSL = float(sl)
        self.InitialTP = float(tp)
        self.CurrentSL = float(sl)
        self.CurrentTP = float(tp)
        self.PositionSize = position_size
        self.ExitTime = None
        self.ExitPrice = None
        self.PnL = None

    def to_dict(self):
        return {
            "EntryTime": self.EntryTime,
            "EntryPrice": self.EntryPrice,
            "Type": self.Type,
            "InitialSL": self.InitialSL,
            "InitialTP": self.InitialTP,
            "CurrentSL": self.CurrentSL,
            "CurrentTP": self.CurrentTP,
            "PositionSize": self.PositionSize,
            "ExitTime": self.ExitTime,
            "ExitPrice": self.ExitPrice,
            "PnL": self.PnL,
        }


# %% [markdown]
# ## 7. Position Sizing and Risk Management


# %%
def calculate_position_size(entry_price, stop_loss_price):
    """
    Calculate the position size based on account balance and risk per trade.
    """
    amount_at_risk = ACCOUNT_BALANCE * RISK_PER_TRADE
    price_difference = abs(entry_price - stop_loss_price)
    position_size = amount_at_risk / price_difference
    return position_size


# %% [markdown]
# ## 8. Plotting the Strategy with Plotly Dash


# %%
def create_dash_app():
    """
    Create and configure the Dash app.
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H1(f"{SYMBOL} Live Trading Dashboard"),
                                className="mb-2",
                            )
                        ]
                    ),
                    dbc.Row([dbc.Col(dcc.Graph(id="live-trading-graph"), width=12)]),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Interval(
                                    id="interval-component",
                                    interval=PLOT_UPDATE_INTERVAL
                                    * 1000,  # in milliseconds
                                    n_intervals=0,
                                ),
                                width=12,
                            )
                        ]
                    ),
                    dbc.Row([dbc.Col(html.Div(id="trade-statistics"), width=12)]),
                ]
            )
        ]
    )

    @app.callback(
        [
            Output("live-trading-graph", "figure"),
            Output("trade-statistics", "children"),
        ],
        [Input("interval-component", "n_intervals")],
    )
    def update_graph_live(n):
        global df, trades, current_trade, data_lock
        with data_lock:
            if df is None or trades is None:
                return go.Figure(), html.Div("No data available.")

            df_copy = df.copy()
            trades_copy = trades.copy()

        # Recreate the figure
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.01)

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_copy.index,
                open=df_copy["Open"],
                high=df_copy["High"],
                low=df_copy["Low"],
                close=df_copy["Close"],
                name="Candlestick",
            )
        )

        # EMAs
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy["EMA_fast"],
                name="EMA Fast",
                line=dict(color="blue", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy["EMA_slow"],
                name="EMA Slow",
                line=dict(color="orange", width=1),
            )
        )

        # Nadaraya-Watson envelopes
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy["Upper_Envelope"],
                name="Upper Envelope",
                line=dict(color="rgba(255,0,0,0.3)", width=1, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_copy.index,
                y=df_copy["Lower_Envelope"],
                name="Lower Envelope",
                line=dict(color="rgba(0,255,0,0.3)", width=1, dash="dot"),
            )
        )

        # Add trade information
        for _, trade in trades_copy.iterrows():
            # Entry point
            fig.add_trace(
                go.Scatter(
                    x=[trade["EntryTime"]],
                    y=[trade["EntryPrice"]],
                    mode="markers",
                    marker=dict(
                        symbol=(
                            "triangle-up"
                            if trade["Type"] == "Long"
                            else "triangle-down"
                        ),
                        size=12,
                        color="green" if trade["Type"] == "Long" else "red",
                        line=dict(width=2, color="black"),
                    ),
                    name="Entry",
                    showlegend=False,
                )
            )

            # Only plot Exit and PnL if the trade is closed
            if pd.notnull(trade["ExitTime"]) and pd.notnull(trade["ExitPrice"]):
                # Plot Initial TP
                fig.add_shape(
                    type="line",
                    x0=trade["EntryTime"],
                    y0=trade["InitialTP"],
                    x1=trade["ExitTime"],
                    y1=trade["InitialTP"],
                    line=dict(color="rgba(0,255,0,0.5)", width=1, dash="dash"),
                )

                # Plot Initial SL
                fig.add_shape(
                    type="line",
                    x0=trade["EntryTime"],
                    y0=trade["InitialSL"],
                    x1=trade["ExitTime"],
                    y1=trade["InitialSL"],
                    line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"),
                )

                # Trailing SL line
                fig.add_trace(
                    go.Scatter(
                        x=[trade["EntryTime"], trade["ExitTime"]],
                        y=[trade["InitialSL"], trade["CurrentSL"]],
                        mode="lines",
                        line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dot"),
                        name="Trailing SL",
                        showlegend=False,
                    )
                )

                # Exit point
                exit_color = "green" if trade["PnL"] > 0 else "red"
                fig.add_trace(
                    go.Scatter(
                        x=[trade["ExitTime"]],
                        y=[trade["ExitPrice"]],
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=12,
                            color=exit_color,
                            line=dict(width=2, color="black"),
                        ),
                        name="Exit",
                        showlegend=False,
                    )
                )

                # Annotate P/L at Exit Price
                fig.add_annotation(
                    x=trade["ExitTime"],
                    y=trade["ExitPrice"],
                    text=f"P/L: {trade['PnL']:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=exit_color,
                    ax=20,
                    ay=-40,
                    bordercolor=exit_color,
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="white",
                    opacity=0.8,
                )

        # Update layout
        fig.update_layout(
            title=f"{SYMBOL} Chart with Signals and Trade Information",
            xaxis_title="Date",
            yaxis_title="Price (USDT)",
            xaxis_rangeslider_visible=False,
            legend_title="Legend",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=800,
        )

        # Trade Statistics
        if not trades_copy.empty:
            total_trades = len(trades_copy)
            winning_trades = len(trades_copy[trades_copy["PnL"] > 0])
            losing_trades = len(trades_copy[trades_copy["PnL"] <= 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            avg_profit = trades_copy["PnL"].mean()
            avg_profit_winners = (
                trades_copy[trades_copy["PnL"] > 0]["PnL"].mean()
                if winning_trades > 0
                else 0
            )
            avg_loss_losers = (
                trades_copy[trades_copy["PnL"] <= 0]["PnL"].mean()
                if losing_trades > 0
                else 0
            )
            profit_factor = (
                (
                    trades_copy[trades_copy["PnL"] > 0]["PnL"].sum()
                    / abs(trades_copy[trades_copy["PnL"] <= 0]["PnL"].sum())
                )
                if losing_trades > 0
                else np.inf
            )

            stats_html = html.Div(
                [
                    html.H4("Trade Statistics"),
                    html.P(f"Total Trades: {total_trades}"),
                    html.P(f"Winning Trades: {winning_trades}"),
                    html.P(f"Losing Trades: {losing_trades}"),
                    html.P(f"Win Rate: {win_rate:.2f}%"),
                    html.P(f"Average Profit: {avg_profit:.2f}"),
                    html.P(f"Average Profit (Winners): {avg_profit_winners:.2f}"),
                    html.P(f"Average Loss (Losers): {avg_loss_losers:.2f}"),
                    html.P(f"Profit Factor: {profit_factor:.2f}"),
                ]
            )
        else:
            stats_html = html.Div(
                [html.H4("Trade Statistics"), html.P("No trades yet.")]
            )

        return fig, stats_html

    return app


# %% [markdown]
# ## 9. Advanced Trailing Stop


# %%
def update_trailing_stop(current_trade, current_price, latest_row):
    """
    Update the trailing stop based on ATR.
    """
    atr = latest_row["ATR"]
    if current_trade.Type == "Long":
        new_sl = max(current_trade.CurrentSL, current_price - SL_COEF * atr)
        current_trade.CurrentSL = new_sl
    else:
        new_sl = min(current_trade.CurrentSL, current_price + SL_COEF * atr)
        current_trade.CurrentSL = new_sl


# %% [markdown]
# ## 10. Main Execution Loop


# %%
def main():
    global df, trades, current_trade, data_lock, ACCOUNT_BALANCE
    # Initialize DataFrame with historical data
    df_hist = fetch_historical_data(symbol=SYMBOL)
    with data_lock:
        df = df_hist.copy()
        trades = pd.DataFrame(
            columns=[
                "EntryTime",
                "EntryPrice",
                "Type",
                "InitialSL",
                "InitialTP",
                "CurrentSL",
                "CurrentTP",
                "PositionSize",
                "ExitTime",
                "ExitPrice",
                "PnL",
            ]
        )

    logging.info(
        f"Fetched {len(df)} candles from {df.index[0].astimezone(LOCAL_TIMEZONE)} to {df.index[-1].astimezone(LOCAL_TIMEZONE)}"
    )

    # Initialize current trade
    current_trade = None
    last_known_time = df.index[-1]

    # Process historical data to simulate trades
    logging.info("Processing historical data to simulate trades...")
    for idx, row in df.iterrows():
        signal = row["Total_Signal"]

        with data_lock:
            if current_trade is None:
                if signal == 2:  # Buy signal
                    entry_price = row["Close"] * (1 + SLIPPAGE)
                    sl_distance = row["ATR"] * SL_COEF
                    tp_distance = sl_distance * TP_SL_RATIO
                    sl = entry_price - sl_distance
                    tp = entry_price + tp_distance
                    position_size = calculate_position_size(entry_price, sl)

                    current_trade = Trade(
                        entry_time=idx,
                        entry_price=entry_price,
                        trade_type="Long",
                        sl=sl,
                        tp=tp,
                        position_size=position_size,
                    )
                    logging.info(
                        f"Historical Entry Long at {entry_price:.2f} on {idx.astimezone(LOCAL_TIMEZONE)} with position size {position_size:.6f}"
                    )
                elif signal == 1:  # Sell signal
                    entry_price = row["Close"] * (1 - SLIPPAGE)
                    sl_distance = row["ATR"] * SL_COEF
                    tp_distance = sl_distance * TP_SL_RATIO
                    sl = entry_price + sl_distance
                    tp = entry_price - tp_distance
                    position_size = calculate_position_size(entry_price, sl)

                    current_trade = Trade(
                        entry_time=idx,
                        entry_price=entry_price,
                        trade_type="Short",
                        sl=sl,
                        tp=tp,
                        position_size=position_size,
                    )
                    logging.info(
                        f"Historical Entry Short at {entry_price:.2f} on {idx.astimezone(LOCAL_TIMEZONE)} with position size {position_size:.6f}"
                    )
            else:
                # Update Trailing SL
                current_price = row["Close"]
                update_trailing_stop(current_trade, current_price, row)

                # Check for exit conditions
                exit_price = None
                if current_trade.Type == "Long":
                    if current_price <= current_trade.CurrentSL:
                        exit_price = current_trade.CurrentSL
                    elif current_price >= current_trade.InitialTP:
                        exit_price = current_trade.InitialTP
                else:
                    if current_price >= current_trade.CurrentSL:
                        exit_price = current_trade.CurrentSL
                    elif current_price <= current_trade.InitialTP:
                        exit_price = current_trade.InitialTP

                # Time-based exit
                trade_duration = idx - current_trade.EntryTime
                if trade_duration >= MAX_TRADE_DURATION:
                    exit_price = current_price
                    logging.info(
                        f"Time-based exit for {current_trade.Type} trade at {exit_price:.2f}"
                    )

                if exit_price is not None:
                    current_trade.ExitTime = idx
                    current_trade.ExitPrice = exit_price  # Ensure ExitPrice is set
                    if current_trade.Type == "Long":
                        current_trade.PnL = (
                            exit_price - current_trade.EntryPrice
                        ) * current_trade.PositionSize
                    else:
                        current_trade.PnL = (
                            current_trade.EntryPrice - exit_price
                        ) * current_trade.PositionSize

                    # Update account balance
                    ACCOUNT_BALANCE += current_trade.PnL

                    trade_dict = current_trade.to_dict()
                    trades = pd.concat(
                        [trades, pd.DataFrame([trade_dict])], ignore_index=True
                    )
                    logging.info(
                        f"Historical Exit {current_trade.Type} at {exit_price:.2f} on {idx.astimezone(LOCAL_TIMEZONE)} with PnL: {current_trade.PnL:.2f}"
                    )
                    current_trade = None

    # Create Dash app
    app = create_dash_app()
    logging.info("Initial historical trades processed.")

    # Start Dash app in a separate thread
    def run_dash():
        app.run_server(debug=False, use_reloader=False)

    dash_thread = threading.Thread(target=run_dash)
    dash_thread.daemon = True  # Ensures the thread exits when main thread does
    dash_thread.start()
    logging.info(f"Dash app running at http://127.0.0.1:8050/")

    # Start the live loop
    while True:
        try:
            now = datetime.now(pytz.UTC)  # Make 'now' timezone-aware
            next_agg_candle_time = last_known_time + timedelta(
                minutes=AGG_INTERVAL_MINUTES
            )
            time_until_next_candle = (next_agg_candle_time - now).total_seconds()

            if time_until_next_candle > 0:
                # Sleep until the next aggregated candle time
                sleep_duration = min(
                    time_until_next_candle + 1, 60
                )  # Avoid sleeping too long
                logging.info(
                    f"Sleeping for {sleep_duration:.2f} seconds until next candle."
                )
                time.sleep(sleep_duration)
                continue

            # Fetch 1-minute candles from last_known_time + 1 minute to next_agg_candle_time
            start_time = last_known_time + timedelta(minutes=1)
            end_time = next_agg_candle_time

            df_new_1m = fetch_candles(
                symbol=SYMBOL, interval="1m", start_time=start_time, end_time=end_time
            )
            if df_new_1m is None or df_new_1m.empty:
                logging.info(
                    f"No new candles fetched between {start_time} and {end_time}"
                )
                time.sleep(60)  # Wait before trying again
                continue

            # Aggregate df_new_1m into aggregated candle
            df_new_agg = (
                df_new_1m.resample(f"{AGG_INTERVAL_MINUTES}T")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna()
            )

            if df_new_agg.empty:
                logging.info(
                    f"No new aggregated candle formed between {start_time} and {end_time}"
                )
                time.sleep(60)
                continue

            # Append the new aggregated candle to df
            with data_lock:
                df = pd.concat([df, df_new_agg])

                # Recalculate indicators
                df = calculate_indicators(
                    df, ema_slow=EMA_SLOW, ema_fast=EMA_FAST, atr_length=ATR_LENGTH
                )
                df = calculate_nadaraya_watson(df, bandwidth=BANDWIDTH)
                df = ema_signal(df, backcandles=BACKCANDLES)
                df = total_signal(df)

                # Get the latest row
                latest_row = df.iloc[-1]
                last_known_time = latest_row.name

            # Generate signals
            signal = latest_row["Total_Signal"]
            logging.info(
                f"Signal at {last_known_time.astimezone(LOCAL_TIMEZONE)}: {signal}"
            )

            # Trade Management
            with data_lock:
                if current_trade is None:
                    if signal == 2:  # Buy signal
                        entry_price = latest_row["Close"] * (1 + SLIPPAGE)
                        sl_distance = latest_row["ATR"] * SL_COEF
                        tp_distance = sl_distance * TP_SL_RATIO
                        sl = entry_price - sl_distance
                        tp = entry_price + tp_distance
                        position_size = calculate_position_size(entry_price, sl)

                        current_trade = Trade(
                            entry_time=last_known_time,
                            entry_price=entry_price,
                            trade_type="Long",
                            sl=sl,
                            tp=tp,
                            position_size=position_size,
                        )
                        logging.info(
                            f"Entered Long at {entry_price:.2f} with SL: {sl:.2f} and TP: {tp:.2f} on {last_known_time.astimezone(LOCAL_TIMEZONE)} with position size {position_size:.6f}"
                        )

                    elif signal == 1:  # Sell signal
                        entry_price = latest_row["Close"] * (1 - SLIPPAGE)
                        sl_distance = latest_row["ATR"] * SL_COEF
                        tp_distance = sl_distance * TP_SL_RATIO
                        sl = entry_price + sl_distance
                        tp = entry_price - tp_distance
                        position_size = calculate_position_size(entry_price, sl)

                        current_trade = Trade(
                            entry_time=last_known_time,
                            entry_price=entry_price,
                            trade_type="Short",
                            sl=sl,
                            tp=tp,
                            position_size=position_size,
                        )
                        logging.info(
                            f"Entered Short at {entry_price:.2f} with SL: {sl:.2f} and TP: {tp:.2f} on {last_known_time.astimezone(LOCAL_TIMEZONE)} with position size {position_size:.6f}"
                        )
                else:
                    # Update Trailing SL
                    current_price = latest_row["Close"]
                    update_trailing_stop(current_trade, current_price, latest_row)

                    # Check for exit conditions
                    exit_price = None
                    if current_trade.Type == "Long":
                        if current_price <= current_trade.CurrentSL:
                            exit_price = current_trade.CurrentSL
                        elif current_price >= current_trade.InitialTP:
                            exit_price = current_trade.InitialTP
                    else:
                        if current_price >= current_trade.CurrentSL:
                            exit_price = current_trade.CurrentSL
                        elif current_price <= current_trade.InitialTP:
                            exit_price = current_trade.InitialTP

                    # Time-based exit
                    trade_duration = last_known_time - current_trade.EntryTime
                    if trade_duration >= MAX_TRADE_DURATION:
                        exit_price = current_price
                        logging.info(
                            f"Time-based exit for {current_trade.Type} trade at {exit_price:.2f}"
                        )

                    if exit_price is not None:
                        current_trade.ExitTime = last_known_time
                        current_trade.ExitPrice = exit_price  # Ensure ExitPrice is set
                        if current_trade.Type == "Long":
                            current_trade.PnL = (
                                exit_price - current_trade.EntryPrice
                            ) * current_trade.PositionSize
                        else:
                            current_trade.PnL = (
                                current_trade.EntryPrice - exit_price
                            ) * current_trade.PositionSize

                        # Update account balance
                        ACCOUNT_BALANCE += current_trade.PnL

                        trade_dict = current_trade.to_dict()
                        trades = pd.concat(
                            [trades, pd.DataFrame([trade_dict])], ignore_index=True
                        )
                        logging.info(
                            f"Exited {current_trade.Type} at {exit_price:.2f} on {last_known_time.astimezone(LOCAL_TIMEZONE)} with PnL: {current_trade.PnL:.2f}"
                        )
                        current_trade = None

        except KeyboardInterrupt:
            logging.info("Live trading stopped by user.")
            sys.exit()
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            time.sleep(60)  # Wait before retrying


# %% [markdown]
# ## **End of Script**

# %%
if __name__ == "__main__":
    main()
