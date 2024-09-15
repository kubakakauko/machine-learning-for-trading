# %%
# %% [markdown]
# # Nadaraya-Watson Envelope with EMA and Trailing Stop / TP for Live and Historical Data

# %% [markdown]
# ## 1. Imports

import logging
import sys
import threading
import time
from datetime import datetime, timedelta

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
AGG_INTERVAL_MINUTES = 5  # Desired timeframe to aggregate to

# Standard intervals mapping to Binance intervals
BINANCE_INTERVALS = {
    1: "1m",
    2: "2m",
    3: "3m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "1h",
    120: "2h",
    240: "4h",
    360: "6h",
    720: "12h",
    1440: "1d",
}

BANDWIDTH = 7  # Bandwidth for Nadaraya-Watson
EMA_SLOW = 50
EMA_FAST = 40
ATR_LENGTH = 14
BACKCANDLES = 7
SL_COEF = 1.5
TP_SL_RATIO = 2.0
SLIPPAGE = 0.0005
NW_MULT = 2

TRAIL_PERCENT = 0.00618  # Trailing percentage for both SL and TP

HISTORICAL_DAYS = 20

# Account and Risk Management
ACCOUNT_BALANCE = 100  # Example starting balance in USD
RISK_PER_TRADE = 0.01  # Risk 1% of account per trade

MAX_TRADE_DURATION = timedelta(hours=6)  # Close trades after 6 hours

# Strategy Toggles
ENABLE_VOLATILITY_STRATEGY = False
ENABLE_DCA = True  # Dollar Cost Averaging
DCA_THRESHOLD_PERCENT = 0.005  # Threshold to add to position (0.5%)
MAX_DCA_ATTEMPTS = 3  # Maximum number of times to apply DCA

# Dash Configuration
PLOT_UPDATE_INTERVAL = 30  # seconds
DASH_APP_NAME = "Live Trading Dashboard"

# Timezone Configuration
LOCAL_TIMEZONE = pytz.timezone(
    "Europe/London"
)  # Adjust to your local timezone if different

# Configure logging to output to both file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")

# File handler
file_handler = logging.FileHandler("trading_log.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Global variables and lock for synchronization
df = None
trades = None
live_trades = None  # Separate log for live trades
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

    # Determine if we can fetch data directly at the desired interval
    if agg_interval_minutes in BINANCE_INTERVALS:
        interval_str = BINANCE_INTERVALS[agg_interval_minutes]
        use_aggregation = False
    else:
        interval_str = "1m"
        use_aggregation = True

    # Calculate end_time as the last complete candle
    now = datetime.utcnow().replace(tzinfo=pytz.UTC)
    end_time = align_to_interval(
        now - timedelta(milliseconds=1), agg_interval_minutes
    ) + timedelta(minutes=agg_interval_minutes)

    # Start time is HISTORICAL_DAYS ago, aligned to aggregation interval
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    start_time = align_to_interval(start_time, agg_interval_minutes)

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    all_candles = []
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval_str,
            "startTime": start_ts,
            "endTime": end_ts - 1,  # Binance's endTime is exclusive
            "limit": 1000,  # Maximum allowed by Binance
        }

        response = requests.get(f"{base_url}{endpoint}", params=params)
        if response.status_code != 200:
            logger.error(f"Error fetching data: {response.text}")
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
        logger.error("No historical data fetched.")
        sys.exit(1)

    df_hist = pd.DataFrame(
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

    df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"], unit="ms", utc=True)
    df_hist.set_index("timestamp", inplace=True)

    df_hist[["Open", "High", "Low", "Close", "Volume"]] = df_hist[
        ["Open", "High", "Low", "Close", "Volume"]
    ].astype(float)

    if use_aggregation:
        # Aggregate candles into desired interval
        df_hist = (
            df_hist.resample(f"{agg_interval_minutes}min")
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
    if ENABLE_VOLATILITY_STRATEGY:
        df_hist = volatility_signal(df_hist)
    df_hist = total_signal(df_hist)

    return df_hist


# %% [markdown]
# ## 4. Fetch Candles Between Two Times


# %%
def fetch_candles(symbol, interval, start_time, end_time, agg_interval_minutes):
    """
    Fetch candles from Binance API between two times.
    """
    base_url = "https://api.binance.com"
    endpoint = f"/api/v3/klines"

    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    current_ts = int(datetime.utcnow().timestamp() * 1000)

    # Ensure end_ts does not exceed current time
    end_ts = min(end_ts, current_ts)

    all_candles = []
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts - 1,  # Binance's endTime is exclusive
            "limit": 1000,
        }

        response = requests.get(f"{base_url}{endpoint}", params=params)
        if response.status_code != 200:
            logger.error(f"Error fetching candles: {response.text}")
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
# ## 5. Calculate Indicators: Nadaraya-Watson Envelope, EMAs, and Volatility Signals


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


def calculate_nadaraya_watson(df, bandwidth=BANDWIDTH, mult=NW_MULT):
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
    df["Upper_Envelope"] = df["NW_Fitted"] + mult * std_dev
    df["Lower_Envelope"] = df["NW_Fitted"] - mult * std_dev

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


def volatility_signal(df):
    """
    Generate volatility-based signals using Bollinger Bands.
    """
    df["BB_Middle"] = ta.sma(df["Close"], length=20)
    df["BB_Upper"], df["BB_Lower"] = ta.bbands(
        df["Close"], length=20, std=2, mamode="sma"
    )[["BBU_20_2.0", "BBL_20_2.0"]].T.values

    df["VolatilitySignal"] = 0
    # Buy when price crosses above lower Bollinger Band
    df.loc[
        (df["Close"].shift(1) < df["BB_Lower"].shift(1))
        & (df["Close"] > df["BB_Lower"]),
        "VolatilitySignal",
    ] = 2
    # Sell when price crosses below upper Bollinger Band
    df.loc[
        (df["Close"].shift(1) > df["BB_Upper"].shift(1))
        & (df["Close"] < df["BB_Upper"]),
        "VolatilitySignal",
    ] = 1
    return df


def total_signal(df):
    """
    Generate total trading signals based on EMA, Nadaraya-Watson envelopes, and volatility signals.
    """
    condition_buy = (df["EMASignal"] == 2) & (df["Close"] <= df["Lower_Envelope"])
    condition_sell = (df["EMASignal"] == 1) & (df["Close"] >= df["Upper_Envelope"])

    if ENABLE_VOLATILITY_STRATEGY:
        condition_buy &= df["VolatilitySignal"] == 2
        condition_sell &= df["VolatilitySignal"] == 1

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
        self.DCA_Attempts = 0  # Number of DCA attempts
        self.Live = True  # Indicates if the trade is live or historical

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
            "Live": self.Live,
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

                # Trailing TP line
                fig.add_trace(
                    go.Scatter(
                        x=[trade["EntryTime"], trade["ExitTime"]],
                        y=[trade["InitialTP"], trade["CurrentTP"]],
                        mode="lines",
                        line=dict(color="rgba(0,255,0,0.5)", width=1, dash="dot"),
                        name="Trailing TP",
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
                    html.P(f"Account Balance: {ACCOUNT_BALANCE:.2f} USD"),
                ]
            )
        else:
            stats_html = html.Div(
                [html.H4("Trade Statistics"), html.P("No trades yet.")]
            )

        return fig, stats_html

    return app


# %% [markdown]
# ## 9. Advanced Trailing Stop, Take Profit, and Dollar Cost Averaging


# %%
def update_trailing_levels(current_trade, current_price, latest_row):
    """
    Update the trailing stop and take profit based on TRAIL_PERCENT.
    """
    if current_trade.Type == "Long":
        # Update trailing stop loss (increase SL)
        new_sl = max(current_trade.CurrentSL, current_price * (1 - TRAIL_PERCENT))
        current_trade.CurrentSL = new_sl

        # Update trailing take profit (increase TP)
        new_tp = max(current_trade.CurrentTP, current_price * (1 + TRAIL_PERCENT))
        current_trade.CurrentTP = new_tp
    else:
        # Update trailing stop loss (decrease SL)
        new_sl = min(current_trade.CurrentSL, current_price * (1 + TRAIL_PERCENT))
        current_trade.CurrentSL = new_sl

        # Update trailing take profit (decrease TP)
        new_tp = min(current_trade.CurrentTP, current_price * (1 - TRAIL_PERCENT))
        current_trade.CurrentTP = new_tp


def apply_dca(current_trade, current_price):
    """
    Apply Dollar Cost Averaging if the price moves against the trade.
    """
    if current_trade.DCA_Attempts >= MAX_DCA_ATTEMPTS:
        return  # Maximum DCA attempts reached

    price_movement = (
        current_price - current_trade.EntryPrice
    ) / current_trade.EntryPrice
    if current_trade.Type == "Long" and price_movement <= -DCA_THRESHOLD_PERCENT:
        # Price has moved down by threshold percentage
        logger.info(f"Applying DCA for Long trade at price {current_price:.2f}")
        current_trade.EntryPrice = (
            current_trade.EntryPrice * current_trade.PositionSize
            + current_price * current_trade.PositionSize
        ) / (2 * current_trade.PositionSize)
        current_trade.PositionSize *= 2  # Double the position size
        current_trade.DCA_Attempts += 1

    elif current_trade.Type == "Short" and price_movement >= DCA_THRESHOLD_PERCENT:
        # Price has moved up by threshold percentage
        logger.info(f"Applying DCA for Short trade at price {current_price:.2f}")
        current_trade.EntryPrice = (
            current_trade.EntryPrice * current_trade.PositionSize
            + current_price * current_trade.PositionSize
        ) / (2 * current_trade.PositionSize)
        current_trade.PositionSize *= 2  # Double the position size
        current_trade.DCA_Attempts += 1


# %% [markdown]
# ## 10. Main Execution Loop


# %%
def main():
    global df, trades, live_trades, current_trade, data_lock, ACCOUNT_BALANCE
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
                "Live",
            ]
        )
        live_trades = pd.DataFrame(columns=trades.columns)

    logger.info(
        f"Fetched {len(df)} candles from {df.index[0].astimezone(LOCAL_TIMEZONE)} to {df.index[-1].astimezone(LOCAL_TIMEZONE)}"
    )

    # Initialize current trade
    current_trade = None
    last_known_time = df.index[-1]

    # Process historical data to simulate trades
    logger.info("Processing historical data to simulate trades...")
    for idx, row in df.iterrows():
        # (Processing code remains the same)
        pass  # Include your historical data processing logic here

    # Create Dash app
    app = create_dash_app()
    logger.info("Initial historical trades processed.")

    # Start Dash app in a separate thread
    def run_dash():
        app.run_server(debug=True, use_reloader=False)

    dash_thread = threading.Thread(target=run_dash)
    dash_thread.daemon = True  # Ensures the thread exits when main thread does
    dash_thread.start()
    logger.info(f"Dash app running at http://127.0.0.1:8050/")

    # Start the live loop
    while True:
        try:
            now = datetime.utcnow().replace(tzinfo=pytz.UTC)
            # Align end_time to the last completed candle
            end_time = align_to_interval(
                now - timedelta(milliseconds=1), AGG_INTERVAL_MINUTES
            ) + timedelta(minutes=AGG_INTERVAL_MINUTES)

            if end_time <= last_known_time:
                # No new candles yet
                next_candle_time = end_time + timedelta(minutes=AGG_INTERVAL_MINUTES)
                time_until_next_candle = (next_candle_time - now).total_seconds()
                sleep_duration = min(time_until_next_candle + 1, 60)
                logger.info(
                    f"No new candles yet. Sleeping for {sleep_duration:.2f} seconds."
                )
                time.sleep(sleep_duration)
                continue

            start_time = last_known_time + timedelta(milliseconds=1)

            logger.info(f"Fetching new candles from {start_time} to {end_time}")

            # Determine if we can fetch data directly at the desired interval
            if AGG_INTERVAL_MINUTES in BINANCE_INTERVALS:
                interval_str = BINANCE_INTERVALS[AGG_INTERVAL_MINUTES]
                use_aggregation = False
            else:
                interval_str = "1m"
                use_aggregation = True

            df_new = fetch_candles(
                symbol=SYMBOL,
                interval=interval_str,
                start_time=start_time,
                end_time=end_time,
                agg_interval_minutes=AGG_INTERVAL_MINUTES,
            )

            if df_new is None or df_new.empty:
                logger.info(
                    f"No new candles fetched between {start_time} and {end_time}"
                )
                time.sleep(60)  # Wait before trying again
                continue

            if use_aggregation:
                # Aggregate df_new into aggregated candle
                df_new = (
                    df_new.resample(f"{AGG_INTERVAL_MINUTES}min")
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

            if df_new.empty:
                logger.info(
                    f"No new aggregated candle formed between {start_time} and {end_time}"
                )
                time.sleep(60)
                continue

            # Append the new candle(s) to df
            with data_lock:
                df = pd.concat([df, df_new])
                df = df[~df.index.duplicated(keep="last")]  # Remove duplicates if any
                # Recalculate indicators
                df = calculate_indicators(
                    df, ema_slow=EMA_SLOW, ema_fast=EMA_FAST, atr_length=ATR_LENGTH
                )
                df = calculate_nadaraya_watson(df, bandwidth=BANDWIDTH)
                df = ema_signal(df, backcandles=BACKCANDLES)
                if ENABLE_VOLATILITY_STRATEGY:
                    df = volatility_signal(df)
                df = total_signal(df)

                # Get the latest row
                latest_row = df.iloc[-1]
                last_known_time = latest_row.name

            # Print new candle and debugging data
            logger.info(
                f"New candle added at {last_known_time.astimezone(LOCAL_TIMEZONE)}"
            )
            logger.info(f"Candle data: {latest_row.to_dict()}")

            # Generate signals
            signal = latest_row["Total_Signal"]
            logger.info(
                f"Signal at {last_known_time.astimezone(LOCAL_TIMEZONE)}: {signal}"
            )

            # Trade Management
            # (Trade management code remains the same)

        except KeyboardInterrupt:
            logger.info("Live trading stopped by user.")
            sys.exit()
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            time.sleep(60)  # Wait before retrying


# %% [markdown]
# ## **End of Script**

# %%
if __name__ == "__main__":
    main()
