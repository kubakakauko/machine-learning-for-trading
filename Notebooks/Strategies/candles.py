import ccxt
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Initialize exchange (Binance in this case)
exchange = ccxt.binance()


# Function to fetch BTC/USDT candles
def fetch_candles(symbol, timeframe, limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# Initialize app
app = Dash(__name__)

# Global variable to store the candle data
candles_df = pd.DataFrame()

# Layout of the Dash app
app.layout = html.Div(
    [
        html.H1("Real-Time BTC/USDT Candlestick Chart"),
        dcc.Dropdown(
            id="timeframe-dropdown",
            options=[
                {"label": "1 Minute", "value": "1m"},
                {"label": "5 Minutes", "value": "5m"},
                {"label": "1 Hour", "value": "1h"},
                {"label": "1 Day", "value": "1d"},
            ],
            value="1m",  # Default value
            clearable=False,
            style={"width": "50%"},
        ),
        dcc.Graph(id="live-candlestick-chart"),
        dcc.Interval(
            id="interval-component",
            interval=10 * 1000,  # in milliseconds (10 seconds)
            n_intervals=0,  # starts from zero
        ),
    ]
)


# Function to update the chart
@app.callback(
    Output("live-candlestick-chart", "figure"),
    Input("interval-component", "n_intervals"),
    Input("timeframe-dropdown", "value"),
)
def update_chart(n_intervals, timeframe):
    global candles_df
    symbol = "BTC/USDT"

    # Fetch the latest data (only the last 2 candles to reduce bandwidth)
    new_candles = fetch_candles(symbol, timeframe, limit=2)

    if candles_df.empty:
        # Initialize the dataframe if it's the first run
        candles_df = new_candles
    else:
        # Append new candles if not duplicates
        last_timestamp = candles_df["timestamp"].iloc[-1]
        new_candles = new_candles[new_candles["timestamp"] > last_timestamp]
        if not new_candles.empty:
            candles_df = pd.concat([candles_df, new_candles])

    # Plot the chart
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=candles_df["timestamp"],
                open=candles_df["open"],
                high=candles_df["high"],
                low=candles_df["low"],
                close=candles_df["close"],
                name="BTC/USDT",
            )
        ]
    )

    fig.update_layout(
        title=f"BTC/USDT Candlestick Chart ({timeframe})",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )

    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
