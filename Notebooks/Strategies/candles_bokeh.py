import asyncio
import datetime as dt
import json
import time
from threading import Thread

import ccxt
import pandas as pd
import websockets
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure

# Binance WebSocket URL for real-time BTC/USDT price updates
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# Initialize exchange for historical data fetching
exchange = ccxt.binance()


# Function to fetch historical BTC/USDT candles
def fetch_historical_candles(symbol="BTC/USDT", timeframe="1m", limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# Fetch initial historical data
candles_df = fetch_historical_candles()

# Bokeh data source for real-time plotting
source = ColumnDataSource(
    data=dict(
        time=candles_df["timestamp"],
        open=candles_df["open"],
        high=candles_df["high"],
        low=candles_df["low"],
        close=candles_df["close"],
    )
)

# Create the Bokeh figure
p = figure(
    x_axis_type="datetime",
    title="BTC/USDT Real-Time Candlestick Chart",
    sizing_mode="stretch_width",
)
p.xaxis.axis_label = "Time"
p.yaxis.axis_label = "Price (USDT)"

# Add candlestick glyphs
p.segment(x0="time", y0="high", x1="time", y1="low", source=source, color="black")
p.vbar(
    x="time",
    width=2000000,
    top="open",
    bottom="close",
    fill_color="green",
    line_color="black",
    source=source,
)

# Current candle being updated live
current_candle = {
    "timestamp": None,
    "open": None,
    "high": None,
    "low": None,
    "close": None,
    "volume": 0,
}


# Function to update Bokeh chart with new data
def update_chart():
    if current_candle["timestamp"] is not None:
        source.stream(
            {
                "time": [current_candle["timestamp"]],
                "open": [current_candle["open"]],
                "high": [current_candle["high"]],
                "low": [current_candle["low"]],
                "close": [current_candle["close"]],
            },
            rollover=200,
        )  # Rollover keeps the last 200 candles on the chart


# Function to handle WebSocket messages from Binance
async def binance_websocket():
    async with websockets.connect(BINANCE_WS_URL) as websocket:
        global current_candle
        global candles_df

        while True:
            message = await websocket.recv()
            trade = json.loads(message)
            price = float(trade["p"])  # Latest trade price
            volume = float(trade["q"])  # Latest trade volume
            timestamp = dt.datetime.now()

            # If no candle is active, start a new one
            if current_candle["timestamp"] is None:
                current_candle["timestamp"] = timestamp
                current_candle["open"] = price
                current_candle["high"] = price
                current_candle["low"] = price
                current_candle["close"] = price
                current_candle["volume"] = volume
            else:
                # Update the current candle
                current_candle["high"] = max(current_candle["high"], price)
                current_candle["low"] = min(current_candle["low"], price)
                current_candle["close"] = price
                current_candle["volume"] += volume

            # Stream updates to the chart
            curdoc().add_next_tick_callback(update_chart)


# Function to close the current candle and start a new one every `timeframe_seconds`
def close_and_start_new_candle(timeframe_seconds=60):
    global current_candle
    global candles_df

    while True:
        time.sleep(timeframe_seconds)

        if current_candle["timestamp"] is not None:
            # Append the current candle to the DataFrame
            candles_df = candles_df.append(current_candle, ignore_index=True)

            # Reset the current candle for the next one
            current_candle = {
                "timestamp": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": 0,
            }


# Start the WebSocket connection in a new thread
def start_websocket():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(binance_websocket())


# Start the real-time update thread for candle closing
def start_candle_update_thread():
    Thread(target=close_and_start_new_candle, daemon=True).start()


# Start WebSocket thread for real-time updates
Thread(target=start_websocket, daemon=True).start()
start_candle_update_thread()

# Add layout to the document
curdoc().add_root(column(p))
curdoc().title = "BTC/USDT Real-Time Candlestick Chart"
