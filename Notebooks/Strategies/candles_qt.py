import ccxt
import pandas as pd
import time
import json
import asyncio
import websockets
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

# Initialize the Binance exchange using ccxt
exchange = ccxt.binance()


# Function to fetch historical BTC/USDT candles
def fetch_historical_candles(symbol="BTC/USDT", timeframe="1m", limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# Fetch initial historical data (100 candles)
candles_df = fetch_historical_candles()

# Initialize WebSocket URL for Binance trade updates
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# Current open candle being updated
current_candle = {
    "timestamp": None,
    "open": None,
    "high": None,
    "low": None,
    "close": None,
    "volume": 0,
}


class CandlestickChart(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create the PyQtGraph plot window
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True)

        # Initialize the candlestick data plot
        self.plot_candles()

        # Set up the timer to call update functions periodically
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(500)  # Update every 500 ms (0.5 seconds)

    def plot_candles(self):
        """Plot historical candles on the chart."""
        self.candle_items = []
        for i in range(len(candles_df)):
            self.draw_candle(i)

    def draw_candle(self, i):
        """Draw a single candle at position i."""
        time = (
            candles_df["timestamp"].astype(np.int64) // 10**6
        )  # Convert timestamp to ms
        open_price = candles_df["open"].iloc[i]
        high_price = candles_df["high"].iloc[i]
        low_price = candles_df["low"].iloc[i]
        close_price = candles_df["close"].iloc[i]

        # Draw a line from high to low
        high_low_line = pg.PlotDataItem(
            [time[i], time[i]], [low_price, high_price], pen=pg.mkPen("k")
        )
        self.plot_widget.addItem(high_low_line)

        # Draw a rectangle between open and close prices
        if close_price >= open_price:
            color = (0, 255, 0)  # Green for bullish candle
        else:
            color = (255, 0, 0)  # Red for bearish candle

        candle_rect = pg.PlotDataItem(
            [time[i], time[i]], [open_price, close_price], pen=pg.mkPen(color, width=6)
        )
        self.plot_widget.addItem(candle_rect)
        self.candle_items.append((high_low_line, candle_rect))

    def update_chart(self):
        """Update the current open candle in real time."""
        if current_candle["timestamp"] is not None:
            # Append or update the current live candle
            time_stamp = int(current_candle["timestamp"].timestamp() * 1000)
            self.draw_real_time_candle(time_stamp)

    def draw_real_time_candle(self, time_stamp):
        """Draw or update the real-time candle."""
        i = len(candles_df)  # The current open candle is the next in line

        high_low_line = pg.PlotDataItem(
            [time_stamp, time_stamp],
            [current_candle["low"], current_candle["high"]],
            pen=pg.mkPen("k"),
        )

        open_close_line = pg.PlotDataItem(
            [time_stamp, time_stamp],
            [current_candle["open"], current_candle["close"]],
            pen=pg.mkPen(
                "g" if current_candle["close"] >= current_candle["open"] else "r",
                width=6,
            ),
        )

        self.plot_widget.addItem(high_low_line)
        self.plot_widget.addItem(open_close_line)

    def close_candle(self):
        """Close the current candle and append it to candles_df."""
        global current_candle
        if current_candle["timestamp"] is not None:
            candles_df.loc[len(candles_df)] = current_candle

            current_candle = {
                "timestamp": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": 0,
            }


# WebSocket function to handle real-time data
async def binance_websocket(chart):
    global current_candle

    async with websockets.connect(BINANCE_WS_URL) as websocket:
        while True:
            message = await websocket.recv()
            trade = json.loads(message)
            price = float(trade["p"])  # Latest trade price
            volume = float(trade["q"])  # Latest trade volume
            timestamp = dt.datetime.now()

            # If no candle is open, start a new one
            if current_candle["timestamp"] is None:
                current_candle["timestamp"] = timestamp
                current_candle["open"] = price
                current_candle["high"] = price
                current_candle["low"] = price
                current_candle["close"] = price
                current_candle["volume"] = volume
            else:
                # Update the current open candle
                current_candle["high"] = max(current_candle["high"], price)
                current_candle["low"] = min(current_candle["low"], price)
                current_candle["close"] = price
                current_candle["volume"] += volume


# Function to close the current candle every timeframe (e.g., 1 minute)
def close_and_start_new_candle(chart, timeframe_seconds=60):
    global current_candle

    while True:
        time.sleep(timeframe_seconds)
        if current_candle["timestamp"] is not None:
            chart.close_candle()


# Run the PyQt application and WebSocket connection in parallel
def run_app():
    app = QApplication([])

    chart = CandlestickChart()

    loop = asyncio.get_event_loop()

    # Run the WebSocket in a new thread
    websocket_thread = QtCore.QThread()
    loop.create_task(binance_websocket(chart))
    websocket_thread.start()

    # Close and start a new candle every 60 seconds
    close_candle_thread = QtCore.QThread()
    close_candle_thread.run = lambda: close_and_start_new_candle(
        chart, timeframe_seconds=60
    )
    close_candle_thread.start()

    chart.show()
    app.exec_()


if __name__ == "__main__":
    run_app()
