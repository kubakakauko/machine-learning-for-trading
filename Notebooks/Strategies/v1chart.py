import asyncio
import json
import sys
from datetime import datetime

import pandas as pd
import pyqtgraph as pg
import requests
import websockets
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel, QVBoxLayout
from pyqtgraph.Qt import QtCore
from qasync import QEventLoop, asyncSlot


class LiveDataThread(QThread):
    new_candle_signal = pyqtSignal(dict)

    def __init__(self, symbol, timeframe):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe

    async def stream_live_data(self):
        """Stream live data from Binance WebSocket and emit new candles."""
        stream_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.timeframe}"
        async with websockets.connect(stream_url) as websocket:
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                self.new_candle_signal.emit(data["k"])  # Emit the live candle data

    def run(self):
        loop = QEventLoop(self)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.stream_live_data())


class ChartingPlatform(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BTC/USDT Charting Platform")
        self.setGeometry(100, 100, 1200, 800)

        self.initUI()

        # Parameters
        self.symbol = "BTCUSDT"
        self.timeframe = "1m"  # Default timeframe
        self.candle_data = pd.DataFrame()  # Historical + live data
        self.last_candle_open_time = None

        # Load initial historical data
        self.load_historical_data()

        # Start the live data thread
        self.live_data_thread = LiveDataThread(self.symbol, self.timeframe)
        self.live_data_thread.new_candle_signal.connect(self.process_live_data)
        self.live_data_thread.start()

    def initUI(self):
        """Setup the user interface."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Candlestick chart
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        layout.addWidget(self.plot_widget)

        # Label for current cursor info
        self.cursor_label = QLabel("Candle Info: ")
        layout.addWidget(self.cursor_label)

        # Setup plot
        self.candle_plot = []  # Store items for candles (Rectangles and Lines)
        self.plot_widget.setLabel("left", "Price", units="USDT")
        self.plot_widget.setLabel("bottom", "Time")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMouseEnabled(x=True, y=False)

        # Connect mouse movement to display candle info
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)

    def load_historical_data(self, days=1):
        """Fetch historical data from Binance API and update the chart."""
        url = f"https://api.binance.com/api/v3/klines?symbol={self.symbol}&interval={self.timeframe}&limit={days * 1440}"
        response = requests.get(url).json()

        # Process historical data
        data = []
        for candle in response:
            timestamp = int(candle[0]) // 1000
            data.append(
                {
                    "timestamp": datetime.fromtimestamp(timestamp),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                }
            )

        self.candle_data = pd.DataFrame(data)
        self.plot_candles()

    def plot_candles(self):
        """Plot candle data on the chart."""
        self.plot_widget.clear()  # Clear the previous candles

        for i in range(len(self.candle_data)):
            candle = self.candle_data.iloc[i]
            timestamp = candle["timestamp"].timestamp()

            # Draw the candle body (open-close as a rectangle)
            rect_color = "g" if candle["close"] >= candle["open"] else "r"
            rect = QtWidgets.QGraphicsRectItem(
                timestamp,
                min(candle["open"], candle["close"]),
                60,
                abs(candle["close"] - candle["open"]),
            )
            rect.setBrush(pg.mkBrush(rect_color))
            self.plot_widget.addItem(rect)

            # Draw the high-low line
            line = QtWidgets.QGraphicsLineItem(
                timestamp + 30, candle["low"], timestamp + 30, candle["high"]
            )
            line.setPen(pg.mkPen(rect_color))
            self.plot_widget.addItem(line)

            self.candle_plot.append((rect, line))

    @asyncSlot(dict)
    def process_live_data(self, candle):
        """Process and add live candle data."""
        timestamp = int(candle["t"]) // 1000
        candle_data = pd.DataFrame(
            {
                "timestamp": [datetime.fromtimestamp(timestamp)],
                "open": [float(candle["o"])],
                "high": [float(candle["h"])],
                "low": [float(candle["l"])],
                "close": [float(candle["c"])],
                "volume": [float(candle["v"])],
            }
        )

        # If the last candle already exists, update it
        if (
            not self.candle_data.empty
            and self.candle_data.iloc[-1]["timestamp"]
            == candle_data.iloc[0]["timestamp"]
        ):
            self.candle_data.iloc[-1] = candle_data.iloc[0]
        else:
            # Otherwise, append a new candle
            self.candle_data = pd.concat(
                [self.candle_data, candle_data], ignore_index=True
            )

        # Plot updated candles
        self.plot_candles()

    def mouse_moved(self, event):
        """Display the candle information where the cursor is pointing."""
        pos = event
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x = mouse_point.x()

            # Find the closest timestamp in the candle data
            idx = (
                (self.candle_data["timestamp"].apply(lambda t: t.timestamp()) - x)
                .abs()
                .argmin()
            )
            candle = self.candle_data.iloc[idx]

            info = f"Time: {candle['timestamp']}, Open: {candle['open']}, Close: {candle['close']}"
            self.cursor_label.setText(f"Candle Info: {info}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = ChartingPlatform()
    window.show()
    with loop:
        loop.run_forever()
