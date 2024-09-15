import json
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyqtgraph as pg
import requests
import websocket
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget


class WebSocketThread(QThread):
    message_received = pyqtSignal(dict)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.ws = None

    def run(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.ws.run_forever()

    def on_message(self, ws, message):
        trade = json.loads(message)
        self.message_received.emit(trade)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws):
        print("WebSocket connection closed")

    def stop(self):
        if self.ws:
            self.ws.close()


class CandlestickItem(pg.GraphicsObject):
    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.picture = pg.QtGui.QPicture()
        self.data = []

    def set_data(self, data):
        self.data = data
        self.generatePicture()
        self.informViewBoundsChanged()

    def generatePicture(self):
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen("w"))
        if len(self.data) > 1:
            w = (self.data[1][0] - self.data[0][0]) / 3.0
            for t, open, close, low, high, volume in self.data:
                p.drawLine(pg.QtCore.QPointF(t, low), pg.QtCore.QPointF(t, high))
                if open > close:
                    p.setBrush(pg.mkBrush("r"))
                else:
                    p.setBrush(pg.mkBrush("g"))
                p.drawRect(pg.QtCore.QRectF(t - w, open, w * 2, close - open))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return pg.QtCore.QRectF(self.picture.boundingRect())


class BTCLiveChart(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live BTC/USDT Chart")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.graph_widget = pg.PlotWidget()
        layout.addWidget(self.graph_widget)

        self.candlestick_item = CandlestickItem()
        self.graph_widget.addItem(self.candlestick_item)

        self.data = []
        self.timeframe = 60  # 1 minute candles
        self.current_candle = None

        self.fetch_historical_data()
        self.setup_websocket()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_chart)
        self.update_timer.start(1000)  # Update every second

        # Set up hover event
        self.hover_label = pg.LabelItem(justify="left")
        self.graph_widget.addItem(self.hover_label, row=1, col=0)
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.graph_widget.addItem(self.vLine, ignoreBounds=True)
        self.graph_widget.addItem(self.hLine, ignoreBounds=True)
        self.proxy = pg.SignalProxy(
            self.graph_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved
        )

        # Set up x-axis time format
        self.graph_widget.getAxis("bottom").setStyle(showValues=True)
        self.graph_widget.getAxis("bottom").setLabel("Time")
        self.graph_widget.getAxis("bottom").tickStrings = self.tickStrings

        # Enable auto-ranging for y-axis, but keep it turned off for x-axis
        self.graph_widget.enableAutoRange(axis="y")
        self.graph_widget.disableAutoRange(axis="x")

        # Store the initial view range
        self.view_range = self.graph_widget.viewRange()

    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(value).strftime("%H:%M:%S") for value in values]

    def fetch_historical_data(self):
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100"
        response = requests.get(url)
        klines = response.json()

        for kline in klines:
            timestamp = kline[0] / 1000
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            self.data.append(
                (timestamp, open_price, close_price, low_price, high_price, volume)
            )

        self.update_chart(initial=True)

    def setup_websocket(self):
        self.ws_thread = WebSocketThread(
            "wss://stream.binance.com:9443/ws/btcusdt@trade"
        )
        self.ws_thread.message_received.connect(self.process_websocket_message)
        self.ws_thread.start()

    def process_websocket_message(self, trade):
        timestamp = trade["E"] / 1000
        price = float(trade["p"])

        if (
            self.current_candle is None
            or timestamp >= self.current_candle[0] + self.timeframe
        ):
            if self.current_candle is not None:
                self.data.append(self.current_candle)
            self.current_candle = [
                timestamp,
                price,
                price,
                price,
                price,
                float(trade["q"]),
            ]
        else:
            self.current_candle[2] = price  # Update close price
            self.current_candle[3] = min(
                self.current_candle[3], price
            )  # Update low price
            self.current_candle[4] = max(
                self.current_candle[4], price
            )  # Update high price
            self.current_candle[5] += float(trade["q"])  # Update volume

    def update_chart(self, initial=False):
        if self.current_candle is not None:
            data_to_plot = self.data + [self.current_candle]
        else:
            data_to_plot = self.data

        self.candlestick_item.set_data(data_to_plot)

        if initial and data_to_plot:
            # Set initial view to show the last 100 candles
            self.graph_widget.setXRange(data_to_plot[-100][0], data_to_plot[-1][0])
        elif data_to_plot:
            # Update the view range only if the latest data point is out of view
            view_range = self.graph_widget.viewRange()
            if data_to_plot[-1][0] > view_range[0][1]:
                self.graph_widget.setXRange(
                    view_range[0][0] + self.timeframe,
                    view_range[0][1] + self.timeframe,
                    padding=0,
                )

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.graph_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.graph_widget.plotItem.vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            if 0 <= index < len(self.data):
                timestamp, open, close, low, high, volume = self.data[index]
                self.hover_label.setText(
                    f"""
                    <span style='font-size: 12pt'>
                    Time: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}<br>
                    Open: {open:.2f}<br>
                    Close: {close:.2f}<br>
                    Low: {low:.2f}<br>
                    High: {high:.2f}<br>
                    Volume: {volume:.2f}
                    </span>
                """
                )
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def closeEvent(self, event):
        self.ws_thread.stop()
        self.ws_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = BTCLiveChart()
    main_window.show()
    sys.exit(app.exec_())
