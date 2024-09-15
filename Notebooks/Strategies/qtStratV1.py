import json
import sys
import threading
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
import pyqtgraph as pg
import requests
import websocket
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from statsmodels.nonparametric.kernel_regression import KernelReg

# Configuration Parameters
SYMBOL = "BTCUSDT"
INTERVAL = "1m"  # Fetch 1-minute candles
EMA_SLOW = 50
EMA_FAST = 40
ATR_LENGTH = 14
BACKCANDLES = 7
SL_COEF = 1.5
TP_SL_RATIO = 2.0
SLIPPAGE = 0.0005
NW_MULT = 2
BANDWIDTH = 7
TRAIL_PERCENT = 0.00618  # Trailing percentage for both SL and TP
MAX_TRADE_DURATION = timedelta(hours=6)  # Close trades after 6 hours
ACCOUNT_BALANCE = 100  # Example starting balance in USD
RISK_PER_TRADE = 0.1  # Risk 10% of account per trade


class WebSocketThread(QtCore.QThread):
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


class Trade:
    """
    Class to represent a trade with entry/exit details and PnL calculation.
    """

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


class BTCLiveChart(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live BTC/USDT Chart")
        self.resize(1200, 800)

        # Initialize variables
        self.data = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )  # DataFrame to store OHLCV data
        self.current_candle = None
        self.timeframe = 60  # 1 minute in seconds
        self.current_trade = None
        self.trades = []
        self.ACCOUNT_BALANCE = ACCOUNT_BALANCE

        # Set up the UI components
        self.setup_ui()

        # Fetch historical data and start websocket
        self.fetch_historical_data()
        self.setup_websocket()

        # Set up a timer to update the chart periodically
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_chart)
        self.update_timer.start(1000)  # Update every second

    def setup_ui(self):
        """
        Set up the UI components, including the plot widget and cursor lines.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create the plot widget
        self.graph_widget = pg.PlotWidget()
        layout.addWidget(self.graph_widget)

        # Initialize plot items
        self.candlestick_item = CandlestickItem()
        self.ema_fast_line = self.graph_widget.plot(
            [], [], pen=pg.mkPen("b", width=1), name="EMA Fast"
        )
        self.ema_slow_line = self.graph_widget.plot(
            [], [], pen=pg.mkPen("y", width=1), name="EMA Slow"
        )
        self.upper_envelope_line = self.graph_widget.plot(
            [],
            [],
            pen=pg.mkPen("r", width=1, style=QtCore.Qt.DashLine),
            name="Upper Envelope",
        )
        self.lower_envelope_line = self.graph_widget.plot(
            [],
            [],
            pen=pg.mkPen("g", width=1, style=QtCore.Qt.DashLine),
            name="Lower Envelope",
        )
        self.buy_signal_scatter = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush("g"), name="Buy Signals"
        )
        self.sell_signal_scatter = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush("r"), name="Sell Signals"
        )
        self.entry_scatter = pg.ScatterPlotItem(size=12, name="Trade Entries")
        self.exit_scatter = pg.ScatterPlotItem(size=12, name="Trade Exits")

        # Add items to the plot
        self.graph_widget.addItem(self.candlestick_item)
        self.graph_widget.addItem(self.buy_signal_scatter)
        self.graph_widget.addItem(self.sell_signal_scatter)
        self.graph_widget.addItem(self.entry_scatter)
        self.graph_widget.addItem(self.exit_scatter)

        # Set up cursor lines
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.graph_widget.addItem(self.vLine, ignoreBounds=True)
        self.graph_widget.addItem(self.hLine, ignoreBounds=True)
        self.proxy = pg.SignalProxy(
            self.graph_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.mouse_moved,
        )

        # Set up x-axis time format
        axis = self.graph_widget.getAxis("bottom")
        axis.setTickSpacing(major=60, minor=10)
        axis.tickStrings = self.tick_strings

        # Enable auto-ranging for y-axis, but keep it turned off for x-axis
        self.graph_widget.enableAutoRange(axis="y", enable=True)
        self.graph_widget.setAutoVisible(y=True)
        self.graph_widget.showGrid(x=True, y=True, alpha=0.3)

    def tick_strings(self, values, scale, spacing):
        """
        Custom function to format x-axis tick labels as time strings.
        """
        return [datetime.fromtimestamp(value).strftime("%H:%M") for value in values]

    def fetch_historical_data(self):
        """
        Fetch historical data from Binance API and calculate indicators.
        """
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit=1000"
        response = requests.get(url)
        klines = response.json()

        data_list = []
        for kline in klines:
            timestamp = datetime.fromtimestamp(kline[0] / 1000)
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            data_list.append(
                {
                    "timestamp": timestamp,
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                }
            )

        # Create DataFrame and set index
        self.data = pd.DataFrame(data_list)
        self.data.set_index("timestamp", inplace=True)

        # Calculate indicators and signals
        self.calculate_indicators()
        self.calculate_nadaraya_watson()
        self.ema_signal()
        self.total_signal()

        # Manage any trades based on historical data
        self.manage_trades()

        # Initial chart update
        self.update_chart(initial=True)

    def setup_websocket(self):
        """
        Set up the WebSocket connection to receive live trade data.
        """
        self.ws_thread = WebSocketThread(
            f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@trade"
        )
        self.ws_thread.message_received.connect(self.process_websocket_message)
        self.ws_thread.start()

    def process_websocket_message(self, trade):
        """
        Process incoming trade messages and update current candle.
        """
        timestamp = datetime.fromtimestamp(trade["E"] / 1000)
        price = float(trade["p"])

        if (
            self.current_candle is None
            or timestamp >= self.current_candle.name + timedelta(seconds=self.timeframe)
        ):
            if self.current_candle is not None:
                # Append the completed candle to data
                self.data = pd.concat([self.data, self.current_candle.to_frame().T])

                # Recalculate indicators and signals
                self.calculate_indicators()
                self.calculate_nadaraya_watson()
                self.ema_signal()
                self.total_signal()

                # Manage trades
                self.manage_trades()

            # Start a new candle
            self.current_candle = pd.Series(
                {
                    "Open": price,
                    "High": price,
                    "Low": price,
                    "Close": price,
                    "Volume": float(trade["q"]),
                },
                name=timestamp,
            )
        else:
            # Update current candle
            self.current_candle["Close"] = price
            self.current_candle["High"] = max(self.current_candle["High"], price)
            self.current_candle["Low"] = min(self.current_candle["Low"], price)
            self.current_candle["Volume"] += float(trade["q"])

    def calculate_indicators(self):
        """
        Calculate EMA and ATR indicators.
        """
        self.data["EMA_slow"] = ta.ema(self.data["Close"], length=EMA_SLOW)
        self.data["EMA_fast"] = ta.ema(self.data["Close"], length=EMA_FAST)
        self.data["ATR"] = ta.atr(
            self.data["High"],
            self.data["Low"],
            self.data["Close"],
            length=ATR_LENGTH,
        )

    def calculate_nadaraya_watson(self):
        """
        Calculate Nadaraya-Watson envelopes.
        """
        df = self.data.copy()
        X = np.arange(len(df))
        y = df["Close"].values

        # Perform Nadaraya-Watson kernel regression
        model = KernelReg(endog=y, exog=X, var_type="c", bw=[BANDWIDTH])
        fitted_values, _ = model.fit(X)

        # Store the fitted values
        df["NW_Fitted"] = fitted_values

        # Calculate the residuals and standard deviation
        residuals = df["Close"] - df["NW_Fitted"]
        std_dev = np.std(residuals)

        # Create the envelopes
        df["Upper_Envelope"] = df["NW_Fitted"] + NW_MULT * std_dev
        df["Lower_Envelope"] = df["NW_Fitted"] - NW_MULT * std_dev

        self.data = df

    def ema_signal(self):
        """
        Generate EMA crossover signals.
        """
        df = self.data
        above = df["EMA_fast"] > df["EMA_slow"]
        below = df["EMA_fast"] < df["EMA_slow"]

        above_all = (
            above.rolling(window=BACKCANDLES)
            .apply(lambda x: x.all(), raw=True)
            .fillna(0)
            .astype(bool)
        )
        below_all = (
            below.rolling(window=BACKCANDLES)
            .apply(lambda x: x.all(), raw=True)
            .fillna(0)
            .astype(bool)
        )

        df["EMASignal"] = 0
        df.loc[above_all, "EMASignal"] = 2
        df.loc[below_all, "EMASignal"] = 1

        self.data = df

    def total_signal(self):
        """
        Generate total trading signals based on EMA and Nadaraya-Watson envelopes.
        """
        df = self.data
        condition_buy = (df["EMASignal"] == 2) & (df["Close"] <= df["Lower_Envelope"])
        condition_sell = (df["EMASignal"] == 1) & (df["Close"] >= df["Upper_Envelope"])

        df["Total_Signal"] = 0
        df.loc[condition_buy, "Total_Signal"] = 2
        df.loc[condition_sell, "Total_Signal"] = 1

        self.data = df

    def calculate_position_size(self, entry_price, stop_loss_price):
        """
        Calculate the position size based on account balance and risk per trade.
        """
        amount_at_risk = self.ACCOUNT_BALANCE * RISK_PER_TADE
        price_difference = abs(entry_price - stop_loss_price)
        position_size = amount_at_risk / price_difference
        return position_size

    def update_trailing_levels(self, current_trade, current_price):
        """
        Update the trailing stop loss and take profit levels.
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

    def manage_trades(self):
        """
        Manage trades by generating signals and handling entry/exit logic.
        """
        latest_row = self.data.iloc[-1]
        signal = latest_row["Total_Signal"]

        if self.current_trade is None:
            if signal == 2:  # Buy signal
                entry_price = latest_row["Close"] * (1 + SLIPPAGE)
                sl_distance = latest_row["ATR"] * SL_COEF
                tp_distance = sl_distance * TP_SL_RATIO
                sl = entry_price - sl_distance
                tp = entry_price + tp_distance
                position_size = self.calculate_position_size(entry_price, sl)

                self.current_trade = Trade(
                    entry_time=latest_row.name,
                    entry_price=entry_price,
                    trade_type="Long",
                    sl=sl,
                    tp=tp,
                    position_size=position_size,
                )
                print(
                    f"Entered Long at {entry_price:.2f} with SL: {sl:.2f} and TP: {tp:.2f}"
                )

            elif signal == 1:  # Sell signal
                entry_price = latest_row["Close"] * (1 - SLIPPAGE)
                sl_distance = latest_row["ATR"] * SL_COEF
                tp_distance = sl_distance * TP_SL_RATIO
                sl = entry_price + sl_distance
                tp = entry_price - tp_distance
                position_size = self.calculate_position_size(entry_price, sl)

                self.current_trade = Trade(
                    entry_time=latest_row.name,
                    entry_price=entry_price,
                    trade_type="Short",
                    sl=sl,
                    tp=tp,
                    position_size=position_size,
                )
                print(
                    f"Entered Short at {entry_price:.2f} with SL: {sl:.2f} and TP: {tp:.2f}"
                )
        else:
            # Update Trailing SL and TP
            current_price = latest_row["Close"]
            self.update_trailing_levels(self.current_trade, current_price)

            # Check for exit conditions
            exit_price = None
            if self.current_trade.Type == "Long":
                if current_price <= self.current_trade.CurrentSL:
                    exit_price = self.current_trade.CurrentSL
                elif current_price >= self.current_trade.CurrentTP:
                    exit_price = self.current_trade.CurrentTP
            else:
                if current_price >= self.current_trade.CurrentSL:
                    exit_price = self.current_trade.CurrentSL
                elif current_price <= self.current_trade.CurrentTP:
                    exit_price = self.current_trade.CurrentTP

            # Time-based exit
            trade_duration = latest_row.name - self.current_trade.EntryTime
            if trade_duration >= MAX_TRADE_DURATION:
                exit_price = current_price
                print(
                    f"Time-based exit for {self.current_trade.Type} trade at {exit_price:.2f}"
                )

            if exit_price is not None:
                self.current_trade.ExitTime = latest_row.name
                self.current_trade.ExitPrice = exit_price
                if self.current_trade.Type == "Long":
                    self.current_trade.PnL = (
                        exit_price - self.current_trade.EntryPrice
                    ) * self.current_trade.PositionSize
                else:
                    self.current_trade.PnL = (
                        self.current_trade.EntryPrice - exit_price
                    ) * self.current_trade.PositionSize

                # Update account balance
                self.ACCOUNT_BALANCE += self.current_trade.PnL

                self.trades.append(self.current_trade)
                print(
                    f"Exited {self.current_trade.Type} at {exit_price:.2f} with PnL: {self.current_trade.PnL:.2f}"
                )
                self.current_trade = None

    def update_chart(self, initial=False):
        """
        Update the chart with new data, indicators, and trade signals.
        """
        if self.current_candle is not None:
            data_to_plot = pd.concat([self.data, self.current_candle.to_frame().T])
        else:
            data_to_plot = self.data

        # Update candlestick data
        candlestick_data = []
        for idx, row in data_to_plot.iterrows():
            timestamp = idx.timestamp()
            open_price = row["Open"]
            close_price = row["Close"]
            low_price = row["Low"]
            high_price = row["High"]
            volume = row["Volume"]
            candlestick_data.append(
                (timestamp, open_price, close_price, low_price, high_price, volume)
            )
        self.candlestick_item.set_data(candlestick_data)

        # Update EMA lines
        timestamps = data_to_plot.index.map(datetime.timestamp).tolist()
        self.ema_fast_line.setData(timestamps, data_to_plot["EMA_fast"].tolist())
        self.ema_slow_line.setData(timestamps, data_to_plot["EMA_slow"].tolist())

        # Update Nadaraya-Watson envelopes
        self.upper_envelope_line.setData(
            timestamps, data_to_plot["Upper_Envelope"].tolist()
        )
        self.lower_envelope_line.setData(
            timestamps, data_to_plot["Lower_Envelope"].tolist()
        )

        # Update buy and sell signals
        buy_signals = data_to_plot[data_to_plot["Total_Signal"] == 2]
        sell_signals = data_to_plot[data_to_plot["Total_Signal"] == 1]
        self.buy_signal_scatter.setData(
            x=buy_signals.index.map(datetime.timestamp).tolist(),
            y=buy_signals["Close"].tolist(),
        )
        self.sell_signal_scatter.setData(
            x=sell_signals.index.map(datetime.timestamp).tolist(),
            y=sell_signals["Close"].tolist(),
        )

        # Update trade entries and exits
        entry_x = []
        entry_y = []
        exit_x = []
        exit_y = []
        entry_brushes = []
        exit_brushes = []
        for trade in self.trades:
            # Entry point
            entry_x.append(trade.EntryTime.timestamp())
            entry_y.append(trade.EntryPrice)
            entry_color = "g" if trade.Type == "Long" else "r"
            entry_brushes.append(pg.mkBrush(entry_color))

            if trade.ExitTime and trade.ExitPrice:
                # Exit point
                exit_x.append(trade.ExitTime.timestamp())
                exit_y.append(trade.ExitPrice)
                pnl_color = "g" if trade.PnL > 0 else "r"
                exit_brushes.append(pg.mkBrush(pnl_color))

        self.entry_scatter.setData(
            x=entry_x, y=entry_y, brush=entry_brushes, symbol="t1", size=12
        )
        self.exit_scatter.setData(
            x=exit_x, y=exit_y, brush=exit_brushes, symbol="o", size=12
        )

        # Adjust view range
        if initial and not data_to_plot.empty:
            # Show the last 100 candles
            self.graph_widget.setXRange(
                data_to_plot.index[-100].timestamp(),
                data_to_plot.index[-1].timestamp(),
            )
        elif not data_to_plot.empty:
            # Update the view range if the latest data point is out of view
            view_range = self.graph_widget.viewRange()
            if data_to_plot.index[-1].timestamp() > view_range[0][1]:
                self.graph_widget.setXRange(
                    view_range[0][0] + self.timeframe,
                    view_range[0][1] + self.timeframe,
                    padding=0,
                )

    def mouse_moved(self, evt):
        """
        Update cursor lines when the mouse is moved over the plot.
        """
        pos = evt[0]
        if self.graph_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.graph_widget.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mouse_point.x())
            self.hLine.setPos(mouse_point.y())

    def closeEvent(self, event):
        """
        Clean up threads when the application is closed.
        """
        self.ws_thread.stop()
        self.ws_thread.wait()
        event.accept()


class CandlestickItem(pg.GraphicsObject):
    """
    Custom GraphicsObject to draw candlestick charts.
    """

    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.picture = pg.QtGui.QPicture()
        self.data = []

    def set_data(self, data):
        """
        Set the data for the candlesticks and generate the picture.
        """
        self.data = data
        self.generate_picture()
        self.informViewBoundsChanged()

    def generate_picture(self):
        """
        Generate the picture object containing all the candlesticks.
        """
        self.picture = pg.QtGui.QPicture()
        p = pg.QtGui.QPainter(self.picture)
        w = 0.4  # Width of the candlestick bodies
        for t, open, close, low, high, volume in self.data:
            # Set pen and brush based on candle direction
            if open > close:
                p.setPen(pg.mkPen("r"))
                p.setBrush(pg.mkBrush("r"))
            else:
                p.setPen(pg.mkPen("g"))
                p.setBrush(pg.mkBrush("g"))
            # Draw wick
            p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            # Draw body
            p.drawRect(
                QtCore.QRectF(t - w / 2, open, w, close - open)
            )  # open and close are y-coordinates
        p.end()

    def paint(self, p, *args):
        """
        Paint the picture onto the canvas.
        """
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        """
        Define the bounding rectangle for the item.
        """
        return QtCore.QRectF(self.picture.boundingRect())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = BTCLiveChart()
    main_window.show()
    sys.exit(app.exec_())
