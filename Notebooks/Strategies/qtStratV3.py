import sys
import json
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from statsmodels.nonparametric.kernel_regression import KernelReg
import websocket

# Configuration Parameters
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"  # Fetch 1-minute candles
AGG_INTERVAL_MINUTES = 1  # Desired timeframe to aggregate to
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
HISTORICAL_LIMIT = 1000  # Number of historical candles to fetch

# Account and Risk Management
ACCOUNT_BALANCE = 10000  # Example starting balance in USD
RISK_PER_TRADE = 0.01  # Risk 1% of account per trade
MAX_TRADE_DURATION = timedelta(hours=6)  # Close trades after 6 hours

# Timeframe in seconds
TIMEFRAME_SECONDS = 60  # 1 minute


# Trade class to manage individual trades
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


# WebSocket thread to receive live trade data
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


# Custom CandlestickItem for plotting candlesticks in pyqtgraph
class CandlestickItem(pg.GraphicsObject):
    def __init__(self):
        super().__init__()
        self.picture = QtGui.QPicture()
        self.data = []

    def set_data(self, data):
        self.data = data
        self.generate_picture()
        self.informViewBoundsChanged()

    def generate_picture(self):
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        # Ensure we have enough data to calculate width
        if len(self.data) > 1:
            w = (self.data[1][0] - self.data[0][0]) / 3.0
        else:
            w = 0

        for t, open_price, close_price, low_price, high_price in self.data:
            if open_price > close_price:
                painter.setPen(pg.mkPen("r"))
                painter.setBrush(pg.mkBrush("r"))
            else:
                painter.setPen(pg.mkPen("g"))
                painter.setBrush(pg.mkBrush("g"))
            painter.drawLine(
                QtCore.QPointF(t, low_price), QtCore.QPointF(t, high_price)
            )
            rect = QtCore.QRectF(t - w, open_price, w * 2, close_price - open_price)
            # Normalize the rectangle in case of negative height
            rect = rect.normalized()
            painter.drawRect(rect)
        painter.end()

    def paint(self, painter, option, widget):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        # Ensure boundingRect returns a QRectF
        return QtCore.QRectF(self.picture.boundingRect())


# Main class for the live trading chart and strategy
class BTCLiveChart(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live BTC/USDT Chart")
        self.resize(1200, 800)

        # Initialize account balance and trade variables
        self.account_balance = ACCOUNT_BALANCE
        self.trades = []
        self.current_trade = None

        # Initialize data DataFrame
        self.data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        # Set up the UI components
        self.setup_ui()

        # Fetch historical data and set up the chart
        self.fetch_historical_data()
        self.setup_websocket()

        # Timer to update the chart periodically
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_chart)
        self.update_timer.start(1000)  # Update every second

        # Initialize current candle
        self.current_candle = None

        # Keep track of plotted trade markers and levels
        self.trade_markers = []
        self.level_lines = []

    def setup_ui(self):
        """Set up the user interface components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create the graph widget
        self.graph_widget = pg.PlotWidget()
        layout.addWidget(self.graph_widget)

        self.graph_widget.showGrid(x=True, y=True)

        # Initialize the candlestick item and add to the graph
        self.candlestick_item = CandlestickItem()
        self.graph_widget.addItem(self.candlestick_item)

        # Initialize curves for indicators
        self.ema_fast_curve = pg.PlotCurveItem(pen=pg.mkPen("b", width=1))
        self.graph_widget.addItem(self.ema_fast_curve)
        self.ema_slow_curve = pg.PlotCurveItem(pen=pg.mkPen("y", width=1))
        self.graph_widget.addItem(self.ema_slow_curve)
        self.upper_envelope_curve = pg.PlotCurveItem(
            pen=pg.mkPen("r", width=1, style=QtCore.Qt.DashLine)
        )
        self.graph_widget.addItem(self.upper_envelope_curve)
        self.lower_envelope_curve = pg.PlotCurveItem(
            pen=pg.mkPen("g", width=1, style=QtCore.Qt.DashLine)
        )
        self.graph_widget.addItem(self.lower_envelope_curve)

        # Set up cursor lines
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.graph_widget.addItem(self.vLine, ignoreBounds=True)
        self.graph_widget.addItem(self.hLine, ignoreBounds=True)
        self.proxy = pg.SignalProxy(
            self.graph_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved
        )

        # Enable auto-ranging for y-axis, but keep it turned off for x-axis
        self.graph_widget.enableAutoRange(axis="y")
        self.graph_widget.disableAutoRange(axis="x")

        # Set up x-axis time format
        self.graph_widget.getAxis("bottom").setStyle(showValues=True)
        self.graph_widget.getAxis("bottom").setLabel("Time")
        self.graph_widget.getAxis("bottom").tickStrings = self.tickStrings

    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(value).strftime("%H:%M:%S") for value in values]

    def fetch_historical_data(self):
        """Fetch historical data from Binance and initialize the data DataFrame."""
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={TIMEFRAME}&limit={HISTORICAL_LIMIT}"
        response = requests.get(url)
        klines = response.json()

        data_list = []
        for kline in klines:
            timestamp = int(kline[0] / 1000)
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            data_list.append(
                [timestamp, open_price, high_price, low_price, close_price, volume]
            )

        self.data = pd.DataFrame(
            data_list, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], unit="s")
        self.data.set_index("timestamp", inplace=True)

        # Resample data to AGG_INTERVAL_MINUTES if necessary
        if AGG_INTERVAL_MINUTES > 1:
            self.data = (
                self.data.resample(f"{AGG_INTERVAL_MINUTES}T")
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

        # Calculate indicators and generate initial signals
        self.calculate_indicators()
        self.generate_signals()

    def setup_websocket(self):
        """Set up the WebSocket connection to receive live data."""
        self.ws_thread = WebSocketThread(
            f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@trade"
        )
        self.ws_thread.message_received.connect(self.process_websocket_message)
        self.ws_thread.start()

    def process_websocket_message(self, trade):
        """Process incoming trade data from the WebSocket."""
        timestamp = trade["E"] / 1000  # Event time in seconds
        price = float(trade["p"])
        volume = float(trade["q"])

        # Align timestamp to AGG_INTERVAL_MINUTES
        candle_timestamp = int(timestamp // (AGG_INTERVAL_MINUTES * 60)) * (
            AGG_INTERVAL_MINUTES * 60
        )

        # Check if we need to start a new candle
        if (
            self.current_candle is None
            or candle_timestamp
            >= self.current_candle["timestamp"] + AGG_INTERVAL_MINUTES * 60
        ):
            if self.current_candle is not None:
                # Append the current candle to the data
                new_row = pd.DataFrame(
                    {
                        "Open": [self.current_candle["open"]],
                        "High": [self.current_candle["high"]],
                        "Low": [self.current_candle["low"]],
                        "Close": [self.current_candle["close"]],
                        "Volume": [self.current_candle["volume"]],
                    },
                    index=[pd.to_datetime(self.current_candle["timestamp"], unit="s")],
                )
                self.data = pd.concat([self.data, new_row])
                # After updating data, recalculate indicators and manage trades
                self.calculate_indicators()
                self.generate_signals()
                self.update_trades()

            # Start a new candle
            self.current_candle = {
                "timestamp": candle_timestamp,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
        else:
            # Update the current candle
            self.current_candle["close"] = price
            self.current_candle["high"] = max(self.current_candle["high"], price)
            self.current_candle["low"] = min(self.current_candle["low"], price)
            self.current_candle["volume"] += volume

    def calculate_indicators(self):
        """Calculate technical indicators: EMA, ATR, and Nadaraya-Watson envelopes."""
        self.data["EMA_fast"] = ta.ema(self.data["Close"], length=EMA_FAST)
        self.data["EMA_slow"] = ta.ema(self.data["Close"], length=EMA_SLOW)
        self.data["ATR"] = ta.atr(
            self.data["High"], self.data["Low"], self.data["Close"], length=ATR_LENGTH
        )
        self.calculate_nadaraya_watson()

    def calculate_nadaraya_watson(self):
        """Calculate Nadaraya-Watson envelopes."""
        df = self.data[["Close"]].dropna()
        if len(df) < BANDWIDTH:
            # Ensure columns exist in self.data, set to NaN
            self.data["NW_Fitted"] = np.nan
            self.data["Upper_Envelope"] = np.nan
            self.data["Lower_Envelope"] = np.nan
            return  # Not enough data

        df = df.reset_index()
        df["index"] = np.arange(len(df))

        X = df["index"].values.reshape(-1, 1)
        y = df["Close"].values

        model = KernelReg(endog=y, exog=X, var_type="c", bw=[BANDWIDTH])
        fitted_values, _ = model.fit(X)

        df["NW_Fitted"] = fitted_values
        residuals = df["Close"] - df["NW_Fitted"]
        std_dev = np.std(residuals)

        df["Upper_Envelope"] = df["NW_Fitted"] + NW_MULT * std_dev
        df["Lower_Envelope"] = df["NW_Fitted"] - NW_MULT * std_dev

        df.set_index("timestamp", inplace=True)

        # Assign the calculated columns directly to self.data
        self.data = self.data.merge(
            df[["NW_Fitted", "Upper_Envelope", "Lower_Envelope"]],
            left_index=True,
            right_index=True,
            how="left",
            suffixes=("", "_new"),
        )
        # Remove any old columns
        for col in ["NW_Fitted", "Upper_Envelope", "Lower_Envelope"]:
            if f"{col}_new" in self.data.columns:
                self.data[col] = self.data[f"{col}_new"]
                self.data.drop(columns=[f"{col}_new"], inplace=True)

    def generate_signals(self):
        """Generate trading signals based on EMA crossover and Nadaraya-Watson envelopes."""
        if len(self.data) < BACKCANDLES:
            return  # Not enough data

        # Ensure that required columns exist
        if not all(
            col in self.data.columns
            for col in ["EMA_fast", "EMA_slow", "Lower_Envelope", "Upper_Envelope"]
        ):
            return

        self.data["EMASignal"] = 0
        above = self.data["EMA_fast"] > self.data["EMA_slow"]
        below = self.data["EMA_fast"] < self.data["EMA_slow"]

        self.data.loc[
            above.rolling(window=BACKCANDLES)
            .apply(lambda x: x.all(), raw=True)
            .fillna(0)
            .astype(bool),
            "EMASignal",
        ] = 2
        self.data.loc[
            below.rolling(window=BACKCANDLES)
            .apply(lambda x: x.all(), raw=True)
            .fillna(0)
            .astype(bool),
            "EMASignal",
        ] = 1

        condition_buy = (self.data["EMASignal"] == 2) & (
            self.data["Close"] <= self.data["Lower_Envelope"]
        )
        condition_sell = (self.data["EMASignal"] == 1) & (
            self.data["Close"] >= self.data["Upper_Envelope"]
        )

        self.data["Total_Signal"] = 0
        self.data.loc[condition_buy, "Total_Signal"] = 2
        self.data.loc[condition_sell, "Total_Signal"] = 1

    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk management."""
        amount_at_risk = self.account_balance * RISK_PER_TRADE
        price_difference = abs(entry_price - stop_loss_price)
        position_size = amount_at_risk / price_difference
        return position_size

    def update_trailing_levels(self, current_trade, current_price):
        """Update trailing stop loss and take profit levels."""
        if current_trade.Type == "Long":
            new_sl = max(current_trade.CurrentSL, current_price * (1 - TRAIL_PERCENT))
            current_trade.CurrentSL = new_sl
            new_tp = max(current_trade.CurrentTP, current_price * (1 + TRAIL_PERCENT))
            current_trade.CurrentTP = new_tp
        else:
            new_sl = min(current_trade.CurrentSL, current_price * (1 + TRAIL_PERCENT))
            current_trade.CurrentSL = new_sl
            new_tp = min(current_trade.CurrentTP, current_price * (1 - TRAIL_PERCENT))
            current_trade.CurrentTP = new_tp

    def update_trades(self):
        """Manage trades based on signals and price movements."""
        latest_row = self.data.iloc[-1]
        signal = latest_row.get("Total_Signal", 0)
        current_price = latest_row["Close"]
        timestamp = latest_row.name

        if self.current_trade is None:
            if signal == 2:  # Buy signal
                entry_price = current_price * (1 + SLIPPAGE)
                sl_distance = latest_row["ATR"] * SL_COEF
                tp_distance = sl_distance * TP_SL_RATIO
                sl = entry_price - sl_distance
                tp = entry_price + tp_distance
                position_size = self.calculate_position_size(entry_price, sl)

                self.current_trade = Trade(
                    entry_time=timestamp,
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
                entry_price = current_price * (1 - SLIPPAGE)
                sl_distance = latest_row["ATR"] * SL_COEF
                tp_distance = sl_distance * TP_SL_RATIO
                sl = entry_price + sl_distance
                tp = entry_price - tp_distance
                position_size = self.calculate_position_size(entry_price, sl)

                self.current_trade = Trade(
                    entry_time=timestamp,
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
            trade_duration = timestamp - self.current_trade.EntryTime
            if trade_duration >= MAX_TRADE_DURATION:
                exit_price = current_price
                print(
                    f"Time-based exit for {self.current_trade.Type} trade at {exit_price:.2f}"
                )

            if exit_price is not None:
                self.current_trade.ExitTime = timestamp
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
                self.account_balance += self.current_trade.PnL

                self.trades.append(self.current_trade)
                print(
                    f"Exited {self.current_trade.Type} at {exit_price:.2f} with PnL: {self.current_trade.PnL:.2f}"
                )
                print(f"New account balance: {self.account_balance:.2f}")
                self.current_trade = None

    def update_chart(self):
        """Update the chart with the latest data, indicators, and trades."""
        if self.data is not None and not self.data.empty:
            # Prepare data for plotting
            df = self.data.copy()
            if self.current_candle is not None:
                # Include the current candle
                current_candle_df = pd.DataFrame(
                    {
                        "Open": [self.current_candle["open"]],
                        "High": [self.current_candle["high"]],
                        "Low": [self.current_candle["low"]],
                        "Close": [self.current_candle["close"]],
                        "Volume": [self.current_candle["volume"]],
                    },
                    index=[pd.to_datetime(self.current_candle["timestamp"], unit="s")],
                )
                df = pd.concat([df, current_candle_df])
            df.reset_index(inplace=True)
            df["timestamp"] = (
                df["timestamp"].astype(np.int64) // 10**9
            )  # Convert to UNIX timestamp in seconds

            # Update candlestick data
            candlestick_data = df[
                ["timestamp", "Open", "Close", "Low", "High"]
            ].values.tolist()
            self.candlestick_item.set_data(candlestick_data)

            # Update EMA curves
            timestamps = df["timestamp"].values
            self.ema_fast_curve.setData(timestamps, df["EMA_fast"].values)
            self.ema_slow_curve.setData(timestamps, df["EMA_slow"].values)

            # Update Nadaraya-Watson envelopes
            self.upper_envelope_curve.setData(timestamps, df["Upper_Envelope"].values)
            self.lower_envelope_curve.setData(timestamps, df["Lower_Envelope"].values)

            # Plot trade entries and exits
            self.plot_trades()

            # Adjust the view to show the latest data
            if len(df) > 100:
                self.graph_widget.setXRange(
                    df["timestamp"].iloc[-100], df["timestamp"].iloc[-1]
                )
            else:
                self.graph_widget.setXRange(
                    df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
                )

    def plot_trades(self):
        """Plot trade entry and exit points on the chart."""
        # Remove existing trade markers and lines
        for item in self.trade_markers + self.level_lines:
            self.graph_widget.removeItem(item)
        self.trade_markers = []
        self.level_lines = []

        for trade in self.trades:
            # Entry point
            entry_timestamp = int(trade.EntryTime.timestamp())
            entry_marker = pg.ScatterPlotItem(
                x=[entry_timestamp],
                y=[trade.EntryPrice],
                brush="g" if trade.Type == "Long" else "r",
                symbol="t" if trade.Type == "Long" else "t1",
                size=10,
                pen=pg.mkPen("k"),
            )
            self.graph_widget.addItem(entry_marker)
            self.trade_markers.append(entry_marker)

            # SL and TP levels
            sl_line = pg.InfiniteLine(
                pos=trade.InitialSL,
                angle=0,
                pen=pg.mkPen("r", style=QtCore.Qt.DashLine),
            )
            tp_line = pg.InfiniteLine(
                pos=trade.InitialTP,
                angle=0,
                pen=pg.mkPen("g", style=QtCore.Qt.DashLine),
            )
            self.graph_widget.addItem(sl_line)
            self.graph_widget.addItem(tp_line)
            self.level_lines.extend([sl_line, tp_line])

            # Exit point
            if trade.ExitTime and trade.ExitPrice:
                exit_timestamp = int(trade.ExitTime.timestamp())
                exit_marker = pg.ScatterPlotItem(
                    x=[exit_timestamp],
                    y=[trade.ExitPrice],
                    brush="g" if trade.PnL > 0 else "r",
                    symbol="o",
                    size=10,
                    pen=pg.mkPen("k"),
                )
                self.graph_widget.addItem(exit_marker)
                self.trade_markers.append(exit_marker)
        # Plot current trade levels
        if self.current_trade:
            # Entry point
            entry_timestamp = int(self.current_trade.EntryTime.timestamp())
            entry_marker = pg.ScatterPlotItem(
                x=[entry_timestamp],
                y=[self.current_trade.EntryPrice],
                brush="g" if self.current_trade.Type == "Long" else "r",
                symbol="t" if self.current_trade.Type == "Long" else "t1",
                size=10,
                pen=pg.mkPen("k"),
            )
            self.graph_widget.addItem(entry_marker)
            self.trade_markers.append(entry_marker)

            # Current SL and TP levels (Trailing stops)
            sl_line = pg.InfiniteLine(
                pos=self.current_trade.CurrentSL,
                angle=0,
                pen=pg.mkPen("r", style=QtCore.Qt.DashLine),
            )
            tp_line = pg.InfiniteLine(
                pos=self.current_trade.CurrentTP,
                angle=0,
                pen=pg.mkPen("g", style=QtCore.Qt.DashLine),
            )
            self.graph_widget.addItem(sl_line)
            self.graph_widget.addItem(tp_line)
            self.level_lines.extend([sl_line, tp_line])

    def mouseMoved(self, evt):
        """Update crosshair position."""
        pos = evt[0]
        if self.graph_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.graph_widget.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def closeEvent(self, event):
        """Handle the window close event."""
        self.ws_thread.stop()
        self.ws_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = BTCLiveChart()
    main_window.show()
    sys.exit(app.exec_())
