import sys
import json
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QMessageBox,
)
import pyqtgraph as pg
from statsmodels.nonparametric.kernel_regression import KernelReg
import websocket
import threading
import time

# Configuration Parameters
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"  # Fetch 1-minute candles
AGG_INTERVAL_MINUTES = 9  # Desired timeframe to aggregate to
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
HISTORICAL_DAYS = 7  # Number of historical days to fetch

# Account and Risk Management
ACCOUNT_BALANCE = 10000  # Example starting balance in USD
RISK_PER_TRADE = 0.01  # Risk 1% of account per trade
MAX_TRADE_DURATION = timedelta(hours=6)  # Close trades after 6 hours

# Timeframe in seconds
TIMEFRAME_SECONDS = 60  # 1 minute


# Trade class to manage individual trades
class Trade:
    def __init__(
        self, trade_id, entry_time, entry_price, trade_type, sl, tp, position_size
    ):
        self.TradeID = trade_id
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
            "TradeID": self.TradeID,
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
        self.connected = False
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open,
            )
            self.ws.run_forever()
            time.sleep(5)  # Wait before reconnecting

    def on_open(self, ws):
        self.connected = True
        print("WebSocket connection opened")

    def on_message(self, ws, message):
        trade = json.loads(message)
        self.message_received.emit(trade)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        if not self.stop_flag.is_set():
            print(
                "WebSocket connection closed unexpectedly, attempting to reconnect..."
            )
        else:
            print("WebSocket connection closed")

    def stop(self):
        self.stop_flag.set()
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
        self.resize(1400, 800)

        # Initialize account balance and trade variables
        self.initial_balance = ACCOUNT_BALANCE
        self.account_balance = ACCOUNT_BALANCE
        self.trades = []
        self.current_trade = None
        self.trade_counter = 0  # For assigning Trade IDs

        # Initialize data DataFrame
        self.data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        # Initialize settings
        self.enter_on_signal = False  # Toggle for entering trades before candle close

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

        # For keeping track of chart zoom/pan
        self.initial_view_set = False

        # Initialize performance metrics
        self.performance_metrics = {}

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
            pen=pg.mkPen("r", width=1, style=Qt.DashLine)
        )
        self.graph_widget.addItem(self.upper_envelope_curve)
        self.lower_envelope_curve = pg.PlotCurveItem(
            pen=pg.mkPen("g", width=1, style=Qt.DashLine)
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

        # Add controls
        controls_layout = QHBoxLayout()

        self.enter_on_signal_checkbox = QCheckBox("Enter Trades Before Candle Close")
        self.enter_on_signal_checkbox.stateChanged.connect(self.toggle_enter_on_signal)
        controls_layout.addWidget(self.enter_on_signal_checkbox)

        # Button to export trade log
        self.export_button = QPushButton("Export Trade Log")
        self.export_button.clicked.connect(self.export_trade_log)
        controls_layout.addWidget(self.export_button)

        # Input for historical days
        self.historical_days_input = QLineEdit(str(HISTORICAL_DAYS))
        self.historical_days_input.setFixedWidth(50)
        controls_layout.addWidget(QLabel("Historical Days:"))
        controls_layout.addWidget(self.historical_days_input)

        # Button to reload data
        self.reload_button = QPushButton("Reload Data")
        self.reload_button.clicked.connect(self.reload_data)
        controls_layout.addWidget(self.reload_button)

        layout.addLayout(controls_layout)

        # Add performance metrics display
        self.metrics_label = QLabel()
        layout.addWidget(self.metrics_label)

    def toggle_enter_on_signal(self, state):
        self.enter_on_signal = state == Qt.Checked
        # Reprocess historical trades when toggled
        self.process_historical_trades()

    def tickStrings(self, values, scale, spacing):
        return [
            datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M") for value in values
        ]

    def fetch_historical_data(self):
        """Fetch historical data from Binance and initialize the data DataFrame."""
        limit_per_request = 1000
        total_days = int(self.historical_days_input.text())
        total_minutes = total_days * 24 * 60
        num_requests = (
            total_minutes // int(TIMEFRAME_SECONDS / 60) + limit_per_request - 1
        ) // limit_per_request

        end_time = int(datetime.utcnow().timestamp() * 1000)
        data_list = []

        for i in range(num_requests):
            params = {
                "symbol": SYMBOL,
                "interval": TIMEFRAME,
                "limit": limit_per_request,
                "endTime": end_time,
            }
            try:
                response = requests.get(
                    "https://api.binance.com/api/v3/klines", params=params
                )
                klines = response.json()
                if not klines:
                    break
                for kline in klines:
                    timestamp = int(int(kline[0]) / 1000)
                    open_price = float(kline[1])
                    high_price = float(kline[2])
                    low_price = float(kline[3])
                    close_price = float(kline[4])
                    volume = float(kline[5])
                    data_list.append(
                        [
                            timestamp,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                            volume,
                        ]
                    )
                end_time = klines[0][0] - 1  # Prepare for next batch
            except Exception as e:
                print(f"Error fetching historical data: {e}")
                break

        data_list.reverse()  # Oldest first

        self.data = pd.DataFrame(
            data_list, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"], unit="s")
        self.data.set_index("timestamp", inplace=True)

        # Resample data to AGG_INTERVAL_MINUTES if necessary
        if AGG_INTERVAL_MINUTES > 1:
            self.data = (
                self.data.resample(f"{AGG_INTERVAL_MINUTES}min")
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

        # Process historical data to generate trades
        self.process_historical_trades()

    def reload_data(self):
        """Reload data based on new historical days."""
        self.data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        self.trades = []
        self.current_trade = None
        self.trade_counter = 0
        self.initial_view_set = False
        self.fetch_historical_data()

    def process_historical_trades(self):
        """Process historical data to generate trades."""
        self.trades = []
        self.current_trade = None
        for idx in range(len(self.data)):
            latest_row = self.data.iloc[idx]
            timestamp = self.data.index[idx]
            self.update_trades(latest_row, timestamp)

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
                new_row.index.name = "timestamp"
                self.data = pd.concat([self.data, new_row])
                # After updating data, recalculate indicators and manage trades
                self.calculate_indicators()
                self.generate_signals()
                latest_row = self.data.iloc[-1]
                timestamp_dt = self.data.index[-1]
                self.update_trades(latest_row, timestamp_dt)
                self.calculate_performance_metrics()

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

            # Update trades with the latest price
            if self.current_trade or self.enter_on_signal:
                latest_row = pd.Series(
                    {
                        "Close": price,
                        "ATR": (
                            self.data["ATR"].iloc[-1] if not self.data.empty else np.nan
                        ),
                    }
                )
                timestamp_dt = pd.to_datetime(timestamp, unit="s")
                self.update_trades(latest_row, timestamp_dt, live_update=True)
                self.calculate_performance_metrics()

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

    def update_trades(self, latest_row, timestamp, live_update=False):
        """Manage trades based on signals and price movements."""
        signal = latest_row.get("Total_Signal", 0)
        current_price = latest_row["Close"]

        if self.current_trade is None:
            if signal == 2:  # Buy signal
                if live_update and not self.enter_on_signal:
                    return  # Wait for candle close
                entry_price = current_price * (1 + SLIPPAGE)
                sl_distance = latest_row["ATR"] * SL_COEF
                tp_distance = sl_distance * TP_SL_RATIO
                sl = entry_price - sl_distance
                tp = entry_price + tp_distance
                position_size = self.calculate_position_size(entry_price, sl)

                self.trade_counter += 1
                self.current_trade = Trade(
                    trade_id=self.trade_counter,
                    entry_time=timestamp,
                    entry_price=entry_price,
                    trade_type="Long",
                    sl=sl,
                    tp=tp,
                    position_size=position_size,
                )
                print(
                    f"Trade {self.current_trade.TradeID}: Entered Long at {entry_price:.2f} with SL: {sl:.2f} and TP: {tp:.2f}"
                )
            elif signal == 1:  # Sell signal
                if live_update and not self.enter_on_signal:
                    return  # Wait for candle close
                entry_price = current_price * (1 - SLIPPAGE)
                sl_distance = latest_row["ATR"] * SL_COEF
                tp_distance = sl_distance * TP_SL_RATIO
                sl = entry_price + sl_distance
                tp = entry_price - tp_distance
                position_size = self.calculate_position_size(entry_price, sl)

                self.trade_counter += 1
                self.current_trade = Trade(
                    trade_id=self.trade_counter,
                    entry_time=timestamp,
                    entry_price=entry_price,
                    trade_type="Short",
                    sl=sl,
                    tp=tp,
                    position_size=position_size,
                )
                print(
                    f"Trade {self.current_trade.TradeID}: Entered Short at {entry_price:.2f} with SL: {sl:.2f} and TP: {tp:.2f}"
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
                    f"Trade {self.current_trade.TradeID}: Time-based exit for {self.current_trade.Type} trade at {exit_price:.2f}"
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
                    f"Trade {self.current_trade.TradeID}: Exited {self.current_trade.Type} at {exit_price:.2f} with PnL: {self.current_trade.PnL:.2f}"
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
                current_candle_df.index.name = "timestamp"
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

            # Adjust the view to show the latest data only once
            if not self.initial_view_set:
                if len(df) > 100:
                    self.graph_widget.setXRange(
                        df["timestamp"].iloc[-100], df["timestamp"].iloc[-1]
                    )
                else:
                    self.graph_widget.setXRange(
                        df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
                    )
                self.initial_view_set = True

            # Update performance metrics display
            self.update_performance_metrics_display()

    def plot_trades(self):
        """Plot trade entry and exit points on the chart."""
        # Remove existing trade markers and lines
        for item in self.trade_markers + self.level_lines:
            self.graph_widget.removeItem(item)
        self.trade_markers = []
        self.level_lines = []

        current_time = int(datetime.now().timestamp())

        for trade in self.trades:
            # Entry point
            entry_timestamp = int(trade.EntryTime.timestamp())
            # Exit timestamp
            exit_timestamp = (
                int(trade.ExitTime.timestamp()) if trade.ExitTime else current_time
            )

            # Entry marker
            entry_marker = pg.ScatterPlotItem(
                x=[entry_timestamp],
                y=[trade.EntryPrice],
                brush="g" if trade.Type == "Long" else "r",
                symbol=(
                    "t" if trade.Type == "Long" else "t1"
                ),  # Use 't1' for downward triangle
                size=10,
                pen=pg.mkPen("k"),
            )
            self.graph_widget.addItem(entry_marker)
            self.trade_markers.append(entry_marker)

            # SL and TP lines
            sl_line = pg.PlotCurveItem(
                x=[entry_timestamp, exit_timestamp],
                y=[trade.InitialSL, trade.InitialSL],
                pen=pg.mkPen("r", style=Qt.DashLine),
            )
            tp_line = pg.PlotCurveItem(
                x=[entry_timestamp, exit_timestamp],
                y=[trade.InitialTP, trade.InitialTP],
                pen=pg.mkPen("g", style=Qt.DashLine),
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
            # Current time
            exit_timestamp = int(datetime.now().timestamp())

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
            sl_line = pg.PlotCurveItem(
                x=[entry_timestamp, exit_timestamp],
                y=[self.current_trade.CurrentSL, self.current_trade.CurrentSL],
                pen=pg.mkPen("r", style=Qt.DashLine),
            )
            tp_line = pg.PlotCurveItem(
                x=[entry_timestamp, exit_timestamp],
                y=[self.current_trade.CurrentTP, self.current_trade.CurrentTP],
                pen=pg.mkPen("g", style=Qt.DashLine),
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

    def export_trade_log(self):
        """Export trade log to CSV."""
        if not self.trades:
            QMessageBox.information(self, "No Trades", "There are no trades to export.")
            return
        df = pd.DataFrame([trade.to_dict() for trade in self.trades])
        df.to_csv("trade_log.csv", index=False)
        QMessageBox.information(
            self, "Export Successful", "Trade log exported to trade_log.csv"
        )

    def calculate_performance_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return
        df = pd.DataFrame([trade.to_dict() for trade in self.trades])
        df["CumPnL"] = df["PnL"].cumsum()
        total_return = df["PnL"].sum()
        returns = df["PnL"]

        # Sharpe Ratio
        if returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(returns))
        else:
            sharpe_ratio = 0.0

        # Maximum Drawdown
        cum_returns = df["CumPnL"]
        drawdown = cum_returns - cum_returns.cummax()
        max_drawdown = drawdown.min()

        # Win Rate
        wins = df["PnL"] > 0
        win_rate = wins.mean()

        # Profit Factor
        gross_profit = df.loc[df["PnL"] > 0, "PnL"].sum()
        gross_loss = -df.loc[df["PnL"] < 0, "PnL"].sum()
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

        self.performance_metrics = {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
        }

    def update_performance_metrics_display(self):
        """Update the performance metrics display."""
        if not self.performance_metrics:
            return
        metrics_text = f"""
        Total Return: {self.performance_metrics['Total Return']:.2f}
        Sharpe Ratio: {self.performance_metrics['Sharpe Ratio']:.2f}
        Max Drawdown: {self.performance_metrics['Max Drawdown']:.2f}
        Win Rate: {self.performance_metrics['Win Rate'] * 100:.2f}%
        Profit Factor: {self.performance_metrics['Profit Factor']:.2f}
        """
        self.metrics_label.setText(metrics_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = BTCLiveChart()
    main_window.show()
    sys.exit(app.exec_())
