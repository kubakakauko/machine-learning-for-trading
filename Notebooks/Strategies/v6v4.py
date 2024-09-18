import json
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyqtgraph as pg
import requests
# import pandas_ta as ta
import talib as ta
import websocket
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout,
                             QLabel, QMainWindow, QSpinBox, QVBoxLayout,
                             QWidget)
from statsmodels.nonparametric.kernel_regression import KernelReg

"""
Strategy Overview:
------------------
This algorithmic trading strategy is based on technical indicators, specifically EMAs (Exponential Moving Averages), ATR (Average True Range), and Nadaraya-Watson kernel regression (for price smoothing and prediction). The strategy aims to trade the BTC/USDT pair based on crossovers between fast and slow EMAs, along with confirmation from price deviations beyond the Nadaraya-Watson envelopes. It also incorporates trailing stop-losses and take-profits to lock in profits while minimizing losses.

1. **Entry Signals:**
   - **EMA Crossovers**: The strategy looks for a crossover between a fast (40-period) and slow (50-period) EMA to generate signals:
     - **Long Entry (Buy)**: A long signal is triggered when the fast EMA crosses above the slow EMA, and the price dips below the lower Nadaraya-Watson envelope (indicating an oversold condition).
     - **Short Entry (Sell)**: A short signal is triggered when the fast EMA crosses below the slow EMA, and the price exceeds the upper Nadaraya-Watson envelope (indicating an overbought condition).

2. **Stop Loss Calculation:**
   - The stop-loss (SL) is based on the volatility of the market, measured by the ATR (14-period). The `SL_COEF` defines how far the stop-loss is placed from the entry price. Specifically:
     - For long trades, the stop-loss is set at `entry_price - SL_COEF * ATR`.
     - For short trades, the stop-loss is set at `entry_price + SL_COEF * ATR`.
   - This ensures the stop-loss is dynamic and adjusts to the current market volatility.

3. **Take Profit Calculation:**
   - The take-profit (TP) is set as a multiple of the stop-loss distance. The ratio is determined by `TP_SL_RATIO`:
     - TP = SL distance * TP_SL_RATIO (in this case, 2x the SL distance).
   - This ensures the take-profit target is proportional to the risk being taken on the trade, aiming for a 2:1 reward-to-risk ratio.

4. **Trailing Stop Loss (TRAIL_PERCENT):**
   - After entering a trade, a trailing stop is applied to protect profits as the price moves in the favorable direction. The trailing stop moves as follows:
     - For long trades, the stop-loss is adjusted upwards when the price rises by 0.12% (`TRAIL_PERCENT = 0.0012`).
     - For short trades, the stop-loss is adjusted downwards when the price drops by 0.12%.
   - This ensures that the stop-loss moves dynamically with the market, locking in profits while limiting losses.

5. **Slippage Handling (SLIPPAGE):**
   - A slippage factor (`SLIPPAGE = 0.0005`, or 0.05%) is incorporated to account for the price deviation during trade execution. This slippage is added to the entry price for long trades and subtracted for short trades. It helps simulate real-world trading conditions where the execution price can slightly differ from the intended price.

6. **Risk Management (RISK_PER_TRADE):**
   - The strategy risks a fixed percentage of the account balance per trade (`RISK_PER_TRADE = 0.001`, or 0.1%). This is the maximum capital you are willing to lose on any single trade.
   - **Position Size Calculation**: The position size is calculated based on the distance between the entry price and stop-loss, ensuring that if the stop-loss is hit, the total loss will not exceed the defined risk (0.1% of the account balance). The formula `position_size = amount_at_risk / price_difference` adjusts the size of the trade based on volatility (measured by the ATR) and the stop-loss distance.
7. **Trade Duration Limit:**
   - A maximum trade duration is enforced (`MAX_TRADE_DURATION = 6 hours`), after which the trade is closed automatically, regardless of the price movement. This prevents holding positions too long in volatile markets where the initial strategy logic may no longer apply.
8. **Impact of Indicators and Variables:**
   - **ATR (Volatility)**: Higher ATR values increase the stop-loss distance, meaning the strategy tolerates more price movement before exiting the trade. Lower ATR values result in tighter stop-losses.
   - **EMA Fast/Slow**: The periods of the EMAs control the sensitivity of the trend detection. Shorter EMAs react faster to price changes but can generate more false signals, while longer EMAs are slower but more reliable.
   - **TRAIL_PERCENT**: This variable directly impacts how closely the trailing stop follows the price. A smaller value will result in a tighter stop, locking in profits sooner but potentially stopping out prematurely. A larger value will give the trade more room but may allow profits to decrease before the stop is triggered.
   - **SL_COEF and TP_SL_RATIO**: These determine the aggressiveness of the strategy in terms of stop-loss and take-profit levels. A higher `SL_COEF` results in a larger stop-loss, giving the trade more room, while a higher `TP_SL_RATIO` aims for larger take-profit targets, enhancing the risk-to-reward ratio.
   - **RISK_PER_TRADE**: Defines the fixed percentage of the account balance to be risked per trade. This ensures that the overall account remains protected by limiting potential losses to a small fraction of the balance.
"""

# TODO:
# 1) Add logging of the individual indicators and singnals for each trade to verify corectness. (This is because on lower timefarmes there are some inconsistencies with how signals are generated).
# 2)


# Configuration Parameters
SYMBOL = "BTCUSDT"
BANDWIDTH = 7  # Bandwidth for Nadaraya-Watson
EMA_SLOW = 50
EMA_FAST = 40
ATR_LENGTH = 14  # Period for calculating the Average True Range (ATR) indicator (measuring volatility)
BACKCANDLES = 7  # Number of previous candles to consider for trading signals
SL_COEF = 1.5  # Stop Loss coefficient: distance from entry price to stop-loss is 1.5 times the ATR
TP_SL_RATIO = 2.0  # Take Profit to Stop Loss ratio: TP is 2x the SL distance
SLIPPAGE = 0.0005  # Assumed slippage rate for entry/exit prices (0.05%)
NW_MULT = 2  # Multiplier for Nadaraya-Watson envelope distance (standard deviation from the fitted price)

TRAIL_PERCENT = 0.0012  # Trailing percentage for both SL and TP (0.12%)
# TRAIL_PERCENT defines the percentage price movement required to trigger the adjustment of the trailing stop.
# For long trades, the trailing stop (SL) is updated to be 0.12% below the current price.
# For short trades, the trailing stop is updated to be 0.12% above the current price.

# Account and Risk Management
ACCOUNT_BALANCE = 100  # Starting account balance in USD for simulations
RISK_PER_TRADE = (
    0.1  # Risk 10% of the account balance per trade (based on stop-loss distance)
)
MAX_TRADE_DURATION = timedelta(
    hours=6
)  # Maximum trade duration: close the trade after 6 hours


# Trade class to manage individual trades
class Trade:
    # This is the constructor for the Trade class
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


# WebSocket thread to receive live data
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
        data = json.loads(message)
        self.message_received.emit(data)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
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
        self.trade_counter = 0  # For assigning Trade IDs

        # Initialize data DataFrame
        self.data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        self.data.index.name = "timestamp"

        # Initialize settings
        self.enter_on_signal = False  # Toggle for entering trades before candle close

        # Default timeframe and historical days
        self.TIMEFRAME = "1m"
        self.HISTORICAL_DAYS = 2

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

        # Add controls for timeframe selection and historical days
        controls_layout = QHBoxLayout()
        self.timeframe_combobox = QComboBox()
        self.timeframe_combobox.addItems(["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
        self.timeframe_combobox.setCurrentText(self.TIMEFRAME)  # default value
        self.timeframe_combobox.currentTextChanged.connect(self.on_timeframe_changed)
        controls_layout.addWidget(QLabel("Timeframe:"))
        controls_layout.addWidget(self.timeframe_combobox)

        self.historical_days_spinbox = QSpinBox()
        self.historical_days_spinbox.setRange(1, 30)
        self.historical_days_spinbox.setValue(self.HISTORICAL_DAYS)  # default value
        self.historical_days_spinbox.valueChanged.connect(
            self.on_historical_days_changed
        )
        controls_layout.addWidget(QLabel("Historical Days:"))
        controls_layout.addWidget(self.historical_days_spinbox)

        self.enter_on_signal_checkbox = QCheckBox("Enter Trades Before Candle Close")
        # added
        self.enter_on_signal_checkbox.setChecked(self.enter_on_signal)
        self.enter_on_signal_checkbox.stateChanged.connect(self.toggle_enter_on_signal)
        controls_layout.addWidget(self.enter_on_signal_checkbox)
        layout.addLayout(controls_layout)

        # Add label to display candle information
        self.candle_info_label = QLabel()
        layout.addWidget(self.candle_info_label)

    def toggle_enter_on_signal(self, state):
        self.enter_on_signal = state == Qt.Checked
        # Reset trades and reprocess historical trades
        self.reset_trades_and_reprocess()

    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(value).strftime("%H:%M:%S") for value in values]

    def on_timeframe_changed(self, timeframe):
        self.TIMEFRAME = timeframe
        self.reset_data_and_fetch_new()

    def on_historical_days_changed(self, days):
        self.HISTORICAL_DAYS = days
        self.reset_data_and_fetch_new()

    def reset_data_and_fetch_new(self):
        # Stop the websocket thread and update timer
        self.ws_thread.stop()
        self.ws_thread.wait()
        self.update_timer.stop()
        # Reset data
        self.data = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        self.data.index.name = "timestamp"
        self.current_candle = None
        # Reset trades
        self.trades = []
        self.current_trade = None
        self.trade_counter = 0
        # Fetch new historical data
        self.fetch_historical_data()
        # Restart websocket and update timer
        self.setup_websocket()
        self.update_timer.start(1000)

    def reset_trades_and_reprocess(self):
        # Reset trades
        self.trades = []
        self.current_trade = None
        self.trade_counter = 0
        # Reprocess historical trades
        self.process_historical_trades()

    def timeframe_to_minutes(self, timeframe):
        if timeframe.endswith("m"):
            return int(timeframe[:-1])
        elif timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith("d"):
            return int(timeframe[:-1]) * 60 * 24
        else:
            return 1  # default to 1 minute

    def timeframe_to_seconds(self, timeframe):
        return self.timeframe_to_minutes(self.TIMEFRAME) * 60

    def calculate_candles_limit(self):
        timeframe_minutes = self.timeframe_to_minutes(self.TIMEFRAME)
        candles_per_day = 24 * 60 / timeframe_minutes
        limit = int(candles_per_day * self.HISTORICAL_DAYS)
        # Binance API limit is 1000 candles per request
        if limit > 1000:
            limit = 1000
        return limit

    def fetch_historical_data(self):
        """Fetch historical data from Binance and initialize the data DataFrame."""
        limit = self.calculate_candles_limit()
        url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={self.TIMEFRAME}&limit={limit}"
        response = requests.get(url)
        klines = response.json()

        data_list = []
        for kline in klines:
            timestamp = int(int(kline[0]) / 1000)
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
        self.data.index.name = "timestamp"

        # Calculate indicators and generate initial signals
        self.calculate_indicators()
        self.generate_signals()

        # Process historical data to generate trades
        self.process_historical_trades()

    def process_historical_trades(self):
        """Process historical data to generate trades."""
        for idx in range(len(self.data)):
            latest_row = self.data.iloc[idx]
            timestamp = self.data.index[idx]
            self.update_trades(latest_row, timestamp)

    def setup_websocket(self):
        """Set up the WebSocket connection to receive live data."""
        self.ws_thread = WebSocketThread(
            f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@kline_{self.TIMEFRAME}"
        )
        self.ws_thread.message_received.connect(self.process_websocket_message)
        self.ws_thread.start()

    def process_websocket_message(self, message):
        """Process incoming kline data from the WebSocket."""
        data = message["k"]
        is_candle_closed = data["x"]
        timestamp = int(data["t"] / 1000)  # Kline start time
        open_price = float(data["o"])
        high_price = float(data["h"])
        low_price = float(data["l"])
        close_price = float(data["c"])
        volume = float(data["v"])

        candle_data = {
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": close_price,
            "Volume": volume,
        }
        candle_index = pd.to_datetime(timestamp, unit="s")

        if is_candle_closed:
            # Check if this timestamp is already in data
            if candle_index in self.data.index:
                # Update the existing row
                self.data.loc[candle_index] = candle_data
            else:
                # Append the candle to the data
                new_row = pd.DataFrame(candle_data, index=[candle_index])
                self.data = pd.concat([self.data, new_row])
            # Ensure index name is 'timestamp'
            self.data.index.name = "timestamp"

            # Recalculate indicators and manage trades
            self.calculate_indicators()
            self.generate_signals()
            latest_row = self.data.loc[candle_index]
            self.update_trades(latest_row, candle_index)
        else:
            # Update current candle
            self.current_candle = candle_data
            self.current_candle["timestamp"] = candle_index
            # Update trades with the latest price
            if self.current_trade or self.enter_on_signal:
                latest_row = pd.Series(
                    {
                        "Close": close_price,
                        "ATR": (
                            self.data["ATR"].iloc[-1] if not self.data.empty else np.nan
                        ),
                        "Total_Signal": (
                            self.data["Total_Signal"].iloc[-1]
                            if not self.data.empty
                            else 0
                        ),
                    }
                )
                self.update_trades(latest_row, candle_index, live_update=True)

    def calculate_indicators(self):
        """Calculate technical indicators: EMA, ATR, and Nadaraya-Watson envelopes."""
        # Ensure there is enough data to calculate the indicators
        if len(self.data) < max(EMA_FAST, EMA_SLOW, ATR_LENGTH):
            return  # Not enough data

        # Calculate EMA using TA-Lib
        self.data["EMA_fast"] = ta.EMA(self.data["Close"], timeperiod=EMA_FAST)
        self.data["EMA_slow"] = ta.EMA(self.data["Close"], timeperiod=EMA_SLOW)

        # Calculate ATR using TA-Lib
        self.data["ATR"] = ta.ATR(
            self.data["High"].values,
            self.data["Low"].values,
            self.data["Close"].values,
            timeperiod=ATR_LENGTH,
        )

        # Calculate Nadaraya-Watson envelopes
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

        if "timestamp" not in df.columns:
            df["timestamp"] = self.data.index

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
            # Save current view range
            x_range, y_range = self.graph_widget.viewRange()

            # Prepare data for plotting
            df = self.data.copy()
            if self.current_candle is not None:
                # Include the current candle
                current_candle_df = pd.DataFrame(
                    [self.current_candle],
                    index=[self.current_candle["timestamp"]],
                )
                # Remove 'timestamp' column to avoid duplication
                current_candle_df.drop(columns=["timestamp"], inplace=True)
                df = pd.concat([df, current_candle_df])

            df.reset_index(inplace=True)
            # Remove duplicate 'timestamp' column if it exists
            if "timestamp" in df.columns:
                df = df.loc[:, ~df.columns.duplicated()]
            df.rename(columns={"index": "timestamp"}, inplace=True)
            df["timestamp"] = df["timestamp"].astype(np.int64) // 10**9

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

            # Restore view range
            self.graph_widget.setXRange(*x_range, padding=0)
            self.graph_widget.setYRange(*y_range, padding=0)

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
        """Update crosshair position and display candle information."""
        pos = evt[0]
        if self.graph_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.graph_widget.plotItem.vb.mapSceneToView(pos)
            x = mousePoint.x()
            y = mousePoint.y()
            self.vLine.setPos(x)
            self.hLine.setPos(y)

            # Find the nearest timestamp in data
            timestamp = int(x)
            df = self.data.copy()
            if self.current_candle is not None:
                current_candle_df = pd.DataFrame(
                    [self.current_candle],
                    index=[self.current_candle["timestamp"]],
                )
                df = pd.concat([df, current_candle_df])

            df.reset_index(inplace=True)
            df["timestamp"] = df["timestamp"].astype(np.int64) // 10**9

            # Find the closest timestamp
            if not df.empty:
                idx = (df["timestamp"] - timestamp).abs().idxmin()
                candle = df.iloc[idx]
                candle_info = f"Time: {datetime.fromtimestamp(candle['timestamp'])} | O:{candle['Open']:.2f} H:{candle['High']:.2f} L:{candle['Low']:.2f} C:{candle['Close']:.2f}"
                self.candle_info_label.setText(candle_info)

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
