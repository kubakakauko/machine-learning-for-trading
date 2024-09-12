# %% [markdown]
# # Nadaraya Watson Envelope with EMA and Trailing Stop / TP

# %% [markdown]
# ## 1. Imports

# %%
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.nonparametric.kernel_regression import KernelReg
from backtesting import Strategy, Backtest
import websocket
import json
import datetime
from datetime import datetime
from datetime import timedelta

# %% [markdown]
# ## 2. Fetch Data
# Fetching data from binance. There is an potion to fetch live data however this is currenty disabled.

# %%
def fetch_historical_data(symbol='BTCUSDT', interval='15m', start_date=None, end_date=None):
    """
    Fetch historical data from Binance API for a specified date range.

    Args:
    symbol (str): Trading pair symbol (default: 'BTCUSDT')
    interval (str): Candlestick interval (default: '15m')
    start_date (str): Start date in format 'YYYY-MM-DD'
    end_date (str): End date in format 'YYYY-MM-DD'

    Returns:
    pd.DataFrame: Historical price data
    """
    base_url = 'https://api.binance.com'
    endpoint = f'/api/v3/klines'

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # Default to 1 year ago
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')  # Default to today

    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    all_candles = []
    while start_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000  # Maximum allowed by Binance
        }

        response = requests.get(f'{base_url}{endpoint}', params=params)
        data = response.json()

        all_candles.extend(data)
        if len(data) == 0:
            break

        start_ts = data[-1][0] + 1  # Start from the next candle

    df = pd.DataFrame(all_candles, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


# WebSocket connection to Binance for live data
def on_message(ws, message):
    global df_live
    data = json.loads(message)
    candle = data['k']
    is_candle_closed = candle['x']

    if is_candle_closed:
        timestamp = pd.to_datetime(candle['t'], unit='ms')
        open_price = float(candle['o'])
        high_price = float(candle['h'])
        low_price = float(candle['l'])
        close_price = float(candle['c'])
        volume = float(candle['v'])

        new_row = pd.DataFrame({
            'Open': [open_price],
            'High': [high_price],
            'Low': [low_price],
            'Close': [close_price],
            'Volume': [volume]
        }, index=[timestamp])

        df_live = pd.concat([df_live, new_row])
        print(df_live.tail(1))  # Print the last row for debugging

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    print("### connection opened ###")

def run_live_stream():
    symbol = 'btcusdt'
    socket = f"wss://stream.binance.com:9443/ws/{symbol}@kline_15m"
    ws = websocket.WebSocketApp(socket,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# %% [markdown]
# ## 3. Calculate Indicators Nadaraya Watson Envelope and EMAs

# %%
def calculate_indicators(df, ema_slow= 50, ema_fast= 40, atr_length= 7):
    """
    Calculate technical indicators: EMA and ATR.

    Args:
    df (pd.DataFrame): Price data

    Returns:
    pd.DataFrame: DataFrame with added indicator columns
    """
    df["EMA_slow"] = ta.ema(df.Close, length=ema_slow)
    df["EMA_fast"] = ta.ema(df.Close, length=ema_fast)
    df['ATR'] = ta.atr(df.High, df.Low, df.Close, length=atr_length)
    return df

def calculate_nadaraya_watson(df, bandwidth=0.1):
    """
    Calculate Nadaraya-Watson envelopes.

    Args:
    df (pd.DataFrame): Price data with indicators
    bandwidth (float): Bandwidth for the kernel, controls smoothness

    Returns:
    pd.DataFrame: DataFrame with added Nadaraya-Watson columns
    """
    # Ensure we're working with a copy to avoid overwriting the original data
    df = df.copy()

    # Convert datetime index to numerical values
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values

    # Perform Nadaraya-Watson kernel regression
    model = KernelReg(endog=y, exog=X, var_type='c', bw=[bandwidth])
    fitted_values, _ = model.fit(X)

    # Store the fitted values
    df['NW_Fitted'] = fitted_values

    # Calculate the residuals
    residuals = df['Close'] - fitted_values

    # Calculate the standard deviation of the residuals
    std_dev = np.std(residuals)

    # Create the envelopes
    df['Upper_Envelope'] = df['NW_Fitted'] + 2 * std_dev
    df['Lower_Envelope'] = df['NW_Fitted'] - 2 * std_dev

    # Print some debug information
    print("NW_Fitted range:", df['NW_Fitted'].min(), "-", df['NW_Fitted'].max())
    print("Close range:", df['Close'].min(), "-", df['Close'].max())
    print("Std Dev:", std_dev)
    print("Upper_Envelope range:", df['Upper_Envelope'].min(), "-", df['Upper_Envelope'].max())
    print("Lower_Envelope range:", df['Lower_Envelope'].min(), "-", df['Lower_Envelope'].max())

    return df

# %% [markdown]
# ## 4. Get signals from the indicators.

# %%
def ema_signal(df, backcandles):
    """
    Generate EMA crossover signals.

    Args:
    df (pd.DataFrame): Price data with indicators
    backcandles (int): Number of candles to look back for confirming the signal

    Returns:
    pd.DataFrame: DataFrame with added EMA signal column
    """
    above = df['EMA_fast'] > df['EMA_slow']
    below = df['EMA_fast'] < df['EMA_slow']

    above_all = above.rolling(window=backcandles).apply(lambda x: x.all(), raw=True).fillna(0).astype(bool)
    below_all = below.rolling(window=backcandles).apply(lambda x: x.all(), raw=True).fillna(0).astype(bool)

    df['EMASignal'] = 0
    df.loc[above_all, 'EMASignal'] = 2
    df.loc[below_all, 'EMASignal'] = 1

    return df

def total_signal(df):
    """
    Generate total trading signals based on EMA and Nadaraya-Watson envelopes.

    Args:
    df (pd.DataFrame): Price data with indicators and EMA signals

    Returns:
    pd.DataFrame: DataFrame with added total signal column
    """
    condition_buy = (df['EMASignal'] == 2) & (df['Close'] <= df['Lower_Envelope'])
    condition_sell = (df['EMASignal'] == 1) & (df['Close'] >= df['Upper_Envelope'])

    df['Total_Signal'] = 0
    df.loc[condition_buy, 'Total_Signal'] = 2
    df.loc[condition_sell, 'Total_Signal'] = 1

    return df

# %% [markdown]
# ## 5. Define the strategy
# Later used to backtest the strategy.

# %%
class RefinedNWStrategy(Strategy):
    mysize = 1.0
    slcoef = 1.5
    TPSLRatio = 2.0
    slippage = 0.0005
    trail_percent = 0.00618

    def init(self):
        super().init()
        self.signal = self.I(lambda: self.data.Total_Signal)
        self.trades_info = []
        self.current_trade = None

        # Add necessary indicators
        self.atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=14)

    def next(self):
        super().next()

        # Update current trade if exists
        if self.current_trade and self.position:
            self.update_sl_tp()
            self.check_exit()

        if not self.position:
            self.enter_new_trade()

    def enter_new_trade(self):
        if self.signal == 2:  # Buy signal
            entry_price = self.data.Close[-1] * (1 + self.slippage)
            sl, tp = self.calculate_sl_tp(entry_price, True)
            self.buy(size=self.mysize)
            self.current_trade = {
                'EntryTime': self.data.index[-1],
                'EntryPrice': entry_price,
                'Type': 'Long',
                'InitialSL': sl,
                'InitialTP': tp,
                'CurrentSL': sl,
                'CurrentTP': tp,
                'ExitTime': None,
                'ExitPrice': None,
                'PnL': None
            }
        elif self.signal == 1:  # Sell signal
            entry_price = self.data.Close[-1] * (1 - self.slippage)
            sl, tp = self.calculate_sl_tp(entry_price, False)
            self.sell(size=self.mysize)
            self.current_trade = {
                'EntryTime': self.data.index[-1],
                'EntryPrice': entry_price,
                'Type': 'Short',
                'InitialSL': sl,
                'InitialTP': tp,
                'CurrentSL': sl,
                'CurrentTP': tp,
                'ExitTime': None,
                'ExitPrice': None,
                'PnL': None
            }

    def calculate_sl_tp(self, entry_price, is_long):
        sl_distance = self.atr[-1] * self.slcoef
        tp_distance = sl_distance * self.TPSLRatio
        sl = entry_price - sl_distance if is_long else entry_price + sl_distance
        tp = entry_price + tp_distance if is_long else entry_price - tp_distance
        return sl, tp

    def update_sl_tp(self):
        current_price = self.data.Close[-1]
        if self.position.is_long:
            self.current_trade['CurrentSL'] = max(self.current_trade['CurrentSL'], current_price * (1 - self.trail_percent))
            self.current_trade['CurrentTP'] = max(self.current_trade['CurrentTP'], current_price * (1 + self.trail_percent))
        else:
            self.current_trade['CurrentSL'] = min(self.current_trade['CurrentSL'], current_price * (1 + self.trail_percent))
            self.current_trade['CurrentTP'] = min(self.current_trade['CurrentTP'], current_price * (1 - self.trail_percent))

    def check_exit(self):
        exit_price = None
        current_price = self.data.Close[-1]

        if self.current_trade['Type'] == 'Long':
            if current_price <= self.current_trade['CurrentSL']:
                exit_price = self.current_trade['CurrentSL']
            elif current_price >= self.current_trade['CurrentTP']:
                exit_price = self.current_trade['CurrentTP']
        elif self.current_trade['Type'] == 'Short':
            if current_price >= self.current_trade['CurrentSL']:
                exit_price = self.current_trade['CurrentSL']
            elif current_price <= self.current_trade['CurrentTP']:
                exit_price = self.current_trade['CurrentTP']

        if exit_price:
            self.position.close()
            self.current_trade.update({
                'ExitTime': self.data.index[-1],
                'ExitPrice': exit_price,
                'PnL': (exit_price - self.current_trade['EntryPrice']) * self.mysize if self.current_trade['Type'] == 'Long'
                       else (self.current_trade['EntryPrice'] - exit_price) * self.mysize
            })
            self.trades_info.append(self.current_trade)
            self.current_trade = None

    def on_backtesting_done(self):
        if self.current_trade:
            self.current_trade.update({
                'ExitTime': self.data.index[-1],
                'ExitPrice': self.data.Close[-1],
                'PnL': self.position.pl if self.position else None
            })
            self.trades_info.append(self.current_trade)


# Function to run backtest
def run_backtest(df, strategy_class, cash=200000, commission=0.002, margin=1/75):
    bt = Backtest(df, strategy_class, cash=cash, commission=commission, margin=margin)
    stats = bt.run()
    return stats, pd.DataFrame(stats._strategy.trades_info)

# %% [markdown]
# ## 6. Plotting the strategy

# %%
def plot_chart(df, trades):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.01)

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='BTC/USDT'))

    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_fast'], name='EMA Fast', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_slow'], name='EMA Slow', line=dict(color='orange', width=1)))

    # Nadaraya-Watson envelopes
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Envelope'], name='Upper Envelope', line=dict(color='rgba(255,0,0,0.3)', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Envelope'], name='Lower Envelope', line=dict(color='rgba(0,255,0,0.3)', width=1, dash='dot')))

    # Add trade information
    for _, trade in trades.iterrows():
        # Entry point
        fig.add_trace(go.Scatter(x=[trade['EntryTime']], y=[trade['EntryPrice']],
                                 mode='markers',
                                 marker=dict(symbol='triangle-up' if trade['Type'] == 'Long' else 'triangle-down',
                                             size=12,
                                             color='green' if trade['Type'] == 'Long' else 'red',
                                             line=dict(width=2, color='black')),
                                 name='Entry'))

        # Initial TP and SL lines
        fig.add_shape(type="line", x0=trade['EntryTime'], y0=trade['InitialTP'], x1=trade['ExitTime'], y1=trade['InitialTP'],
                      line=dict(color="rgba(0,255,0,0.5)", width=1, dash="dash"))
        fig.add_shape(type="line", x0=trade['EntryTime'], y0=trade['InitialSL'], x1=trade['ExitTime'], y1=trade['InitialSL'],
                      line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"))

        # Trailing TP and SL lines
        fig.add_trace(go.Scatter(x=[trade['EntryTime'], trade['ExitTime']], y=[trade['InitialTP'], trade['CurrentTP']],
                                 mode='lines', line=dict(color="rgba(0,255,0,0.5)", width=1, dash="dot"), name='Trailing TP'))
        fig.add_trace(go.Scatter(x=[trade['EntryTime'], trade['ExitTime']], y=[trade['InitialSL'], trade['CurrentSL']],
                                 mode='lines', line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dot"), name='Trailing SL'))

        # Exit point
        if pd.notnull(trade['ExitTime']):
            # Determine exit color based on profitability
            exit_color = 'green' if trade['PnL'] > 0 else 'red'

            fig.add_trace(go.Scatter(x=[trade['ExitTime']], y=[trade['ExitPrice']],
                                     mode='markers',
                                     marker=dict(symbol='circle', size=12, color=exit_color,
                                                 line=dict(width=2, color='black')),
                                     name='Exit'))

            # Annotate P/L
            if pd.notnull(trade['PnL']):
                fig.add_annotation(x=trade['ExitTime'], y=trade['ExitPrice'],
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
                                   opacity=0.8)

    # Update layout
    fig.update_layout(
        title='BTC/USDT Chart with Signals and Trade Information',
        xaxis_title='Date',
        yaxis_title='Price (USDT)',
        xaxis_rangeslider_visible=False,
        legend_title='Legend',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800
    )

    # Show the plot
    fig.show()

# %% [markdown]
# ## 7. Main execution

# %%

# Fetch more historical data (in days)
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

df = fetch_historical_data(symbol='BTCUSDT', interval='15m', start_date=start_date, end_date=end_date)
print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")

df = calculate_indicators(df)
df = calculate_nadaraya_watson(df, bandwidth=7)
df = ema_signal(df, backcandles=7)
df = total_signal(df)

# Run backtest
bt = Backtest(df, RefinedNWStrategy, cash=200000, commission=.002, margin=1/75)
stats = bt.run()
trades = pd.DataFrame(stats._strategy.trades_info)

# Print results
print(stats)
print("\nTrade Statistics:")
print(f"Total Trades: {len(trades)}")
print(f"Winning Trades: {len(trades[trades['PnL'] > 0])}")
print(f"Losing Trades: {len(trades[trades['PnL'] <= 0])}")
print(f"Win Rate: {len(trades[trades['PnL'] > 0]) / len(trades):.2%}")
print(f"Average Profit: {trades['PnL'].mean():.2f}")
print(f"Average Profit (Winners): {trades[trades['PnL'] > 0]['PnL'].mean():.2f}")
print(f"Average Loss (Losers): {trades[trades['PnL'] <= 0]['PnL'].mean():.2f}")
print(f"Profit Factor: {abs(trades[trades['PnL'] > 0]['PnL'].sum() / trades[trades['PnL'] <= 0]['PnL'].sum()):.2f}")

# Visualize results
plot_chart(df, trades)


