import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from numba import jit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datetime import datetime

@jit(nopython=True)
def calculate_qp_single_numba(historical_returns, current_return):
    if len(historical_returns) == 0:
        return np.nan
    
    if current_return < 0:
        percentile = np.mean(historical_returns <= current_return) * 100
        left_area = np.mean(historical_returns < 0)
        return percentile / left_area if left_area > 0 else 100
    else:
        percentile = np.mean(historical_returns >= current_return) * 100
        right_area = np.mean(historical_returns > 0)
        return (100 - percentile) / right_area if right_area > 0 else 100

def calculate_qp(group, windows=[3, 5, 10, 20], lookback_years=5):
    close = group['close'].values if isinstance(group, pd.DataFrame) else group
    dates = group['date'].values if isinstance(group, pd.DataFrame) else np.arange(len(close))
    
    results = {}
    for window in windows:
        returns = np.zeros_like(close)
        returns[window:] = (close[window:] - close[:-window]) / close[:-window]
        
        qp_values = np.zeros_like(close)
        for i in range(window, len(close)):
            # Convert lookback_years to days for compatibility with numpy datetime64
            lookback_days = lookback_years * 365
            lookback_index = np.searchsorted(dates, dates[i] - np.timedelta64(lookback_days, 'D'))
            historical_returns = returns[lookback_index:i]
            qp_values[i] = calculate_qp_single_numba(historical_returns, returns[i])
        
        results[f'return_{window}'] = returns
        results[f'qp_{window}'] = qp_values
    
    if isinstance(group, pd.DataFrame):
        for key, value in results.items():
            group[key] = value
        return group
    else:
        return results

def calculate_features(group):
    if not isinstance(group, pd.DataFrame):
        group = pd.DataFrame({
            'close': group,
            'date': pd.date_range(end=pd.Timestamp.now(), periods=len(group)),
            'high': group,  # Assuming close prices for high and low if not provided
            'low': group,
            'volume': np.ones_like(group)  # Dummy volume data
        })
    
    # Calculate QP indicator for different windows
    group = calculate_qp(group, windows=[3, 5, 10, 20])
    
    # Calculate RSI for different windows
    for window in [3, 5, 10, 20]:
        group[f'rsi_{window}'] = talib.RSI(group['close'].values, timeperiod=window)
    
    # Calculate other technical indicators
    group['sma_200'] = talib.SMA(group['close'].values, timeperiod=200)
    group['atr'] = talib.ATR(group['high'].values, group['low'].values, group['close'].values)
    
    # Calculate rates of change
    for window in [5, 10, 20, 60, 120, 250]:
        group[f'roc_{window}'] = group['close'].pct_change(window)
    
    # IBS (Internal Bar Strength)
    group['ibs'] = (group['close'] - group['low']) / (group['high'] - group['low'])
    
    # Normalized ATR
    group['natr'] = group['atr'] / group['close']
    
    # Closing price distance to 200-day SMA
    group['dist_sma_200'] = (group['close'] - group['sma_200']) / group['sma_200']
    
    # Turnover
    group['turnover'] = group['volume'] * group['close']
    
    return group

def prepare_data(df, forward_days=5):
    df = df.copy()
    df.loc[:, 'target'] = (df.groupby('symbol')['close'].shift(-forward_days) > df['close']).astype(int)
    df = df.drop('model_pred', axis=1)
    df = df.dropna()
    
    
    features = ['open', 'high', 'low', 'close', 'volume', 'adjusted',
       'return_3', 'qp_3', 'return_5', 'qp_5', 'return_10', 'qp_10',
       'return_20', 'qp_20', 'rsi_3', 'rsi_5', 'rsi_10', 'rsi_20', 'sma_200',
       'atr', 'roc_5', 'roc_10', 'roc_20', 'roc_60', 'roc_120', 'roc_250',
       'ibs', 'natr', 'dist_sma_200', 'turnover']
    
    X = df[features]
    y = df['target']
    
    # Clip extreme values
    for col in X.columns:
        lower_bound = X[col].quantile(0.01)
        upper_bound = X[col].quantile(0.99)
        X.loc[:, col] = X[col].clip(lower_bound, upper_bound)

    
    return X, y

def train_model(df, to_year, lookback_years=15):
    start_year = to_year - lookback_years
    end_year = to_year - 1
    
    train_data = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
    
    X, y = prepare_data(train_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    
    return model

def train_and_predict(df, symbol, to_year, lookback_years=5):
    start_year = to_year - lookback_years
    end_year = to_year - 1
    
    train_data = df[(df['symbol'] == symbol) & (df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]
    test_data = df[(df['symbol'] == symbol) & (df['date'].dt.year == to_year)]

    if len(train_data) < 2:
        print(f"Not enough data to train model for symbol {symbol} in year {to_year}")
        return None, np.full(len(test_data), np.nan)
    
    X, y = prepare_data(train_data)
    X_test, y_test = prepare_data(test_data)
    

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = XGBClassifier(eval_metric='logloss', n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # 01 classification
    return model, model.predict_proba(X_test)[:, 1]

def process_symbol(df, symbol, unique_years):
    print(f"Processing symbol: {symbol}")
    symbol_data = df[df['symbol'] == symbol]
    latest_year = symbol_data['date'].dt.year.max()
    
    results = []
    for year in range(2022, 2023):
        model, predictions = train_and_predict(df, symbol, year)
        if model is not None:
            mask = (df['symbol'] == symbol) & (df['date'].dt.year == year)
            results.append((mask, predictions))
    
    return symbol, results, model if model is not None else None

def main_modelpred(df):
    df['model_pred'] = np.nan
    unique_years = sorted(df['date'].dt.year.unique())
    unique_symbols = df['symbol'].unique()[:4]

    num_cores = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_symbol, df, symbol, unique_years) for symbol in unique_symbols]
        results = [future.result() for future in futures]

    modelmap = {}
    for symbol, symbol_results, model in results:
        for mask, predictions in symbol_results:
            df.loc[mask, 'model_pred'] = predictions
        modelmap[symbol] = model

    return df, modelmap
    

def main():
    # Load your data here
    df = pd.read_csv("data/russel_3000_prices.csv").drop("Unnamed: 0", axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    
    # Parallel processing for feature calculation
    num_cores = multiprocessing.cpu_count()
    symbol_groups = [group for _, group in df.groupby('symbol')]
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(calculate_features, symbol_groups))
    
    df = pd.concat(results)
    
    # Filter for the strategy conditions
    df['strategy_condition'] = (df['qp_3'] < 15) & (df['close'] > df['sma_200']) & (df['return_3'] < 0)
    
    # Add model predictions for each year
    df['model_pred'] = np.nan
    unique_years = sorted(df['date'].dt.year.unique())
    unique_symbols = df['symbol'].unique()
    
    modelmap = {}
    
    for symbol in unique_symbols:
        print(f"Processing symbol: {symbol}")
        symbol_data = df[df['symbol'] == symbol]
        latest_year = symbol_data['date'].dt.year.max()
        
        for year in range(unique_years[4], latest_year + 1):
            model, predictions = train_and_predict(df, symbol, year)
            mask = (df['symbol'] == symbol) & (df['date'].dt.year == year)
            df.loc[mask, 'model_pred'] = predictions
            
            if year == latest_year:
                modelmap[symbol] = model
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the processed DataFrame to HDF5 with timestamp in filename
    filename = f'data/processed_data_{timestamp}.h5'
    df.to_hdf(filename, key='df', mode='w', format='table')
    
    return df, filename

if __name__ == '__main__':
    # df, saved_filename = main()
    # print("Processing completed. DataFrame shape:", df.shape)
    # print(f"Processed data saved to '{saved_filename}'")
    
    # Read the saved HDF5 file
    saved_filename = "data/processed_data_20240830_154643.h5"
    read_df = pd.read_hdf(saved_filename, key='df')
    # model = train_model(read_df, 2022)
    
    df, modelmap = main_modelpred(read_df)
    
    print("\nReading the saved file:")
    print("Read DataFrame shape:", read_df.shape)
    print("First few rows of the read DataFrame:")
    print(df.head())
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'data/processed_data_with_each_symbol_model_preds_{timestamp}.h5'
    df.to_hdf(filename, key='df', mode='w', format='table')
    print(f"Processed data saved to '{filename}'")
