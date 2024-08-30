import os
import glob
import pandas as pd

DATA_ROOT = "/Users/bread/Documents/options_backtest/intraday/onemin"

def load_ticker(ticker: str, start_date: str = None, end_date: str = None):
    '''
    Load min ticker
    '''
    files = list(sorted(glob.glob(f"{DATA_ROOT}/{ticker}/*")))
    mlist = []
    for f in files:
        df = pd.read_csv(f)
        mlist.append(df)
        
    df = pd.concat(mlist)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index("Datetime", inplace=True)
    df.columns = df.columns.str.lower()
    return df