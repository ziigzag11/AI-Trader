import ccxt
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_historical_data():
    exchange = ccxt.kraken({
        'enableRateLimit': True,
    })

    end = datetime.now()
    start = end - timedelta(days=30)
    
    print("Fetching BTC/USDT data...")
    
    ohlcv = exchange.fetch_ohlcv(
        'BTC/USDT',
        '1m',
        int(start.timestamp() * 1000),
        limit=1000
    )

    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    output_path = 'crypto_trader/data/training_data.csv'
    df.to_csv(output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    fetch_historical_data()
