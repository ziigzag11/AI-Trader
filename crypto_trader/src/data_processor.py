import pandas as pd
import numpy as np
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.base_features = ['open', 'high', 'low', 'close', 'volume']

    def process_data(self, df):
        # Ensure we have enough data for our window sizes
        if len(df) < 256:  # Minimum required length
            raise ValueError(f"Not enough data points. Need at least 256, got {len(df)}")
            
        # Make copy to avoid modifying original data
        df = df.copy()
        
        # Sort by date if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Reset index after all processing
        df = df.reset_index(drop=True)
        
        return df

    def add_technical_indicators(self, df):
        # Add basic technical indicators
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        
        return df

