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
        df = df.copy()
        
        # Keep only base features
        df = df[self.base_features]
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df)
        df = pd.DataFrame(scaled_data, columns=self.base_features, index=df.index)
        
        return df.fillna(method='ffill')

    def add_technical_indicators(self, df):
        # Add basic technical indicators
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        
        return df

