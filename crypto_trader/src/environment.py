import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnvironment(gym.Env):
    def __init__(self, df, config):
        super().__init__()
        self.df = df
        self.config = config
        self.current_step = 0
        self.balance = config.get('initial_capital', 1000.0)
        self.position = 0
        self.entry_price = 0
        
        # Normalize prices using the mean price, with NaN handling
        self.price_normalizer = np.nanmean(self.df['close'].values)
        if np.isnan(self.price_normalizer):
            self.price_normalizer = 1.0
        
        # Define observation space for 4 price features + position
        self.observation_space = spaces.Box(
            low=-10, 
            high=10, 
            shape=(5,),  # 5 features: open, high, low, close, position
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
    
    def step(self, action):
        # Get current price with NaN check
        current_price = self.df.iloc[self.current_step]['close']
        if np.isnan(current_price):
            current_price = self.df.iloc[self.current_step-1]['close'] if self.current_step > 0 else 1.0
        current_price = float(current_price)
        
        reward = -0.0001  # Small holding penalty
        info = {}
        
        try:
            if action == 1 and self.position == 0:  # Buy
                position_size = self._calculate_position_size(current_price)
                self.position = position_size
                self.entry_price = current_price
                reward = 0.001
                info['trade'] = {'type': 'buy', 'price': current_price, 'size': position_size}
                
            elif action == 2 and self.position > 0:  # Sell
                profit = (current_price - self.entry_price) * self.position
                profit_pct = profit / (self.entry_price * self.position) if self.entry_price > 0 else 0
                reward = np.clip(profit_pct, -1, 1)  # Clip rewards
                self.balance = max(0, self.balance + profit)
                
                # Calculate R-factor for the trade
                r_factor = profit_pct / self.config.get('risk_per_trade', 0.02)
                info['trade'] = {
                    'type': 'sell',
                    'entry': self.entry_price,
                    'exit': current_price,
                    'profit': profit,
                    'r_factor': r_factor
                }
                
                self.position = 0
            
            # Safety clipping
            self.balance = np.clip(self.balance, 0, 1e9)
            reward = np.clip(reward, -1, 1)
            
        except Exception as e:
            print(f"Error in step: {e}")
            reward = -1
            self.position = 0
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        obs = self._get_observation()
        # Handle NaN values in observation
        if np.any(np.isnan(obs)):
            obs = np.nan_to_num(obs, nan=0.0)
            
        return obs, float(reward), done, False, info
    
    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        
        # Get previous values for NaN filling
        prev_row = self.df.iloc[self.current_step-1] if self.current_step > 0 else row
        
        # Handle potential NaN values in features by using previous values
        open_price = row['open'] if not np.isnan(row['open']) else prev_row['open']
        high_price = row['high'] if not np.isnan(row['high']) else prev_row['high']
        low_price = row['low'] if not np.isnan(row['low']) else prev_row['low']
        close_price = row['close'] if not np.isnan(row['close']) else prev_row['close']
        
        obs = np.array([
            float(open_price) / self.price_normalizer,
            float(high_price) / self.price_normalizer,
            float(low_price) / self.price_normalizer,
            float(close_price) / self.price_normalizer,
            float(self.position) / (self.balance if self.balance > 0 else 1.0)
        ], dtype=np.float32)
        
        # Replace any remaining NaN values with 0
        obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def _calculate_position_size(self, price):
        if price <= 0:
            return 0
            
        risk_amount = self.config.get('risk_per_trade', 0.02) * self.balance
        max_size = self.config.get('max_position_size', 1.0) * self.balance / price
        leverage = self.config.get('leverage', 1.0)
        
        position_size = (risk_amount * leverage) / price
        if position_size > max_size:
            position_size = max_size
        return position_size
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.config.get('initial_capital', 1000.0)
        self.position = 0
        self.entry_price = 0
        return self._get_observation(), {}
