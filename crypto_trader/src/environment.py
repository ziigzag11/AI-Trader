import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnvironment(gym.Env):
    def __init__(self, df, config):
        super().__init__()
        # Verify data length
        if len(df) < config['n_steps']:
            raise ValueError(f"Not enough data points. Need at least {config['n_steps']}, got {len(df)}")
            
        # Preprocess the dataframe
        self.df = df.copy()
        
        # Forward fill NaN values
        self.df.fillna(method='ffill', inplace=True)
        # Backward fill any remaining NaN values
        self.df.fillna(method='bfill', inplace=True)
        
        # Replace any remaining NaN values with reasonable defaults
        self.df['volume'] = self.df['volume'].replace({0: 1.0, np.nan: 1.0})
        
        # Reset index to ensure continuous indexing
        self.df = self.df.reset_index(drop=True)
        
        self.config = config
        self.current_step = 0
        self.balance = config.get('initial_capital', 100.0)
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profits = []
        self.risk_per_trade = 10.0
        self.min_r_ratio = 2.0
        self.trade_history = []
        
        # Calculate price normalizer with NaN handling
        valid_prices = self.df['close'].replace([np.inf, -np.inf], np.nan).dropna()
        self.price_normalizer = valid_prices.mean() if len(valid_prices) > 0 else 1.0
        
        # Define observation space for 5 features to match existing model
        self.observation_space = spaces.Box(
            low=-10, 
            high=10,
            shape=(5,),  # Keep original 5 features for now
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
    
    def step(self, action):
        # Convert action to scalar if it's an array
        if isinstance(action, (np.ndarray, list)):
            if isinstance(action, np.ndarray) and action.ndim == 0:
                action = action.item()
            elif len(action) > 0:
                action = action[0]
            if isinstance(action, np.ndarray):
                action = action.item()
                
        current_price = float(self.df.iloc[self.current_step]['close'])
        info = {}  # Initialize as empty dict
        reward = 0
        
        try:
            if action == 1 and self.position == 0:  # Buy
                stop_loss = self._calculate_stop_loss(current_price, action)
                # Add safety check for stop loss calculation
                if stop_loss >= current_price:  # Invalid stop loss
                    reward = -0.1
                    return self._get_observation(), reward, True, False, info
                    
                # Add safety check for r_potential calculation
                try:
                    r_potential = (current_price - stop_loss) / stop_loss if stop_loss > 0 else 0
                except ZeroDivisionError:
                    r_potential = 0
                
                if r_potential >= self.min_r_ratio:
                    position_size = self._calculate_position_size(current_price, stop_loss)
                    if position_size > 0:
                        self.position = position_size
                        self.entry_price = current_price
                        self.stop_loss = stop_loss
                        self.entry_step = self.current_step  # Track entry step
                        # Set multiple take profits
                        self.take_profits = [
                            current_price + (current_price - stop_loss) * 2,  # 2R
                            current_price + (current_price - stop_loss) * 3,  # 3R
                            current_price + (current_price - stop_loss) * 4   # 4R
                        ]
                        info = {
                            'trade_type': 'buy',
                            'price': float(current_price),
                            'size': float(position_size),
                            'stop_loss': float(stop_loss)
                        }
                        
            elif action == 2 and self.position > 0:  # Sell
                profit = (current_price - self.entry_price) * self.position
                # Add safety check for r_factor calculation
                try:
                    stop_loss_diff = (self.entry_price - self.stop_loss)
                    r_factor = (current_price - self.entry_price) / stop_loss_diff if stop_loss_diff != 0 else 0
                except ZeroDivisionError:
                    r_factor = 0
                
                info = {
                    'trade_type': 'sell',
                    'entry': float(self.entry_price),
                    'exit': float(current_price),
                    'profit': float(profit),
                    'r_factor': float(r_factor)
                }
                
                self.balance += profit
                self.position = 0
                self.trade_history.append({
                    'r_factor': r_factor,
                    'profit': profit
                })
                
            # Check stop loss
            elif self.position > 0:
                if current_price <= self.stop_loss:
                    loss = (current_price - self.entry_price) * self.position
                    self.balance += loss
                    self.position = 0
                    info = {
                        'trade_type': 'stop_loss',
                        'profit': float(loss),
                        'r_factor': -1
                    }
                    
            # Calculate reward based on R-factor
            reward = self._calculate_r_based_reward(info)
            
            self.current_step += 1
            terminated = self.current_step >= len(self.df) - 1
            truncated = False  # We don't use truncation in this environment
            
            obs = self._get_observation()
            reward = float(reward)
            
            # Add episode info to info dict
            info.update({
                'step': self.current_step,
                'balance': float(self.balance),
                'position': float(self.position)
            })
            
            # Return all 5 values required by Gymnasium
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            return self._get_observation(), -1.0, True, False, {}
    
    def _get_observation(self):
        try:
            # Get current state
            current = self.df.iloc[self.current_step]
            
            # Basic 5-feature observation
            obs = np.array([
                float(current['open']) / self.price_normalizer,
                float(current['high']) / self.price_normalizer,
                float(current['low']) / self.price_normalizer,
                float(current['close']) / self.price_normalizer,
                self.position / (self.balance if self.balance > 0 else 1.0)
            ], dtype=np.float32)
            
            # Safety check for NaN values
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return obs
            
        except Exception as e:
            print(f"Error in _get_observation: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _calculate_rsi(self, prices, period=14):
        try:
            # Handle NaN values in prices
            prices = prices.fillna(method='ffill').fillna(method='bfill')
            
            deltas = np.diff(prices)
            if len(deltas) < period:
                return 50.0  # Return neutral RSI if not enough data
                
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum()/period
            down = -seed[seed < 0].sum()/period
            rs = up/down if down != 0 else 1.0
            rsi = np.zeros_like(prices)
            rsi[:period] = 100. - 100./(1. + rs)

            for i in range(period, len(prices)):
                delta = deltas[i - 1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta

                up = (up*(period-1) + upval)/period
                down = (down*(period-1) + downval)/period
                rs = up/down if down != 0 else 1.0
                rsi[i] = 100. - 100./(1. + rs)

            return float(rsi[-1])
            
        except Exception as e:
            print(f"Error in _calculate_rsi: {e}")
            return 50.0  # Return neutral RSI on error

    def _calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on R-factor and fixed risk amount"""
        try:
            if entry_price <= 0 or stop_loss <= 0:
                return 0
                
            price_difference = abs(entry_price - stop_loss)
            if price_difference == 0:
                return 0
                
            # Calculate base units based on fixed risk amount
            base_units = self.risk_per_trade / price_difference
            
            # Calculate potential reward based on R-ratio
            target_price = entry_price + (price_difference * self.min_r_ratio)
            potential_reward = (target_price - entry_price) * base_units
            
            # Apply leverage if needed (up to 5x)
            required_margin = (entry_price * base_units) / self.config['leverage']
            if required_margin > self.balance:
                # Adjust position size based on available balance
                leverage_multiplier = min(5, self.balance / required_margin) if required_margin > 0 else 0
                base_units *= leverage_multiplier
            
            return base_units
            
        except ZeroDivisionError:
            print("Division by zero in position size calculation")
            return 0
        except Exception as e:
            print(f"Error in position size calculation: {e}")
            return 0

    def _calculate_stop_loss(self, entry_price, action):
        """Calculate stop loss based on recent price action"""
        try:
            window = 20
            start_idx = max(0, self.current_step - window)
            price_history = self.df.iloc[start_idx:self.current_step+1]
            
            if action == 1:  # Buy
                # Stop loss below recent low
                stop_loss = price_history['low'].min()
                # Ensure minimum R-ratio potential
                price_difference = entry_price - stop_loss
                target = entry_price + (price_difference * self.min_r_ratio)
                
                # Validate stop loss distance
                if stop_loss <= 0 or (entry_price - stop_loss) / entry_price < 0.005:
                    stop_loss = entry_price * 0.995
                    
            else:  # Sell
                # Stop loss above recent high
                stop_loss = price_history['high'].max()
                price_difference = stop_loss - entry_price
                target = entry_price - (price_difference * self.min_r_ratio)
                
                if stop_loss <= 0 or (stop_loss - entry_price) / entry_price < 0.005:
                    stop_loss = entry_price * 1.005
                    
            return max(0.01, stop_loss)  # Ensure stop loss is never zero or negative
            
        except Exception as e:
            print(f"Error in stop loss calculation: {e}")
            return entry_price * 0.995  # Default to 0.5% below entry

    def _calculate_r_based_reward(self, info):
        """Calculate reward based on R-factor"""
        if not info.get('trade_type'):
            return -0.0001  # Small penalty for holding
            
        trade_type = info['trade_type']
        if trade_type == 'buy':
            return 0.1  # Small reward for valid entry
            
        if trade_type in ['sell', 'stop_loss']:
            r_factor = info.get('r_factor', 0)
            if r_factor >= 2:
                return 1.0
            elif r_factor >= 1:
                return 0.5
            elif r_factor >= 0:
                return 0.1
            else:
                return -0.2
                
        return 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.config.get('initial_capital', 1000.0)
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profits = []
        self.trade_history = []
        
        # Get initial observation
        obs = self._get_observation()
        
        # Create info dict
        info = {
            'initial_balance': self.balance,
            'current_step': self.current_step,
            'position': self.position,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss
        }
        
        # Return observation and info dict (no need to wrap obs in array)
        return obs, info

    def _calculate_trend(self):
        """Calculate short-term trend strength"""
        window = 10
        prices = self.df.iloc[max(0, self.current_step-window):self.current_step+1]['close']
        return np.polyfit(range(len(prices)), prices, 1)[0]

    def _calculate_trade_frequency(self):
        """Calculate recent trading frequency"""
        window = 20
        trades = sum(1 for x in self.trade_history[-window:] if x)
        return trades / window
