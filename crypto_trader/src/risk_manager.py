# src/risk_manager.py
class RiskManager:
    def __init__(self, config):
        self.max_risk_per_trade = config['risk_per_trade']
        self.min_r_ratio = config['min_r_ratio']
        self.leverage = config['leverage']
    
    def calculate_position_size(self, capital, entry_price, stop_loss):
        risk_amount = min(capital * 0.02, self.max_risk_per_trade)  # 2% max risk per trade
        position_size = (risk_amount * self.leverage) / abs(entry_price - stop_loss)
        return position_size
    
    def validate_trade(self, entry_price, stop_loss, take_profit):
        r_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        return r_ratio >= self.min_r_ratio
