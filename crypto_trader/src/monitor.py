import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

class TrainingMonitor:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.trades = []
        self.rewards = []
        self.portfolio_values = []
        
    def log_trade(self, trade_info):
        self.trades.append({
            'timestamp': datetime.now(),
            **trade_info
        })
        
    def update_metrics(self, reward, portfolio_value):
        self.rewards.append(reward)
        self.portfolio_values.append(portfolio_value)
        
        if len(self.rewards) % 10 == 0:  # Plot every 10 episodes
            self.plot_metrics()
            
    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(131)
        plt.plot(self.rewards)
        plt.title('Training Rewards')
        
        # Plot portfolio value
        plt.subplot(132)
        plt.plot(self.portfolio_values)
        plt.title('Portfolio Value')
        
        # Plot R-factor distribution if trades exist
        if self.trades:
            plt.subplot(133)
            r_factors = [t['r_factor'] for t in self.trades if 'r_factor' in t]
            plt.hist(r_factors, bins=20)
            plt.title('R-Factor Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/metrics.png')
        plt.close()

if __name__ == "__main__":
    # Test the monitor
    monitor = TrainingMonitor()
    
    # Simulate some data
    for i in range(100):
        monitor.log_trade({
            'r_factor': np.random.normal(2, 0.5),
            'profit': np.random.normal(0, 1)
        })
        monitor.update_metrics(
            reward=np.random.normal(0, 1),
            portfolio_value=1000 * (1 + np.random.normal(0, 0.01))
        )
