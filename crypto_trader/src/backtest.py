# src/backtest.py
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

class Backtester:
    def __init__(self, model, env, save_path="results"):
        self.model = model
        self.env = env
        self.results = []
        self.trades = []
        self.equity_curve = []
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
    def run_backtest(self, episodes=100):
        print("\n" + "="*50)
        print("RUNNING BACKTEST")
        print("="*50)
        
        total_trades = 0
        total_profit = 0
        
        try:
            for episode in range(episodes):
                state, _ = self.env.reset()
                done = False
                episode_reward = 0
                episode_trades = 0
                
                while not done:
                    action, _ = self.model.predict(state, deterministic=True)  # Deterministic for testing
                    next_state, reward, done, _, info = self.env.step(action)
                    episode_reward += reward
                    self.equity_curve.append(self.env.balance)
                    
                    if info.get('trade'):
                        trade_info = info['trade']
                        trade_info.update({
                            'episode': episode,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'balance': self.env.balance,
                            'reward': reward
                        })
                        self.trades.append(trade_info)
                        episode_trades += 1
                        if trade_info.get('profit'):
                            total_profit += trade_info['profit']
                    
                    state = next_state
                
                total_trades += episode_trades
                print(f"Episode {episode+1}/{episodes} - Trades: {episode_trades} - Reward: {episode_reward:.2f}")
            
            print("\nBacktest completed successfully!")
            print(f"Total trades executed: {total_trades}")
            print(f"Total profit: ${total_profit:.2f}")
            
            return self.analyze_results()
            
        except Exception as e:
            print(f"\n✗ Backtest failed: {e}")
            return None

    def analyze_results(self):
        if not self.trades:
            return {"error": "No trades executed during backtest"}
        
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df)
        profitable_trades = len(df[df['profit'] > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Advanced metrics
        results = {
            'total_trades': total_trades,
            'total_profit': df['profit'].sum(),
            'win_rate': win_rate * 100,
            'avg_profit': df['profit'].mean(),
            'max_profit': df['profit'].max(),
            'max_loss': df['profit'].min(),
            'profit_factor': abs(df[df['profit'] > 0]['profit'].sum() / df[df['profit'] < 0]['profit'].sum()) if len(df[df['profit'] < 0]) > 0 else float('inf'),
            'avg_win': df[df['profit'] > 0]['profit'].mean() if profitable_trades > 0 else 0,
            'avg_loss': df[df['profit'] < 0]['profit'].mean() if total_trades - profitable_trades > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
            'max_drawdown_pct': self._calculate_max_drawdown_percentage(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
        }
        
        # Generate and save plots
        self._plot_equity_curve()
        self._plot_trade_distribution(df)
        
        # Print detailed analysis
        self._print_analysis(results)
        
        return results

    def _calculate_max_drawdown(self):
        peak = self.equity_curve[0]
        max_dd = 0
        for value in self.equity_curve:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _calculate_max_drawdown_percentage(self):
        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        return float(np.max(drawdown) * 100)

    def _calculate_sharpe_ratio(self, risk_free_rate=0.01):
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) < 2:
            return 0
        return (np.mean(returns) - risk_free_rate) / np.std(returns) if np.std(returns) != 0 else 0

    def _plot_equity_curve(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, label='Portfolio Value')
        plt.title('Equity Curve')
        plt.xlabel('Trading Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'equity_curve.png'))
        plt.close()

    def _plot_trade_distribution(self, df):
        plt.figure(figsize=(12, 6))
        plt.hist(df['profit'], bins=50, edgecolor='black')
        plt.title('Trade Profit Distribution')
        plt.xlabel('Profit ($)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, 'trade_distribution.png'))
        plt.close()

    def _print_analysis(self, results):
        print("\n" + "="*50)
        print("BACKTEST ANALYSIS")
        print("="*50)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total Profit: ${results['total_profit']:.2f}")
        print(f"Average Profit per Trade: ${results['avg_profit']:.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Maximum Drawdown: ${results['max_drawdown']:.2f} ({results['max_drawdown_pct']:.2f}%)")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Average Win: ${results['avg_win']:.2f}")
        print(f"Average Loss: ${results['avg_loss']:.2f}")
        print("\nPlots saved in results directory")
        print("="*50)
