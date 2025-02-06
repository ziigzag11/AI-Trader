# test_model.py
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from environment import TradingEnvironment
from data_processor import DataProcessor
from config import CONFIG
import matplotlib.pyplot as plt
import os

def evaluate_model():
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load test data
    print("Loading test data...")
    try:
        data_path = os.path.join(project_root, 'data', 'training_data.csv')
        test_data = pd.read_csv(data_path)
        print(f"Loading data from: {data_path}")
    except FileNotFoundError:
        print(f"Test data file not found at: {data_path}")
        return
        
    processor = DataProcessor()
    df = processor.process_data(test_data)
    
    # Create test environment
    env = TradingEnvironment(df, CONFIG)
    
    # Look for model in temporary directory
    temp_model_dir = os.path.join(os.path.expanduser("~"), ".trading_models")
    model_path = os.path.join(temp_model_dir, "trading_agent")
    
    try:
        if os.path.exists(model_path + ".zip"):
            model = PPO.load(model_path)
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Model file not found at {model_path}.zip")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Initialize tracking metrics
    trades = []
    balance_history = [CONFIG['initial_capital']]
    positions = []
    
    # Run evaluation
    obs, _ = env.reset()
    done = False
    
    try:
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)  # Fixed step return values
            
            # Track performance
            balance_history.append(env.balance)
            positions.append(env.position)
            if info.get('trade'):
                trades.append(info['trade'])
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    # Calculate performance metrics
    total_trades = len(trades)
    if total_trades == 0:
        print("No trades were executed during testing")
        return
        
    profitable_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    final_return = ((env.balance - CONFIG['initial_capital']) / CONFIG['initial_capital']) * 100
    
    # Print results
    print("\nPerformance Summary:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Final Return: {final_return:.2f}%")
    print(f"Final Balance: ${env.balance:.2f}")
    
    # Plot results
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(balance_history, label='Portfolio Value')
        plt.title('Trading Performance')
        plt.xlabel('Time Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'performance.png'))
        plt.close()
        print("Performance plot saved to results/performance.png")
    except Exception as e:
        print(f"Error creating performance plot: {e}")

if __name__ == "__main__":
    evaluate_model()
