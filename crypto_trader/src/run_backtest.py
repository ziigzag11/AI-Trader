from stable_baselines3 import PPO
import os
from environment import TradingEnvironment
from data_processor import DataProcessor
from config import CONFIG
from backtest import Backtester
import pandas as pd

def main():
    try:
        # Get project root and paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Load the trained model
        model_path = os.path.join(os.path.expanduser("~"), ".trading_models", "trading_agent")
        if not os.path.exists(model_path + ".zip"):
            print(f"No trained model found at {model_path}.zip")
            return

        # Load and process data
        print("\nLoading test data...")
        try:
            data_path = os.path.join(project_root, 'data', 'training_data.csv')
            data = pd.read_csv(data_path)
            print(f"Data loaded successfully from: {data_path}")
        except FileNotFoundError:
            print(f"Test data not found at: {data_path}")
            return

        # Process data
        processor = DataProcessor()
        df = processor.process_data(data)

        # Create environment
        env = TradingEnvironment(df, CONFIG)
        
        # Load model
        try:
            model = PPO.load(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Create and run backtester
        backtester = Backtester(model, env, save_path=results_dir)
        results = backtester.run_backtest(episodes=1)  # Start with 1 episode for testing

        if results and not results.get('error'):
            print("\nBacktest completed successfully!")
        else:
            print("\nBacktest failed or no trades were executed")

    except Exception as e:
        print(f"Error during backtest: {e}")

if __name__ == "__main__":
    main() 