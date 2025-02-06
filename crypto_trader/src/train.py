from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
from environment import TradingEnvironment
from data_processor import DataProcessor
from config import CONFIG
import os

def main():
    try:
        # Get the absolute path to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Use a temporary directory in user's home for model saving
        temp_model_dir = os.path.join(os.path.expanduser("~"), ".trading_models")
        os.makedirs(temp_model_dir, exist_ok=True)
        model_path = os.path.join(temp_model_dir, "trading_agent")
        
        print(f"\n{'='*50}")
        print("TRADING MODEL TRAINING")
        print(f"{'='*50}")
        print(f"Model path: {model_path}")

        # Load training data
        print("\n[1/4] Loading training data...")
        try:
            data_path = os.path.join(project_root, 'data', 'test_data.csv')
            if not os.path.exists(data_path):
                data_path = os.path.join(project_root, 'data', 'training_data.csv')
            if not os.path.exists(data_path):
                data_path = os.path.join(project_root, 'data', 'data.csv')
            data = pd.read_csv(data_path)
            print(f"✓ Data loaded successfully from: {data_path}")
            print(f"✓ Data shape: {data.shape}")
        except FileNotFoundError:
            print(f"✗ Training data file not found at: {data_path}")
            return
        
        print("\n[2/4] Processing data...")
        processor = DataProcessor()
        df = processor.process_data(data)
        print("✓ Data processing complete")
        
        # Create environment first and verify observation shape
        print("\n[3/4] Setting up environment...")
        env = TradingEnvironment(df, CONFIG)
        obs, _ = env.reset()
        print(f"✓ Initial observation shape: {obs.shape}")
        
        # Verify no NaN values in initial observation and data
        if np.any(np.isnan(obs)):
            raise ValueError("Initial observation contains NaN values")
        if df.isnull().values.any():
            raise ValueError("Training data contains NaN values")
        
        # Wrap environment after verification
        env = DummyVecEnv([lambda: env])
        print("✓ Environment setup complete")
        
        # Try to load existing model, create new one if not found
        print("\n[4/4] Loading/Creating model...")
        try:
            print("Attempting to load existing model...")
            if os.path.exists(model_path + ".zip"):
                model = PPO.load(model_path, env=env)
                print("✓ Existing model loaded successfully")
                print(f"✓ Model file size: {os.path.getsize(model_path + '.zip')/1024:.2f} KB")
                print("✓ Continuing training from previous state")
            else:
                raise FileNotFoundError("Model file not found")
        except (FileNotFoundError, ValueError) as e:
            print(f"✗ {str(e)}")
            print("Creating new model...")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=0.00001,
                n_steps=256,
                batch_size=16,
                n_epochs=3,
                gamma=0.95,
                ent_coef=0.001,
                max_grad_norm=0.1,
                verbose=1,
                device='cpu',
                policy_kwargs={
                    "net_arch": [64, 32],
                    "log_std_init": -2.0
                }
            )
            print("✓ New model created successfully")
        
        print("\nStarting training...")
        print(f"{'='*50}")
        try:
            # Train with more frequent updates
            model.learn(
                total_timesteps=500000,
                progress_bar=True,
                log_interval=100000,
                reset_num_timesteps=False  # Continue timestep counting from previous training
            )
            model.save(model_path)
            print(f"\n✓ Model saved successfully to {model_path}")
            print(f"✓ Final model size: {os.path.getsize(model_path + '.zip')/1024:.2f} KB")
        except Exception as e:
            print(f"\n✗ Error during training/saving: {e}")
            raise
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return

if __name__ == "__main__":
    main()
