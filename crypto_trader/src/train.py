from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
import torch
import traceback
from environment import TradingEnvironment
from data_processor import DataProcessor
from config import CONFIG
import os
import gymnasium as gym  # Changed from 'gym' to 'gymnasium'
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

def test_environment(env):
    """Helper function to test the environment"""
    try:
        # Test environment initialization
        if isinstance(env, DummyVecEnv):
            obs = env.reset()
            info = {}  # DummyVecEnv doesn't return info
        else:
            obs, info = env.reset()
            
        print("Initial observation shape:", obs.shape)
        print("Initial observation sample:", obs[0])

        # Test environment step
        for i in range(5):
            action = env.action_space.sample()  # No need to wrap in array
            if isinstance(env, DummyVecEnv):
                # DummyVecEnv might return only 4 values (old gym interface)
                step_result = env.step([action])
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                else:
                    obs, reward, terminated, truncated, info = step_result
                
                # DummyVecEnv returns arrays, get first element
                obs = obs[0]
                reward = reward[0]
                terminated = terminated[0] if isinstance(terminated, (list, np.ndarray)) else terminated
                truncated = truncated[0] if isinstance(truncated, (list, np.ndarray)) else truncated
                info = info[0] if info else {}
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                
            print(f"Step {i+1}:")
            print(f"  Action taken: {action}")
            print(f"  Reward received: {reward}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            print(f"  Info: {info}")
            
            if terminated or truncated:
                if isinstance(env, DummyVecEnv):
                    obs = env.reset()
                else:
                    obs, info = env.reset()
                print(f"Episode completed after {i+1} steps")
                
        print("Environment test completed successfully")
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        print("Full error traceback:")
        traceback.print_exc()
        return False

class CustomTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_started = False
    
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.training_started = True
    
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        """
        if not self.training_started:
            self.model.rollout_buffer.reset()
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each step
        """
        return True

class CustomRolloutBuffer(RolloutBuffer):
    def get(self, batch_size: int):
        # Override to avoid the full assertion
        indices = np.random.permutation(self.buffer_size)
        
        # Return everything
        for start_idx in range(0, self.buffer_size, batch_size):
            yield self._get_samples(indices[start_idx:start_idx + batch_size])

class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the default buffer with our custom one
        self.rollout_buffer = CustomRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
    
    def learn(self, *args, **kwargs):
        # Initialize buffer before training
        self._last_obs = self.env.reset()
        self.rollout_buffer.reset()
        
        # Fill buffer with initial rollouts
        for _ in range(self.n_steps):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            
            new_obs, rewards, dones, infos = self.env.step(actions.cpu().numpy())
            
            self.rollout_buffer.add(
                self._last_obs,
                actions.cpu().numpy(),
                rewards,
                self._last_episode_starts,
                values.cpu().numpy(),
                log_probs.cpu().numpy()
            )
            
            self._last_obs = new_obs
            self._last_episode_starts = dones
        
        self.rollout_buffer.compute_returns_and_advantage(
            last_values=values.cpu().numpy(),
            dones=dones
        )
        
        return super().learn(*args, **kwargs)

def main():
    try:
        # Setup paths
        model_dir = os.path.join(os.path.expanduser("~"), ".trading_models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "trading_agent")
        
        print("\n" + "="*50)
        print("CRYPTO TRADING AGENT TRAINING")
        print("="*50)
        print("Model path:", model_path)
        
        # Load data
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, 'data', 'training_data.csv')
        
        try:
            data = pd.read_csv(data_path)
            print("\nData loaded successfully")
            print("Data shape:", data.shape)
            print("Data columns:", data.columns.tolist())
            print("Sample data:\n", data.head())
            
            # Verify data length
            min_required = max(256, CONFIG['n_steps'])
            if len(data) < min_required:
                print(f"Error: Not enough data points. Need at least {min_required}, got {len(data)}")
                return
                
        except FileNotFoundError:
            print("Training data not found")
            return

        # Process data
        try:
            processor = DataProcessor()
            df = processor.process_data(data)
            print("\nData processing complete")
            print("Processed data shape:", df.shape)
            print("Processed columns:", df.columns.tolist())
            print("Any NaN values:", df.isnull().any().any())
        except Exception as e:
            print(f"Error during data processing: {e}")
            return

        # Create environment
        try:
            env = TradingEnvironment(df, CONFIG)
            # Test raw environment first
            print("\nTesting raw environment...")
            if not test_environment(env):
                return
            
            # Then wrap and test vectorized environment
            env = DummyVecEnv([lambda: env])
            print("\nTesting vectorized environment...")
            if not test_environment(env):
                return
            
            print("\nEnvironment tests passed successfully")

        except Exception as e:
            print(f"Error creating environment: {e}")
            return
        
        # Load or create model
        model = None
        if os.path.exists(model_path + ".zip"):
            print("\nLoading existing model for continued training...")
            try:
                model = PPO.load(model_path, env=env, device='cpu')
                model.learning_rate = float(CONFIG['learning_rate'])
                model.n_steps = int(CONFIG['n_steps'])
                model.batch_size = int(CONFIG['batch_size'])
                print("Existing model loaded successfully")
                print("Model size: {:.2f} KB".format(os.path.getsize(model_path + '.zip')/1024))
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating new model instead...")
                model = None
        
        if model is None:
            print("\nCreating new model with optimized parameters...")
            try:
                # Make n_steps even smaller to ensure we can fill the buffer
                n_steps = min(128, len(df) // 4)  # Smaller n_steps
                batch_size = min(32, n_steps // 4)  # Ensure batch_size is smaller than n_steps
                
                model = CustomPPO(
                    "MlpPolicy",
                    env,
                    learning_rate=CONFIG['learning_rate'],
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=CONFIG['n_epochs'],
                    gamma=CONFIG['gamma'],
                    ent_coef=CONFIG['ent_coef'],
                    max_grad_norm=CONFIG['max_grad_norm'],
                    verbose=1,
                    device='cpu',
                    policy_kwargs={
                        'net_arch': dict(
                            pi=[64, 64],
                            vf=[64, 64]
                        ),
                        'activation_fn': torch.nn.Tanh,
                        'normalize_images': False
                    }
                )
                print("New model created successfully")
                print(f"Using n_steps: {n_steps}, batch_size: {batch_size}")

                # Test environment with proper return value handling
                print("\nTesting environment...")
                try:
                    # Test environment initialization
                    obs, info = env.reset()
                    print("Initial observation shape:", obs.shape)
                    print("Initial observation sample:", obs[0])

                    # Test environment step
                    for i in range(5):
                        action = env.action_space.sample()  # No need to wrap in array
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Step {i+1}:")
                        print(f"  Action taken: {action}")
                        print(f"  Reward received: {reward}")
                        print(f"  Terminated: {terminated}")
                        print(f"  Truncated: {truncated}")
                        print(f"  Info: {info}")
                        
                        if terminated or truncated:
                            obs, info = env.reset()
                            print(f"Episode completed after {i+1} steps")
                            
                    print("Environment test completed successfully")
                    
                except Exception as e:
                    print(f"Environment test failed: {e}")
                    print("Full error traceback:")
                    traceback.print_exc()
                    return
                
            except Exception as e:
                print(f"Error creating model: {e}")
                return

        # Train model
        print("Training configuration:")
        print(f"Steps per update: {model.n_steps}")
        print(f"Batch size: {model.batch_size}")
        print(f"Learning rate: {model.learning_rate}")
        
        # Start training
        print("\nStarting training...")
        model.learn(
            total_timesteps=500000,
            progress_bar=True,
            log_interval=1000,
            reset_num_timesteps=True
        )
        
        model.save(model_path)
        print("\nTraining completed successfully")
        print("Model saved to:", model_path)
        print("Final model size: {:.2f} KB".format(os.path.getsize(model_path + '.zip')/1024))
        
        print("\nTraining Summary:")
        print("Total timesteps:", model.num_timesteps)
        print("Final learning rate:", model.learning_rate)
        print("Batch size:", model.batch_size)
        print("Network architecture:", model.policy)

    except Exception as e:
        print("\nSetup failed with error:", str(e))
        print("\nFull error traceback:")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
