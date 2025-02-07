CONFIG = {
    # Trading Parameters
    "initial_capital": 1000.0,
    "leverage": 3,  # Reduced leverage for stability
    "risk_per_trade": 0.02,
    "min_r_ratio": 2.0,
    "max_position_size": 0.2,  # Maximum 20% of capital per trade
    
    # Training Parameters
    "learning_rate": 0.0001,
    "n_steps": 128,  # Reduced to handle smaller datasets
    "batch_size": 32,  # Reduced batch size
    "n_epochs": 10,
    "gamma": 0.99,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "max_grad_norm": 0.5,
    "verbose": 1,
    "seed": 42,
    
    # Network Architecture - Fixed format
    "policy_kwargs": dict(
        net_arch=dict(
            pi=[128, 64, 32],  # Policy network
            vf=[128, 64, 32]   # Value network
        ),
        activation_fn="tanh"
    ),
    
    # Evaluation Parameters
    "eval_episodes": 10,
    "eval_frequency": 50000,
    
    # Paths
    "data_path": "data/training_data.csv",
    "model_save_path": "models/trading_agent",
    "log_dir": "logs/"
}
