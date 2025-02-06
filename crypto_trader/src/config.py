CONFIG = {
    # Trading Parameters
    "initial_capital": 100.0,
    "leverage": 5,
    "risk_per_trade": 0.02,
    "min_r_ratio": 2.0,
    "min_win_rate": 0.35,
    "trading_pair": "BTC/USDT",
    
    # Training Parameters
    "learning_rate": 0.0003,
    "n_steps": 1024,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "clip_range": 0.2,
    "ent_coef": 0.1,
    "verbose": 1,
    "seed": 42,
    
    "eval_interval": 10000,
    "eval_episodes": 10,
    "eval_every_n_steps": 10000,
    "eval_every_n_episodes": 10,
    "eval_every_n_timesteps": 10000,
    
    # Paths
    "data_path": "crypto_trader/data/training_data.csv",
    "model_save_path": "crypto_trader/models/trading_agent",
    "log_dir": "crypto_trader/logs/"
}
