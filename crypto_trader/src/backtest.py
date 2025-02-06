# src/backtest.py
class Backtester:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.results = []
    
    def run_backtest(self, episodes=100):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(state)
                next_state, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                
                if info.get('trade_executed', False):
                    self.results.append(info['trade_info'])
                
                state = next_state
            
        return self.analyze_results()
    
    def analyze_results(self):
        df = pd.DataFrame(self.results)
        return {
            'total_profit': df['profit'].sum(),
            'win_rate': len(df[df['profit'] > 0]) / len(df),
            'avg_r_factor': df['r_factor'].mean(),
            'max_drawdown': self.calculate_max_drawdown(df)
        }
