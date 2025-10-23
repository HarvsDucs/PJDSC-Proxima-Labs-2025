# weather_env_next_day.py
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from joblib import load
import os

class WeatherEnvNextDay(gym.Env):
    """
    Custom Environment for Next-Day Weather Model Selection.

    This environment is designed for a predictive task. The agent observes
    the state on Day T and chooses a model. The reward is calculated based
    on that model's performance on Day T+1.
    """
    metadata = {'render_modes': []}

    def __init__(self, aligned_data_path, scaler_path):
        """
        Initializes the environment.

        Args:
            aligned_data_path (str): Path to the pre-aligned CSV data.
            scaler_path (str): Path to the saved, pre-fitted joblib scaler file.
        """
        super().__init__()
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at '{scaler_path}'. Please run the data alignment script first.")
        self.scaler = load(scaler_path)
        
        self._load_and_prepare_data(aligned_data_path)

        self.action_space = spaces.Discrete(self.action_space_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_space_size,), dtype=np.float32
        )
        self.current_step = 0

    def _load_and_prepare_data(self, aligned_data_path):
        """Loads the pre-aligned data and prepares the state/action spaces."""
        self.data = pd.read_csv(aligned_data_path, index_col='date', parse_dates=True)
        
        # --- CORRECTED ACTION DISCOVERY LOGIC ---
        
        # 1. Find ALL columns that are targets (i.e., from Day T+1)
        all_target_cols = [c for c in self.data.columns if c.endswith('_target')]
        
        # 2. Define the specific column for the ground truth that needs to be excluded
        ground_truth_target_col = 'apparent_temperature_mean_target'
        
        # 3. The action columns are all target columns EXCEPT for the ground truth
        target_model_cols = [c for c in all_target_cols if c != ground_truth_target_col]
        
        # --- End of Correction ---
        
        if not target_model_cols:
            raise ValueError("No model prediction target columns found. Check the aligned data file and column names.")

        # Remove the '_target' suffix to get clean action names
        self.action_names = [c.replace('_target', '') for c in target_model_cols]
        self.action_space_size = len(self.action_names)

        # --- Define State Features (all non-target columns) ---
        self.state_columns = [c for c in self.data.columns if not c.endswith('_target')]
        
        # Transform the state features using the pre-fitted scaler
        self.states_normalized = self.scaler.transform(self.data[self.state_columns]).astype(np.float32)
        
        self.state_space_size = self.states_normalized.shape[1]
        self.n_days = len(self.data)
        print("Next-day environment loaded successfully.")
        print(f"State space size: {self.state_space_size}, Action space size: {self.action_space_size}")

    def _get_obs(self):
        """Returns the current observation."""
        return self.states_normalized[self.current_step]

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        """Execute one time step within the environment."""
        row = self.data.iloc[self.current_step]
        
        # --- Reward Calculation based on NEXT DAY's outcome ---
        actual_hi_next_day = row['apparent_temperature_mean_target']
        
        # Get all model predictions for the next day
        all_model_predictions_next_day = row.filter(like='_target').drop('apparent_temperature_mean_target')
        
        # Calculate errors
        all_errors = abs(actual_hi_next_day - all_model_predictions_next_day)
        min_error = all_errors.min()
        chosen_error = all_errors.iloc[action] # Relies on the order being correct
        
        # Regret-based reward
        reward = (min_error ** 2) - (chosen_error ** 2)
        
        # --- State Update ---
        self.current_step += 1
        terminated = self.current_step >= self.n_days
        truncated = False
        
        # Get next observation, or a dummy one if done
        obs = self._get_obs() if not terminated else np.zeros(self.state_space_size, dtype=np.float32)
        
        return obs, reward, terminated, truncated, {}