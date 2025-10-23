# train_model_next_day.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import the NEW environment
from weather_env_next_day import WeatherEnvNextDay

# --- File Paths for the NEW setup ---
ALIGNED_DATA_PATH = "aligned_next_day_data.csv"
SCALER_PATH = "weather_scaler_next_day.joblib"
MODEL_SAVE_PATH = "ppo_weather_model_next_day.zip" # New model name

if __name__ == "__main__":
    env = WeatherEnvNextDay(
        aligned_data_path=ALIGNED_DATA_PATH,
        scaler_path=SCALER_PATH
    )
    
    print("\n--- Checking Next-Day Environment ---")
    check_env(env)
    print("Environment check passed!")
    
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("\n--- Starting Training for Next-Day Prediction ---")
    model.learn(total_timesteps=100000) # May need more steps for this harder problem
    
    model.save(MODEL_SAVE_PATH)
    print(f"\n--- Training Finished and New Model Saved to {MODEL_SAVE_PATH} ---")