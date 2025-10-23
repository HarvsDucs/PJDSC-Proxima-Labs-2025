# app_next_day.py
import gradio as gr
import numpy as np
from joblib import load
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
import os

# --- 1. IMPORT THE CORRECT CUSTOM ENVIRONMENT ---
# This app requires the environment designed for the next-day prediction task.
from weather_env_next_day import WeatherEnvNextDay

# --- 2. CONFIGURE AND LOAD ASSETS ---

# --- IMPORTANT: UPDATE THESE FOR YOUR NEW MODEL ---
# You should create a new repository on the Hub for this V2 model to avoid confusion.
REPO_ID = "YourUsername/Your-Next-Day-Model-Repo" 
MODEL_FILENAME = "ppo_weather_model_next_day.zip" 

# These files MUST be uploaded to your Hugging Face Space.
SCALER_PATH = "weather_scaler_next_day.joblib"
# The environment needs the aligned data file to initialize and get metadata.
ALIGNED_DATA_PATH = "aligned_next_day_data.csv"

# --- Check for file existence before loading ---
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found at '{SCALER_PATH}'. Make sure you've run the data alignment script.")
if not os.path.exists(ALIGNED_DATA_PATH):
    raise FileNotFoundError(f"Aligned data not found at '{ALIGNED_DATA_PATH}'. Make sure you've run the data alignment script.")

print("Loading scaler for next-day prediction...")
SCALER = load(SCALER_PATH)

print("Loading next-day prediction model from Hub...")
checkpoint = load_from_hub(repo_id=REPO_ID, filename=MODEL_FILENAME)
model = PPO.load(checkpoint)
print("Model loaded successfully.")

print("Instantiating environment to get metadata...")
# Instantiate the environment to get the correct action names and state column order.
temp_env = WeatherEnvNextDay(
    aligned_data_path=ALIGNED_DATA_PATH,
    scaler_path=SCALER_PATH,
)
ACTION_NAMES = temp_env.action_names
STATE_COLUMNS = temp_env.state_columns
del temp_env
print(f"Ready for next-day prediction. Expecting state features: {STATE_COLUMNS}")

# Store the means from the scaler for robust imputation of missing values.
IMPUTATION_VALUES = SCALER.mean_ 

# --- 3. DEFINE THE PREDICTION FUNCTION ---

def predict_best_model_for_tomorrow(day_of_year_today, *weather_features_today):
    """
    Takes today's conditions, imputes missing values, scales the data,
    and returns the agent's recommended model for tomorrow.
    """
    try:
        raw_inputs = [day_of_year_today] + list(weather_features_today)
        
        imputed_inputs = []
        for i, value in enumerate(raw_inputs):
            if value is None or not isinstance(value, (int, float)):
                imputed_inputs.append(IMPUTATION_VALUES[i]) # Use pre-calculated mean
            else:
                imputed_inputs.append(value)

        state_array = np.array(imputed_inputs, dtype=np.float32).reshape(1, -1)
        
        if state_array.shape[1] != len(SCALER.mean_):
            return f"Error: Incorrect number of features. Expected {len(SCALER.mean_)}, got {state_array.shape[1]}."

        scaled_state = SCALER.transform(state_array)
        action, _ = model.predict(scaled_state, deterministic=True)
        chosen_model_name = ACTION_NAMES[int(action)]
        
        return f"🤖 **For tomorrow, the agent recommends trusting:** `{chosen_model_name}`"
    except Exception as e:
        return f"An error occurred during prediction: {e}"

# --- 4. CREATE THE GRADIO INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔮 Next-Day Weather Model Predictor")
    gr.Markdown(
        "Based on **today's** observed weather conditions, the Reinforcement Learning agent will predict which forecast model is most likely to be accurate for **tomorrow**."
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Step 1: Input Today's Weather Conditions")
            
            # Dynamically create input components based on the state columns
            input_components = []
            for feature_name in STATE_COLUMNS:
                # Make labels more user-friendly
                label = feature_name.replace('_', ' ').title()
                if feature_name == 'day_of_year':
                    input_components.append(gr.Slider(1, 366, step=1, label=label))
                else:
                    input_components.append(gr.Number(label=label))

        with gr.Column(scale=1):
            gr.Markdown("### Step 2: Get Tomorrow's Recommendation")
            output_component = gr.Markdown(label="Recommended Model for Tomorrow")
            predict_btn = gr.Button("Predict for Tomorrow", variant="primary")

    predict_btn.click(
        fn=predict_best_model_for_tomorrow,
        inputs=input_components,
        outputs=output_component
    )

    gr.Markdown("---")
    gr.Markdown("This application uses a PPO agent trained with Stable Baselines3. The model and interface are hosted on Hugging Face.")

# This allows the app to be run directly for local testing
if __name__ == "__main__":
    demo.launch()