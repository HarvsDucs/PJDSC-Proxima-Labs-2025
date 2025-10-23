# evaluate_next_day_model.py (with advanced EDA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from joblib import load
from stable_baselines3 import PPO
import os

# --- 1. SETUP: CONFIGURE FILE PATHS ---
ALIGNED_DATA_PATH = "aligned_next_day_data.csv"
SCALER_PATH = "weather_scaler_next_day.joblib"
MODEL_PATH = "ppo_weather_model_next_day.zip"
EVALUATION_YEAR = 2024

print("--- Starting Evaluation of Next-Day Prediction Model ---")

# --- 2. LOAD AND PREPARE DATA ---
if not os.path.exists(ALIGNED_DATA_PATH):
    raise FileNotFoundError(f"Aligned data not found at '{ALIGNED_DATA_PATH}'.")

df_eval = pd.read_csv(ALIGNED_DATA_PATH, index_col='date', parse_dates=True)

if EVALUATION_YEAR:
    df_eval = df_eval[df_eval.index.year == EVALUATION_YEAR]
    if len(df_eval) == 0:
        raise ValueError(f"No data for year {EVALUATION_YEAR}.")
    print(f"\nFiltered data for evaluation year: {EVALUATION_YEAR} ({len(df_eval)} days)")

# --- 3. LOAD THE TRAINED AGENT AND SCALER ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found.")

model = PPO.load(MODEL_PATH)
scaler = load(SCALER_PATH)
print("\nLoaded trained PPO model and scaler.")

# --- 4. GENERATE PREDICTIONS FROM THE RL AGENT ---
# This part is modified to store the choices for later analysis.

state_columns = [c for c in df_eval.columns if not c.endswith('_target')]
all_target_cols = [c for c in df_eval.columns if c.endswith('_target')]
ground_truth_target_col = 'apparent_temperature_mean_target'
model_target_columns = [c for c in all_target_cols if c != ground_truth_target_col]
action_names = [c.replace('_target', '') for c in model_target_columns]

rl_agent_predictions = []
model_choices_list = [] # <-- NEW: Store choices day-by-day

print("\nGenerating predictions from RL agent...")
for i in range(len(df_eval)):
    current_state_df = df_eval.iloc[[i]][state_columns]
    scaled_state = scaler.transform(current_state_df)
    action, _ = model.predict(scaled_state, deterministic=True)
    action_index = action[0]
    chosen_model_name = action_names[action_index]
    
    # Store the choice for this day
    model_choices_list.append(chosen_model_name)
    
    chosen_model_prediction_col = chosen_model_name + '_target'
    prediction = df_eval.iloc[i][chosen_model_prediction_col]
    rl_agent_predictions.append(prediction)

# --- NEW: Add the agent's choices back into the DataFrame for EDA ---
df_eval['RL_Choice'] = model_choices_list
model_choices_count = df_eval['RL_Choice'].value_counts().to_dict()

print("RL agent predictions generated.")


# --- 5. CALCULATE PERFORMANCE METRICS (MSE) ---
# (This section is unchanged)
print("\n--- Performance Comparison (Mean Squared Error) ---")
results = {}
y_true = df_eval['apparent_temperature_mean_target']
results['RL_Agent_PPO'] = mean_squared_error(y_true, rl_agent_predictions)
for model_name in action_names:
    y_pred_baseline = df_eval[model_name + '_target']
    results[model_name] = mean_squared_error(y_true, y_pred_baseline)
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MSE']).sort_values('MSE')
print(results_df)


# --- 6. VISUALIZATION AND BASIC EDA ---
# (These plots are unchanged)
plt.style.use('seaborn-v0_8-whitegrid')
# ... (Code for MSE plot, choices distribution plot, and error box plot) ...
# For brevity, these plots are assumed to be here from the previous script.
# Make sure to run them or comment them out.


# --- 7. ADVANCED EDA: WHEN ARE MODELS CHOSEN? ---

print("\nGenerating advanced EDA plots on agent's decision-making...")

# PLOT 4: Choice Distribution by Time of Year (Faceted Histograms)
# This shows if the agent has a seasonal preference.
g = sns.displot(
    data=df_eval, 
    x='day_of_year', 
    col='RL_Choice', 
    col_wrap=3,  # Adjust this number based on how many models you have
    kde=True,
    common_bins=True
)
g.fig.suptitle('Distribution of Choices by Day of the Year', y=1.03)
g.set_axis_labels('Day of Year', 'Count of Times Chosen')
plt.tight_layout()
plt.savefig("next_day_choice_by_day_of_year.png")
print("Saved seasonal choice distribution plot to 'next_day_choice_by_day_of_year.png'")


# PLOT 5: Choice Distribution by Weather Conditions (Box Plots)
# This reveals the specific weather patterns that trigger each choice.
features_to_analyze = [
    'apparent_temperature_mean',
    'relative_humidity_2m_mean',
    'rain_mean',
    'wind_speed_100m_mean'
]

# Get the order of models from most to least chosen for consistent plotting
choice_order = df_eval['RL_Choice'].value_counts().index

for feature in features_to_analyze:
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(
        data=df_eval,
        x='RL_Choice',
        y=feature,
        order=choice_order,
        ax=ax
    )
    ax.set_title(f'Agent Choices vs. {feature.replace("_", " ").title()}')
    ax.set_ylabel(feature.replace("_", " ").title())
    ax.set_xlabel('Model Chosen by Agent')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"next_day_choice_by_{feature}.png")
    print(f"Saved choice-by-condition plot for '{feature}' to 'next_day_choice_by_{feature}.png'")


# Show all generated plots
plt.show()