# create_aligned_data.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

# --- Configuration ---
ACTUAL_WEATHER_PATH = r"C:\Users\Harvey\Downloads\manila_weather_cleaned_2.csv"
MODEL_PREDICTIONS_PATH = r"C:\Users\Harvey\Downloads\2017 - October 2025 Weather Forecast Data (Heat Index)_2.csv"
OUTPUT_SCALER_PATH = "weather_scaler_next_day.joblib" # New scaler file

print("--- Starting Data Alignment for Next-Day Prediction ---")

# 1. Load original data
actual_df = pd.read_csv(ACTUAL_WEATHER_PATH)
models_df = pd.read_csv(MODEL_PREDICTIONS_PATH)

# Basic cleaning (ensure dates are correct)
actual_df['date'] = pd.to_datetime(actual_df['date'], dayfirst=True).dt.normalize()
models_df.rename(columns={'time': 'date'}, inplace=True)
models_df['date'] = pd.to_datetime(models_df['date']).dt.normalize()
actual_df.set_index('date', inplace=True)
models_df.set_index('date', inplace=True)

# 2. Prepare the two sets of columns: State (Day T) and Target (Day T+1)
full_df = actual_df.join(models_df, how='inner').dropna()
full_df['day_of_year'] = full_df.index.dayofyear

# Columns that will form the STATE (what the agent sees from Day T)
state_feature_cols = ['day_of_year'] + actual_df.columns.tolist()

# Columns that will be used for the REWARD (the outcome on Day T+1)
# We need the actual heat index and all model predictions for the next day.
target_cols = ['apparent_temperature_mean'] + models_df.columns.tolist()

# 3. Create the shifted (lagged) DataFrame
print("Aligning Day T (State) with Day T+1 (Target)...")
aligned_df = full_df[state_feature_cols].copy() # Start with Day T features

# Use shift(-1) to pull the NEXT day's data up to the current row
for col in target_cols:
    aligned_df[f'{col}_target'] = full_df[col].shift(-1)

# The last row has no "next day", so its target values are NaN. Drop it.
aligned_df.dropna(inplace=True)

# 4. Create and save the NEW scaler
# The scaler must ONLY be trained on the state features.
print(f"Fitting new scaler on {len(state_feature_cols)} state features...")
scaler = StandardScaler()
scaler.fit(aligned_df[state_feature_cols])

dump(scaler, OUTPUT_SCALER_PATH)
print(f"✅ New scaler for next-day prediction saved to '{OUTPUT_SCALER_PATH}'")

# Optional: Save the aligned data for inspection
aligned_df.to_csv("aligned_next_day_data.csv")
print("✅ Aligned data saved to 'aligned_next_day_data.csv' for inspection.")