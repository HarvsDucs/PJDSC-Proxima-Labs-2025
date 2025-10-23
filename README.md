# Project Heat Resilience PH: A Reinforcement Learning Approach towards Heat Index Forecast’s Adaptive Model Selection (HIFAMS)

### EXECUTIVE SUMMARY

The Philippines—particularly its densely populated capital, Manila—is acutely vulnerable to extreme heat events intensified by climate change. Under the Project Heat Resilience PH initiative, this study advances a **data-driven forecasting approach** designed to strengthen heat-risk preparedness. Rather than optimizing suspension policies directly, we focus on **improving next-day Heat Index (HI) forecasts**, recognizing that more accurate and reliable predictions are foundational to timely, proportionate decisions by education and public-health authorities. Static heat-index thresholds remain important for action, but their effectiveness depends on the **quality of the forecasts** that trigger them.

This work reframes the problem as **Adaptive Model Selection (AMS)**: each day, the system selects **one best predictive model** from a portfolio (the “model zoo”) based on the day’s context (e.g., meteorological features, season/regime flags, and recent model errors). We implement this as a contextual bandit—using lightweight, data-efficient policies such as LinUCB and Contextual Thompson Sampling—that learns from daily outcomes without requiring a full simulation of real-world interventions. The learning signal is a **differential reward** that compares the chosen model’s error against a **Best-in-Class Benchmark**:

$$ R_{t+1}= (y_{t+1}-\hat{y}_{t+1}_{a*})2 -\min(y_{t+1}-\hat{y}_{t+1}_m)2 $$

so **positive rewards** indicate improvement over the benchmark. This formulation stabilizes learning, stays computationally tractable, and directly targets **predictive utility.** (A threshold-aware penalty near critical HI cutoffs in the Philippines: e.g., 42 °C and 46 °C, can be incorporated as an extension when stakeholders require it.)

Built on daily Open-Meteo inputs and historical observations for Metro Manila, the pipeline **runs all candidate models**, publishes only the selected forecast, and updates the policy when truth arrives the next day—enabling transparent **regret tracking** and steady improvement. In doing so, the system aims to **reduce forecast errors precisely when decisions matter most**, thereby improving the information available to decision-makers (e.g., schools, LGUs, and health agencies). While **policy enactment** (such as class suspensions) remains outside this project’s operational scope, the proposed AMS framework provides a **robust, reproducible, and testable** foundation for **more effective, timely, and justified** heat-risk responses across the Philippines.


### KEYWORDS

Adaptive Model Selection (AMS), Contextual Bandits, Heat Index Forecasting, Metro Manila, Philippines, Extreme Heat Events, Climate Resilience, Decision Support Systems, Differential Reward, Benchmark Model (Best-in-Class), Regret Minimization, Time-Series Forecasting, Numerical Weather Prediction (NWP), LinUCB, Contextual Thompson Sampling (CTS), Threshold-Aware Penalties (42 °C, 46 °C), Open-Meteo Data, Model Zoo, Education Policy (Class Suspension), Public Health Preparedness, Project Heat Resilience PH
