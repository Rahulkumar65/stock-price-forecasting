"""
======================================================================
 REAL-TIME STOCK PRICE FORECASTING — INDUSTRY-STYLE DATA SCIENCE PROJECT
======================================================================
A CV-worthy, real-world-style project used in quant/fintech teams:

 1. Load real historical stock data (AAPL, 1984-2008, ~6000 trading days)
 2. Engineer technical indicators used by real trading systems
    (SMA, EMA, RSI, MACD, Bollinger Bands, volatility, lag returns)
 3. Time-aware train/test split (NEVER shuffle time series data)
 4. Train 3 models: Linear Regression, Random Forest, Gradient Boosting
 5. Evaluate with RMSE, MAE, MAPE, and DIRECTIONAL ACCURACY
    (directional accuracy = the metric that actually matters for trading)
 6. WALK-FORWARD "REAL-TIME" SIMULATION — the key real-world technique:
    the model predicts tomorrow's price using only data available today,
    then the window slides forward one day at a time, just like a live
    system would operate in production (no lookahead/leakage).
 7. Feature importance — what actually drives the prediction

Run:  python3 stock_forecasting_project.py
Outputs: PNG charts + a text report saved in the same folder.
======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

sns.set_style("whitegrid")
report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

# ----------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------
log("=" * 65)
log("STEP 1: LOADING REAL HISTORICAL STOCK DATA (AAPL)")
log("=" * 65)

url = "https://raw.githubusercontent.com/matplotlib/sample_data/master/aapl.csv"
df = pd.read_csv(url, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)  # oldest -> newest, critical for time series

log(f"Rows: {len(df)}  |  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
log(f"Columns: {list(df.columns)}")
log("\nFirst 5 rows:")
log(df.head().to_string())

# ----------------------------------------------------------------
# 2. FEATURE ENGINEERING — TECHNICAL INDICATORS
# ----------------------------------------------------------------
log("\n" + "=" * 65)
log("STEP 2: FEATURE ENGINEERING (TECHNICAL INDICATORS)")
log("=" * 65)

# Moving averages
df["SMA_5"] = df["Close"].rolling(5).mean()
df["SMA_20"] = df["Close"].rolling(20).mean()
df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

# Volatility (rolling std of returns)
df["Return"] = df["Close"].pct_change()
df["Volatility_10"] = df["Return"].rolling(10).std()

# RSI (Relative Strength Index) — classic momentum indicator
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# Bollinger Bands
sma20 = df["Close"].rolling(20).mean()
std20 = df["Close"].rolling(20).std()
df["BB_upper"] = sma20 + 2 * std20
df["BB_lower"] = sma20 - 2 * std20
df["BB_width"] = df["BB_upper"] - df["BB_lower"]

# Lag features (yesterday's info, avoids leakage)
for lag in [1, 2, 3, 5]:
    df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

# Volume change
df["Volume_change"] = df["Volume"].pct_change()

# TARGET: next day's closing price (this is what a real system predicts each morning)
df["Target_next_close"] = df["Close"].shift(-1)

df_model = df.dropna().reset_index(drop=True)
log(f"Engineered {df_model.shape[1] - df.shape[1] + 15} indicator features.")
log(f"Rows available after dropping NaNs from rolling windows: {len(df_model)}")

# ----------------------------------------------------------------
# 3. EDA VISUALIZATION
# ----------------------------------------------------------------
log("\n" + "=" * 65)
log("STEP 3: EXPLORATORY VISUALIZATION")
log("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(df["Date"], df["Close"], label="Close", color="steelblue", linewidth=0.8)
axes[0, 0].plot(df["Date"], df["SMA_20"], label="SMA 20", color="orange", linewidth=1)
axes[0, 0].plot(df["Date"], df["SMA_5"], label="SMA 5", color="green", linewidth=1)
axes[0, 0].set_title("AAPL Close Price with Moving Averages")
axes[0, 0].legend()

axes[0, 1].plot(df["Date"], df["RSI_14"], color="purple", linewidth=0.8)
axes[0, 1].axhline(70, color="red", linestyle="--", linewidth=0.8)
axes[0, 1].axhline(30, color="green", linestyle="--", linewidth=0.8)
axes[0, 1].set_title("RSI (14-day) — Overbought(70)/Oversold(30)")

axes[1, 0].plot(df["Date"], df["MACD"], label="MACD", color="blue", linewidth=0.8)
axes[1, 0].plot(df["Date"], df["MACD_signal"], label="Signal", color="red", linewidth=0.8)
axes[1, 0].set_title("MACD")
axes[1, 0].legend()

axes[1, 1].plot(df["Date"], df["Volatility_10"], color="darkred", linewidth=0.8)
axes[1, 1].set_title("10-day Rolling Volatility")

plt.tight_layout()
plt.savefig("technical_indicators.png", dpi=150)
plt.close()
log("Saved chart: technical_indicators.png")

# ----------------------------------------------------------------
# 4. TIME-AWARE TRAIN/TEST SPLIT (no shuffling — critical for time series!)
# ----------------------------------------------------------------
log("\n" + "=" * 65)
log("STEP 4: TIME-AWARE TRAIN/TEST SPLIT")
log("=" * 65)

features = [
    "SMA_5", "SMA_20", "EMA_10", "Volatility_10", "RSI_14",
    "MACD", "MACD_signal", "BB_width",
    "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5",
    "Volume_change", "Open", "High", "Low", "Volume"
]
X = df_model[features]
y = df_model["Target_next_close"]

split_idx = int(len(df_model) * 0.85)  # last 15% = "future" test data
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df_model["Date"].iloc[split_idx:]

log(f"Train: {len(X_train)} days ({df_model['Date'].iloc[0].date()} to {df_model['Date'].iloc[split_idx-1].date()})")
log(f"Test:  {len(X_test)} days ({df_model['Date'].iloc[split_idx].date()} to {df_model['Date'].iloc[-1].date()})")
log("Note: test set is strictly AFTER training data in time — simulates predicting the unseen future.")

# ----------------------------------------------------------------
# 5. MODEL TRAINING
# ----------------------------------------------------------------
log("\n" + "=" * 65)
log("STEP 5: MODEL TRAINING")
log("=" * 65)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
}

predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
    log(f"Trained: {name}")

# ----------------------------------------------------------------
# 6. EVALUATION — INCLUDING DIRECTIONAL ACCURACY (the real-world metric)
# ----------------------------------------------------------------
log("\n" + "=" * 65)
log("STEP 6: MODEL EVALUATION")
log("=" * 65)

actual_direction = np.sign(y_test.values - X_test["Close_lag_1"].values)

results = []
for name, y_pred in predictions.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    pred_direction = np.sign(y_pred - X_test["Close_lag_1"].values)
    directional_acc = np.mean(pred_direction == actual_direction) * 100
    results.append([name, rmse, mae, mape, directional_acc])
    log(f"\n--- {name} ---")
    log(f"RMSE: ${rmse:.2f}   MAE: ${mae:.2f}   MAPE: {mape:.2f}%   Directional Accuracy: {directional_acc:.1f}%")

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "MAPE", "Directional_Accuracy"])
log("\nSummary table:")
log(results_df.to_string(index=False))

# Plot actual vs predicted for the best model (by RMSE)
best_model_name = results_df.sort_values("RMSE").iloc[0]["Model"]
best_pred = predictions[best_model_name]

plt.figure(figsize=(14, 6))
plt.plot(dates_test, y_test.values, label="Actual Price", color="black", linewidth=1.5)
plt.plot(dates_test, best_pred, label=f"Predicted ({best_model_name})", color="crimson", linewidth=1.2, linestyle="--")
plt.title(f"Actual vs Predicted Next-Day Close Price — {best_model_name}")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()
log(f"\nSaved chart: actual_vs_predicted.png (best model: {best_model_name})")

# ----------------------------------------------------------------
# 7. WALK-FORWARD "REAL-TIME" SIMULATION
#    This is the key real-world technique: at each step, only data up to
#    "today" is used to predict "tomorrow" — exactly like a live system.
# ----------------------------------------------------------------
log("\n" + "=" * 65)
log("STEP 7: WALK-FORWARD REAL-TIME SIMULATION (last 60 trading days)")
log("=" * 65)

sim_days = 30
sim_start = len(df_model) - sim_days
walk_actual, walk_pred, walk_dates = [], [], []

rf_live = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42, n_jobs=-1)

for i in range(sim_start, len(df_model)):
    # Train ONLY on data strictly before day i (no future leakage — real-time constraint)
    train_X = X.iloc[:i]
    train_y = y.iloc[:i]
    rf_live.fit(train_X, train_y)

    today_features = X.iloc[[i]]
    pred_tomorrow = rf_live.predict(today_features)[0]

    walk_pred.append(pred_tomorrow)
    walk_actual.append(y.iloc[i])
    walk_dates.append(df_model["Date"].iloc[i])

walk_rmse = np.sqrt(mean_squared_error(walk_actual, walk_pred))
walk_direction_actual = np.sign(np.array(walk_actual) - X["Close_lag_1"].iloc[sim_start:len(df_model)].values)
walk_direction_pred = np.sign(np.array(walk_pred) - X["Close_lag_1"].iloc[sim_start:len(df_model)].values)
walk_dir_acc = np.mean(walk_direction_actual == walk_direction_pred) * 100

log(f"Walk-forward RMSE over last {sim_days} days: ${walk_rmse:.2f}")
log(f"Walk-forward directional accuracy: {walk_dir_acc:.1f}%")
log("(Model is retrained fresh at every step using only past data — zero lookahead bias.)")

plt.figure(figsize=(14, 6))
plt.plot(walk_dates, walk_actual, label="Actual Next-Day Price", color="black", marker="o", markersize=3)
plt.plot(walk_dates, walk_pred, label="Real-Time Predicted Price", color="crimson", marker="x", markersize=3, linestyle="--")
plt.title(f"Walk-Forward Real-Time Simulation — Last {sim_days} Trading Days")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("walk_forward_simulation.png", dpi=150)
plt.close()
log("Saved chart: walk_forward_simulation.png")

# ----------------------------------------------------------------
# 8. FEATURE IMPORTANCE
# ----------------------------------------------------------------
log("\n" + "=" * 65)
log("STEP 8: FEATURE IMPORTANCE (Random Forest)")
log("=" * 65)

rf_final = models["Random Forest"]
importance = pd.Series(rf_final.feature_importances_, index=features).sort_values(ascending=False)
log(importance.to_string())

plt.figure(figsize=(9, 6))
importance.plot(kind="barh", color="teal")
plt.title("Feature Importance — What Drives the Prediction")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
log("Saved chart: feature_importance.png")

# ----------------------------------------------------------------
# SAVE FULL REPORT
# ----------------------------------------------------------------
with open("project_report.txt", "w") as f:
    f.write("\n".join(report_lines))

log("\n" + "=" * 65)
log("PROJECT COMPLETE. Report saved as project_report.txt")
log("=" * 65)
