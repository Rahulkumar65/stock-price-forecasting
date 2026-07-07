# 📈 Real-Time Stock Price Forecasting — Industry-Style Data Science Project

A project built the way real quant/fintech teams actually approach price
forecasting — not just "fit a model on shuffled data," but with the
techniques and rigor a hiring manager will recognize.

## Why this is CV-worthy (and Titanic isn't)
- Uses **real historical market data** (AAPL daily OHLCV), not a toy dataset.
- Engineers the same **technical indicators used in production trading systems**
  (RSI, MACD, Bollinger Bands, moving averages, volatility).
- Uses a **time-aware split** — never shuffles time series data (a classic
  beginner mistake that silently leaks the future into training).
- Includes **walk-forward validation**, the real-world technique for testing
  time series models: at every step the model only sees data up to "today"
  and predicts "tomorrow," then the window slides forward one day —
  exactly how a live production system would run each morning.
- Reports **directional accuracy** (did it call the price move up or down?),
  which is the metric that actually matters for trading, not just RMSE.

## Pipeline (all in `stock_forecasting_project.py`)
1. **Load data** — real AAPL daily price history (1984–2008, ~6,000 trading days), pulled directly from GitHub.
2. **Feature engineer technical indicators**: SMA(5/20), EMA(10), RSI(14), MACD + signal line, Bollinger Bands, 10-day rolling volatility, lagged closes, volume change.
3. **Time-aware train/test split** — last 15% of days held out as "the future," never touched during training.
4. **Train 3 models**: Linear Regression, Random Forest, Gradient Boosting.
5. **Evaluate**: RMSE, MAE, MAPE, and directional accuracy for each model.
6. **Walk-forward real-time simulation**: over the last 30 trading days, retrain the model fresh at every single step using only past data, predict tomorrow, slide forward one day, repeat — zero lookahead bias, just like production.
7. **Feature importance** — which indicators the model actually relies on.

## Results (this run)
| Model | RMSE | MAE | MAPE | Directional Accuracy |
|---|---|---|---|---|
| Linear Regression | $3.14 | $2.08 | 2.04% | **73.7%** |
| Random Forest | $19.92 | $10.52 | 7.03% | 60.4% |
| Gradient Boosting | $19.72 | $10.52 | 7.09% | 63.1% |

**Walk-forward (real-time) simulation**, last 30 days: RMSE $7.56, directional accuracy 70.0%.

Interesting real-world finding: the simple Linear Regression beat the tree
models on next-day price — a well-known phenomenon in finance, since price
is highly autocorrelated day-to-day and linear models exploit that cleanly.
This is exactly the kind of insight that's good to talk about in an interview.

Top predictive features: **Open price of the same day, SMA-5, EMA-10, Close_lag_1** — confirming the model leans heavily on very recent price action, which is realistic for daily-frequency forecasting.

## Files in this project
- `stock_forecasting_project.py` — the full runnable pipeline
- `project_report.txt` — text log of every step's output
- `technical_indicators.png` — price with moving averages, RSI, MACD, volatility
- `actual_vs_predicted.png` — best model's predictions vs actual price on held-out test data
- `walk_forward_simulation.png` — the real-time simulation results
- `feature_importance_rf.png` — which indicators drive predictions

## How to run it yourself
```bash
pip install pandas scikit-learn matplotlib seaborn
python3 stock_forecasting_project.py
```

## How to talk about this on your CV / in interviews
- "Built a time-series forecasting pipeline for stock prices using engineered technical indicators (RSI, MACD, Bollinger Bands) and walk-forward validation to simulate real-time production deployment, avoiding lookahead bias."
- "Compared linear and ensemble models, evaluating on both price error (RMSE/MAPE) and directional accuracy — the metric relevant to trading decisions."
- Be ready to explain: why you can't shuffle time series data, what walk-forward validation is and why it matters, and why directional accuracy can matter more than RMSE for a trading use case.

## Ideas to extend this project further
- Add an LSTM/GRU deep learning model and compare against the classical models
- Add multiple tickers and turn it into a portfolio-level forecasting tool
- Wrap the walk-forward loop into a live dashboard (Streamlit) showing "today's prediction"
- Backtest a simple trading strategy based on the directional predictions and report Sharpe ratio / returns
- Pull live data via an API (e.g., a market data provider) to make it truly real-time instead of historical replay
