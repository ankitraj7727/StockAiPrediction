# Market Direction Predictor

A full, dynamic web app that:
- Lets a user type any symbol Yahoo Finance tracks (stocks, crypto, forex, indices) and a date range
- Trains an ML model on technical indicators to predict whether price will rise over the next N trading days
- Backtests itself honestly on data it wasn't trained on, and shows real accuracy/return numbers
- Shows today's live BUY / SELL / HOLD signal with a confidence percentage
- Plots buy/sell markers on the price chart and an equity curve vs. plain buy-and-hold

## Project structure
```
market_predictor/
├── app.py            # Streamlit web app (the website itself)
├── data_fetcher.py   # Pulls historical OHLCV data from Yahoo Finance (yfinance)
├── indicators.py      # Technical indicator feature engineering (RSI, MACD, SMA, Bollinger, etc.)
├── model.py           # Builds the training dataset + trains the GradientBoosting classifier
├── backtest.py        # Walk-forward backtest + trade simulation (the "is this real" check)
└── requirements.txt
```

## Run it locally
```bash
cd market_predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
It opens at `http://localhost:8501`. Type a symbol (e.g. `AAPL`, `BTC-USD`, `EURUSD=X`), pick a date range, click **Run Analysis**.

## Publish it as a real website (free)
**Option A — Streamlit Community Cloud (easiest, free)**
1. Push this folder to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app," pick the repo and `app.py`, deploy.
4. You get a public URL like `https://your-app.streamlit.app` you can open on any device.

**Option B — Render / Railway (more control, still free tier available)**
- Create a new "Web Service," point it at your repo.
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## What this actually does — and doesn't do
This model learns statistical patterns between technical indicators (moving averages, RSI, MACD, volatility, volume) and short-term price direction. What it **does not** know about:
- Earnings reports, news, management changes, macro data releases
- Order-book depth, market maker behavior, institutional flow
- Anything that hasn't happened before in the training window

Because of this, treat the accuracy number the backtest gives you as the real, honest answer to "how good is this model," not as a target you should expect to beat in live trading. A directional accuracy meaningfully above 50-55% sustained across different symbols and time periods would already be a strong result for a model of this kind — be skeptical of any version of this (or any trading tool) that claims much higher than that.

## Before using this with real money
1. **Paper trade first.** Run the signals forward in real time without real money for at least a few weeks/months and track whether the live signals actually work as well as the backtest suggested.
2. **Re-run the backtest regularly.** Markets change regime; a model trained on 2022-2024 data may not hold up in 2026. Retrain periodically on recent data.
3. **Position size conservatively.** Never risk money you can't afford to lose, and consider this one input among several, not a sole decision-maker.
4. **This is not financial advice**, and neither Claude nor Anthropic is a registered financial advisor. You are responsible for your own trading decisions.

## Ideas to extend it
- Add more symbols to a watchlist view that runs predictions on several tickers at once
- Add sentiment features from news headlines
- Try an ensemble of models (Random Forest + Gradient Boosting + Logistic Regression) and average their probabilities
- Add email/SMS alerts when a live BUY/SELL signal fires
- Track live paper-trading performance in a small database so you can compare backtest vs. live results over time
