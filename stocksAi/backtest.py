"""
backtest.py
------------
This is the most important file in the project. It answers the question
you actually care about: "is this model's prediction any good, or not?"

It does a WALK-FORWARD backtest:
  - train on the first `train_frac` of the data
  - test only on the data that comes AFTER the training window
    (the model never sees test-period data during training)
  - simulate simple long-only trades based on the model's signal
  - compare the result to a plain buy-and-hold of the same period

This is not a guarantee of future performance. It is a report card on
how the model would have done, historically, on data it wasn't trained on.
"""

import numpy as np
import pandas as pd

from model import build_dataset, train_model, predict_proba_up


def walk_forward_backtest(
    raw_df: pd.DataFrame,
    horizon: int = 5,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.45,
    train_frac: float = 0.7,
):
    """
    Returns a dict with:
      metrics: dict of summary stats
      equity_curve: DataFrame(date, strategy_equity, buy_hold_equity)
      signals: DataFrame(date, close, proba_up, signal) for the test period
      model: the trained model (fit on the training window only)
      latest_features: the feature row for the most recent day (for a live prediction)
    """
    X, y, feat_df = build_dataset(raw_df, horizon=horizon)

    n = len(X)
    split = int(n * train_frac)
    if split < 60 or (n - split) < 30:
        raise ValueError(
            "Not enough data to reliably backtest. Use a longer date range "
            "(at least ~1-2 years of daily data is recommended)."
        )

    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test = X.iloc[split:]

    model = train_model(X_train, y_train)

    test_index = X_test.index
    proba_up = model.predict_proba(X_test)[:, 1]

    test_df = feat_df.loc[test_index, ["Close"]].copy()
    test_df["proba_up"] = proba_up

    signal = np.where(
        test_df["proba_up"] >= buy_threshold, "BUY",
        np.where(test_df["proba_up"] <= sell_threshold, "SELL", "HOLD"),
    )
    test_df["signal"] = signal

    # --- simple long-only trade simulation ---
    cash = 1.0
    position = 0.0  # units of the asset held
    equity_curve = []
    entries = []
    exits = []

    for date, row in test_df.iterrows():
        price = row["Close"]
        if row["signal"] == "BUY" and position == 0.0:
            position = cash / price
            cash = 0.0
            entries.append((date, price))
        elif row["signal"] == "SELL" and position > 0.0:
            cash = position * price
            position = 0.0
            exits.append((date, price))
        equity = cash + position * price
        equity_curve.append(equity)

    # liquidate any open position at the last price so returns are comparable
    if position > 0.0:
        cash = position * test_df["Close"].iloc[-1]
        position = 0.0
    final_equity = cash if cash > 0 else equity_curve[-1]

    test_df["strategy_equity"] = equity_curve
    start_price = test_df["Close"].iloc[0]
    test_df["buy_hold_equity"] = test_df["Close"] / start_price

    strategy_return = final_equity - 1.0
    buy_hold_return = test_df["buy_hold_equity"].iloc[-1] - 1.0

    # directional accuracy: was proba_up on the right side of what actually happened?
    actual_up = y.loc[test_index]
    predicted_up = (test_df["proba_up"] >= 0.5).astype(int)
    accuracy = float((predicted_up.values == actual_up.values).mean())

    n_trades = len(entries)
    win_trades = 0
    for i in range(min(len(entries), len(exits))):
        if exits[i][1] > entries[i][1]:
            win_trades += 1
    win_rate = (win_trades / len(exits)) if exits else float("nan")

    running_max = test_df["strategy_equity"].cummax()
    drawdown = (test_df["strategy_equity"] - running_max) / running_max
    max_drawdown = float(drawdown.min())

    metrics = {
        "test_period_start": str(test_index[0].date()),
        "test_period_end": str(test_index[-1].date()),
        "directional_accuracy": accuracy,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "strategy_return_pct": strategy_return * 100,
        "buy_hold_return_pct": buy_hold_return * 100,
        "max_drawdown_pct": max_drawdown * 100,
    }

    # latest row (most recent day overall) for a live "what should I do today" prediction
    latest_features = X.iloc[[-1]]

    return {
        "metrics": metrics,
        "equity_curve": test_df[["strategy_equity", "buy_hold_equity"]],
        "signals": test_df[["Close", "proba_up", "signal"]],
        "model": model,
        "latest_features": latest_features,
        "latest_date": feat_df.index[-1],
    }
