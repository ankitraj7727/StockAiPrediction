"""
model.py
---------
Builds a labeled dataset (features -> did price go up over the next
`horizon` trading days?) and trains a Gradient Boosting classifier.

IMPORTANT: all train/test splitting is done in TIME ORDER (no shuffling).
Shuffling time-series data before splitting leaks future information into
training and makes accuracy numbers meaningless - a common mistake in
"AI trading" projects that makes them look far better than they really are.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from indicators import add_features, FEATURE_COLUMNS


def build_dataset(raw_df: pd.DataFrame, horizon: int = 5):
    """
    Returns (features_df, labels_series, full_feature_df) where:
      - labels[i] = 1 if Close[i + horizon] > Close[i], else 0
      - rows with NaNs (from indicator warm-up or the label lookahead
        window at the very end) are dropped
    """
    feat_df = add_features(raw_df)

    future_close = feat_df["Close"].shift(-horizon)
    label = (future_close > feat_df["Close"]).astype(int)
    feat_df = feat_df.copy()
    feat_df["label"] = label

    clean = feat_df.dropna(subset=FEATURE_COLUMNS + ["label"])
    X = clean[FEATURE_COLUMNS]
    y = clean["label"]
    return X, y, feat_df


def train_model(X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X, y)
    return model


def predict_proba_up(model: GradientBoostingClassifier, X_row: pd.DataFrame) -> float:
    """Probability the price rises over the prediction horizon."""
    proba = model.predict_proba(X_row)[:, 1]
    return float(proba[-1])
