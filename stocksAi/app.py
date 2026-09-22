"""
app.py
-------
Run locally with:   streamlit run app.py
Deploy for free on Streamlit Community Cloud (see README.md).
"""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_fetcher import fetch_data, DataFetchError
from backtest import walk_forward_backtest
from model import predict_proba_up

st.set_page_config(page_title="Market Direction Predictor", layout="wide")

st.title("📈 Market Direction Predictor")
st.caption(
    "Enter any symbol Yahoo Finance tracks — stock, crypto or forex — and a "
    "date range. The model is trained on the earlier part of that range and "
    "tested, honestly, on the part it never saw."
)

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input(
        "Symbol",
        value="AAPL",
        help="Examples: AAPL, TSLA, RELIANCE.NS, BTC-USD, ETH-USD, EURUSD=X, ^GSPC",
    )
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "From", value=dt.date.today() - dt.timedelta(days=365 * 3)
        )
    with col2:
        end_date = st.date_input("To", value=dt.date.today())

    horizon = st.slider(
        "Prediction horizon (trading days ahead)", min_value=1, max_value=20, value=5
    )
    buy_threshold = st.slider("Buy signal threshold (probability)", 0.50, 0.90, 0.55, 0.01)
    sell_threshold = st.slider("Sell signal threshold (probability)", 0.10, 0.50, 0.45, 0.01)
    train_frac = st.slider(
        "Fraction of data used for training (rest = honest test)", 0.5, 0.9, 0.7, 0.05
    )

    run = st.button("Run Analysis", type="primary", use_container_width=True)

st.warning(
    "⚠️ **This is not financial advice.** This tool reports a model's historical "
    "directional accuracy on data it wasn't trained on — it does not guarantee "
    "future performance. Markets are influenced by news, earnings, macro events "
    "and other factors this model does not see. If you use this with real money, "
    "start with a paper-trading account and position sizes you can afford to lose.",
    icon="⚠️",
)

if run:
    if sell_threshold >= buy_threshold:
        st.error("Sell threshold must be lower than the buy threshold.")
        st.stop()

    try:
        with st.spinner(f"Fetching data for {symbol}..."):
            raw_df = fetch_data(symbol, str(start_date), str(end_date))
    except DataFetchError as e:
        st.error(str(e))
        st.stop()

    try:
        with st.spinner("Training model and running walk-forward backtest..."):
            result = walk_forward_backtest(
                raw_df,
                horizon=horizon,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                train_frac=train_frac,
            )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    metrics = result["metrics"]
    signals = result["signals"]
    equity = result["equity_curve"]

    # ---- Today's live signal ----
    latest_proba = predict_proba_up(result["model"], result["latest_features"])
    if latest_proba >= buy_threshold:
        live_signal, color = "BUY", "green"
    elif latest_proba <= sell_threshold:
        live_signal, color = "SELL", "red"
    else:
        live_signal, color = "HOLD", "gray"

    st.subheader(f"Current signal for {symbol.upper()} as of {result['latest_date'].date()}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model signal", live_signal)
    c2.metric(f"Probability price rises in {horizon} trading day(s)", f"{latest_proba*100:.1f}%")
    c3.metric("Backtested directional accuracy", f"{metrics['directional_accuracy']*100:.1f}%")

    st.markdown(f":{color}[**{live_signal}**] — based on the model trained on data through "
                f"the training window, applied to the most recent available day.")

    st.divider()

    # ---- Price chart with buy/sell markers from the backtest period ----
    st.subheader(f"Backtest period: {metrics['test_period_start']} to {metrics['test_period_end']}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=signals.index, y=signals["Close"], mode="lines", name="Close price"))
    buys = signals[signals["signal"] == "BUY"]
    sells = signals[signals["signal"] == "SELL"]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="Buy signal",
                              marker=dict(color="green", size=9, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="Sell signal",
                              marker=dict(color="red", size=9, symbol="triangle-down")))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10),
                       legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # ---- Backtest metrics ----
    st.subheader("How good is this, honestly?")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Directional accuracy", f"{metrics['directional_accuracy']*100:.1f}%")
    m2.metric("Number of trades", metrics["n_trades"])
    win_rate = metrics["win_rate"]
    m3.metric("Win rate", f"{win_rate*100:.1f}%" if win_rate == win_rate else "n/a")
    m4.metric("Strategy return", f"{metrics['strategy_return_pct']:.1f}%")
    m5.metric("Buy & hold return", f"{metrics['buy_hold_return_pct']:.1f}%")

    st.caption(
        f"Max drawdown during the test period: {metrics['max_drawdown_pct']:.1f}%. "
        f"Directional accuracy of 50% is what random guessing would achieve on a roughly "
        f"balanced market — compare against that, not against 100%."
    )

    # ---- Equity curve: strategy vs buy & hold ----
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=equity.index, y=equity["strategy_equity"], mode="lines",
                               name="Model strategy"))
    fig2.add_trace(go.Scatter(x=equity.index, y=equity["buy_hold_equity"], mode="lines",
                               name="Buy & hold", line=dict(dash="dash")))
    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h"),
                        yaxis_title="Growth of $1")
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Raw signal table"):
        st.dataframe(signals.style.format({"Close": "{:.2f}", "proba_up": "{:.3f}"}))

else:
    st.info("Set your symbol and date range in the sidebar, then click **Run Analysis**.")
