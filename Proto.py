import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📈 Minimal Sharpe Ratio Optimizer")

# --- User input ---
tickers = st.text_input("Enter comma-separated tickers:", "AAPL,MSFT,GOOG,TSLA").split(",")
start = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end = st.date_input("End Date", pd.to_datetime("2024-12-31"))

if st.button("Run Optimization"):

    # --- Fetch price data ---
    st.subheader("🔍 Fetching Data")
    prices = yf.download(tickers, start=start, end=end)["Adj Close"]
    st.write(prices.tail())

    # --- Compute returns and risk model ---
    st.subheader("📊 Computing Metrics")
    returns = prices.pct_change().dropna()
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # --- Optimize portfolio ---
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    st.write("📌 Optimal Weights:")
    st.json(cleaned_weights)

    # --- Plot results ---
    st.subheader("📈 Portfolio Allocation")
    weights_df = pd.Series(cleaned_weights).sort_values(ascending=False)
    st.bar_chart(weights_df)
