# swing_trading_dashboard_live.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

st.set_page_config(page_title="Swing Trading Dashboard", layout="wide")

# --- Helper functions ---
def fetch_data(symbol, period="1y"):
    """Hent historiske priser fra yfinance"""
    df = yf.Ticker(symbol).history(period=period)
    df.reset_index(inplace=True)
    return df[['Date', 'Close']].rename(columns={'Close': symbol})

def create_lstm_prediction(prices, future_days=30):
    """Simuler LSTM-prediktion (placeholder for demo)"""
    return prices[-1] + np.cumsum(np.random.randn(future_days))

def generate_trade_ideas(latest_price):
    """Generer simple LONG/SHORT forslag"""
    if np.random.rand() > 0.5:
        return {"Type": "LONG", "Entry": latest_price, "Target": latest_price*1.05, "Stop": latest_price*0.98}
    else:
        return {"Type": "SHORT", "Entry": latest_price, "Target": latest_price*0.95, "Stop": latest_price*1.02}

# --- Fetch live data ---
corn_symbol = "ZC=F"   # Corn futures
copper_symbol = "HG=F" # Copper futures

corn_df = fetch_data(corn_symbol)
copper_df = fetch_data(copper_symbol)

latest_corn = corn_df[corn_symbol].iloc[-1]
latest_copper = copper_df[copper_symbol].iloc[-1]

corn_trade = generate_trade_ideas(latest_corn)
copper_trade = generate_trade_ideas(latest_copper)

# --- Dashboard layout ---
st.title("📊 Swing Trading Dashboard")
st.markdown("Automatisk genererede trade-idéer og prisprognoser")

# --- Cards (Corn til højre, Copper til venstre) ---
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Copper (USD)", value=f"{latest_copper:.2f}", delta=f"{(copper_trade['Target']-latest_copper)/latest_copper*100:.2f}%")
    st.write("Trade-idé:", copper_trade)

with col2:
    st.metric(label="Corn (USD)", value=f"{latest_corn:.2f}", delta=f"{(corn_trade['Target']-latest_corn)/latest_corn*100:.2f}%")
    st.write("Trade-idé:", corn_trade)

# --- Historical + predicted graph ---
future_days = 30
copper_pred = create_lstm_prediction(copper_df[copper_symbol].values, future_days)
corn_pred = create_lstm_prediction(corn_df[corn_symbol].values, future_days)
future_dates = pd.date_range(start=copper_df['Date'].iloc[-1]+pd.Timedelta(days=1), periods=future_days)

fig = go.Figure()
fig.add_trace(go.Scatter(x=copper_df['Date'], y=copper_df[copper_symbol], mode='lines', name='Copper History'))
fig.add_trace(go.Scatter(x=future_dates, y=copper_pred, mode='lines', name='Copper Pred'))
fig.add_trace(go.Scatter(x=corn_df['Date'], y=corn_df[corn_symbol], mode='lines', name='Corn History'))
fig.add_trace(go.Scatter(x=future_dates, y=corn_pred, mode='lines', name='Corn Pred'))

st.plotly_chart(fig, use_container_width=True)

# --- Daily recommendation ---
st.subheader("📌 Dagens anbefaling")
st.write(f"Copper: {copper_trade['Type']} – Entry: {copper_trade['Entry']:.2f}, Target: {copper_trade['Target']:.2f}, Stop: {copper_trade['Stop']:.2f}")
st.write(f"Corn: {corn_trade['Type']} – Entry: {corn_trade['Entry']:.2f}, Target: {corn_trade['Target']:.2f}, Stop: {corn_trade['Stop']:.2f}")
