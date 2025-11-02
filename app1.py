import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ta

# === Hjælpefunktioner ===
def fetch_price_df(symbol, period="1y"):
    df = yf.Ticker(symbol).history(period=period).reset_index()
    df = df[['Date', 'Close']].rename(columns={'Close': symbol})
    return df

def compute_indicators(df, symbol):
    df["SMA10"] = ta.trend.sma_indicator(df[symbol], window=10)
    df["SMA30"] = ta.trend.sma_indicator(df[symbol], window=30)
    df["RSI14"] = ta.momentum.rsi(df[symbol], window=14)
    return df

def train_lstm(prices, n_past=30, n_future=30, epochs=5):
    scaler = MinMaxScaler(feature_range=(0,1))
    ps = prices.reshape(-1,1)
    scaled = scaler.fit_transform(ps)
    X, y = [], []
    for i in range(n_past, len(scaled) - n_future + 1):
        X.append(scaled[i - n_past:i, 0])
        y.append(scaled[i : i + n_future, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_past,1)))
    model.add(Dense(n_future))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model, scaler

def predict_lstm(model, scaler, recent_prices, n_future=30):
    arr = recent_prices.reshape(-1,1)
    scaled = scaler.transform(arr)
    seq = scaled[-len(arr):].reshape((1, len(arr), 1))
    pred_scaled = model.predict(seq)[0]
    pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).reshape(-1)
    return pred

def generate_trade(latest_price, rsi, sma10, sma30, cot_signal=None, psd_signal=None):
    reason = ""
    if rsi is None or sma10 is None or sma30 is None:
        return {"Type": "HOLD", "Entry": latest_price, "Target": latest_price, "Stop": latest_price,
                "Reason": "Ingen tilstrækkelige indikatorer"}
    sma_ratio = sma10 / sma30 if sma30 != 0 else None

    if cot_signal is not None:
        if cot_signal > 0.2:
            reason = f"COT signal indikerer stor net LONG ({cot_signal:.2f})"
            return {"Type": "LONG", "Entry": latest_price, "Target": latest_price*1.05, "Stop": latest_price*0.97, "Reason": reason}
        if cot_signal < -0.2:
            reason = f"COT signal indikerer stor net SHORT ({cot_signal:.2f})"
            return {"Type": "SHORT", "Entry": latest_price, "Target": latest_price*0.95, "Stop": latest_price*1.03, "Reason": reason}

    if rsi > 70 and sma_ratio is not None and sma_ratio > 1:
        reason = f"RSI {rsi:.1f} >70 og SMA10 > SMA30"
        return {"Type": "SHORT", "Entry": latest_price, "Target": latest_price*0.95, "Stop": latest_price*1.03, "Reason": reason}
    if rsi < 30 and sma_ratio is not None and sma_ratio < 1:
        reason = f"RSI {rsi:.1f} <30 og SMA10 < SMA30"
        return {"Type": "LONG", "Entry": latest_price, "Target": latest_price*1.05, "Stop": latest_price*0.97, "Reason": reason}

    return {"Type": "HOLD", "Entry": latest_price, "Target": latest_price, "Stop": latest_price, "Reason": "Ingen stærkt signal"}

# === API input fra bruger ===
st.sidebar.title("Indtast API nøgler / indstillinger")
cot_api_key = st.sidebar.text_input("COT API nøgle", "")
psd_api_key = st.sidebar.text_input("PSD/USDA API nøgle", "")

def fetch_cot_signal(symbol, api_key=None):
    if not api_key:
        return None  # fallback
    try:
        import cot_reports as cot
        df = cot.cot_all(cot_report_type='legacy_futopt')
        cols = [c for c in df.columns if symbol in c]
        if not cols:
            return None
        net = df[cols[0] + "_Net"].iloc[-1]
        return net / df[cols[0] + "_OpenInterest"].iloc[-1]
    except Exception:
        return None

def fetch_psd_signal(commodity_code, api_key=None):
    if not api_key:
        return None
    try:
        url = f"https://apps.fas.usda.gov/OpenData/api/psd/commodity/{commodity_code}/world/year/2024"
        resp = pd.read_json(url)
        if "Ending Stocks" in resp.columns and "Use" in resp.columns:
            ratio = resp["Ending Stocks"].iloc[0] / resp["Use"].iloc[0]
            return -(ratio - 0.2)
        return None
    except Exception:
        return None

def main():
    corn_sym = "ZC=F"
    copper_sym = "HG=F"

    corn_df = fetch_price_df(corn_sym)
    copper_df = fetch_price_df(copper_sym)
    corn_df = compute_indicators(corn_df, corn_sym)
    copper_df = compute_indicators(copper_df, copper_sym)

    latest_corn = corn_df[corn_sym].iloc[-1]
    latest_copper = copper_df[copper_sym].iloc[-1]
    rsi_corn = corn_df["RSI14"].iloc[-1]
    sma10_corn = corn_df["SMA10"].iloc[-1]
    sma30_corn = corn_df["SMA30"].iloc[-1]
    rsi_copper = copper_df["RSI14"].iloc[-1]
    sma10_copper = copper_df["SMA10"].iloc[-1]
    sma30_copper = copper_df["SMA30"].iloc[-1]

    cot_corn = fetch_cot_signal("Corn", cot_api_key)
    cot_copper = fetch_cot_signal("Copper", cot_api_key)
    psd_corn = fetch_psd_signal("0440000", psd_api_key)
    psd_copper = None

    corn_trade = generate_trade(latest_corn, rsi_corn, sma10_corn, sma30_corn, cot_signal=cot_corn, psd_signal=psd_corn)
    copper_trade = generate_trade(latest_copper, rsi_copper, sma10_copper, sma30_copper, cot_signal=cot_copper, psd_signal=psd_copper)

    st.title("Swing Trading Dashboard – Automatisk fundamentaler")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Copper (USD)", f"{latest_copper:.2f}", delta=f"{(copper_trade['Target']-latest_copper)/latest_copper*100:.2f}%")
        st.write("Trade‑idé:", copper_trade)
    with c2:
        st.metric("Corn (USD)", f"{latest_corn:.2f}", delta=f"{(corn_trade['Target']-latest_corn)/latest_corn*100:.2f}%")
        st.write("Trade‑idé:", corn_trade)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=copper_df["Date"], y=copper_df[copper_sym], name="Copper historik"))
    fig.add_trace(go.Scatter(x=corn_df["Date"], y=corn_df[corn_sym], name="Corn historik"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anbefalinger og begrundelser")
    st.write("### Copper")
    st.write(f"Signal: **{copper_trade['Type']}**")
    st.write("Begrundelse:", copper_trade.get("Reason"))
    st.write("COT signal:", cot_copper)
    st.write("### Corn")
    st.write(f"Signal: **{corn_trade['Type']}**")
    st.write("Begrundelse:", corn_trade.get("Reason"))
    st.write("COT signal:", cot_corn, "PSD signal:", psd_corn)

if __name__ == "__main__":
    main()
