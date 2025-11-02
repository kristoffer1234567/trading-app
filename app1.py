# app_auto_fundamentals.py

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

# Forsøg at importere cot_reports, hvis installeret
try:
    import cot_reports as cot
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False

# === Hjælpefunktioner ===

def fetch_price_df(symbol, period="1y"):
    df = yf.Ticker(symbol).history(period=period)
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df.rename(columns={'Close': symbol}, inplace=True)
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

    # Hvis COT-signal findes, brug det (eksempel): hvis cot_signal > 0 → LONG, < 0 → SHORT
    if cot_signal is not None:
        if cot_signal > 0.2:
            reason = f"COT signal indikerer stor net LONG ({cot_signal:.2f})"
            return {"Type": "LONG",
                    "Entry": latest_price,
                    "Target": latest_price * 1.05,
                    "Stop": latest_price * 0.97,
                    "Reason": reason}
        if cot_signal < -0.2:
            reason = f"COT signal indikerer stor net SHORT ({cot_signal:.2f})"
            return {"Type": "SHORT",
                    "Entry": latest_price,
                    "Target": latest_price * 0.95,
                    "Stop": latest_price * 1.03,
                    "Reason": reason}

    # Teknisk signal:
    if rsi > 70 and sma_ratio is not None and sma_ratio > 1:
        reason = f"RSI {rsi:.1f} >70 og SMA10 > SMA30"
        return {"Type": "SHORT", "Entry": latest_price,
                "Target": latest_price * 0.95, "Stop": latest_price * 1.03,
                "Reason": reason}
    if rsi < 30 and sma_ratio is not None and sma_ratio < 1:
        reason = f"RSI {rsi:.1f} <30 og SMA10 < SMA30"
        return {"Type": "LONG", "Entry": latest_price,
                "Target": latest_price * 1.05, "Stop": latest_price * 0.97,
                "Reason": reason}

    return {"Type": "HOLD", "Entry": latest_price, "Target": latest_price, "Stop": latest_price,
            "Reason": "Ingen stærkt signal"}

def fetch_cot_signal(symbol):
    """
    Hent COT data via cot_reports, return et signal værdi (positiv = net long, negativ = net short)
    Hvis ikke muligt, return None.
    """
    if not COT_AVAILABLE:
        return None
    try:
        # Eksempel med cot_reports: hent seneste år for "legacy_futopt"
        df = cot.cot_all(cot_report_type='legacy_futopt')
        # Filtrer for ønsket symbol
        # Bemærk: symbol benævnelser i COT rapporter kan være anderledes
        # Simpel tilgang: find kolonne der matcher symbol delvis
        cols = [c for c in df.columns if symbol in c]
        if not cols:
            return None
        # Brug net position = long minus short for denne kolonne
        net = df[cols[0] + "_Net"].iloc[-1]  # antager kolonnenavn “_Net”
        # Normaliser signal
        return net / df[cols[0] + "_OpenInterest"].iloc[-1]
    except Exception as e:
        st.write("COT hentning fejlede:", e)
        return None

def fetch_psd_signal(commodity_code):
    """
    Hent USDA PSD / WASDE data via FAS OpenData API, return et "forsyningssignal".
    Hvis ikke muligt, return None.
    """
    try:
        api_base = "https://apps.fas.usda.gov/OpenData/api/psd/commodity"
        # commodity_code fx "0440000" for corn
        url = f"{api_base}/{commodity_code}/world/year/2024"
        resp = pd.read_json(url)  # Bemærk: dette kræver at API svarer JSON
        # Eksempel: find værdi af 'Ending Stocks' og 'Use'
        # Her laver vi en simpel ratio:
        if "Ending Stocks" in resp.columns and "Use" in resp.columns:
            ratio = resp["Ending Stocks"].iloc[0] / resp["Use"].iloc[0]
            # Høj ratio → overskudsperiode → bearish, lav ratio → bullish
            return - (ratio - 0.2)  # simpel transformation
        return None
    except Exception as e:
        # st.write("PSD hentning fejlede:", e)
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

    # Fetch fundamentale signaler
    cot_corn = fetch_cot_signal("Corn")
    cot_copper = fetch_cot_signal("Copper")
    # For PSD, vi bruger “commodity codes” – fx for corn “0440000”
    psd_corn = fetch_psd_signal("0440000")
    psd_copper = None  # Vi har ikke PSD for kobber

    corn_trade = generate_trade(latest_corn, rsi_corn, sma10_corn, sma30_corn, cot_signal=cot_corn, psd_signal=psd_corn)
    copper_trade = generate_trade(latest_copper, rsi_copper, sma10_copper, sma30_copper, cot_signal=cot_copper, psd_signal=psd_copper)

    # LSTM prediktioner
    try:
        model_corn, scaler_corn = train_lstm(corn_df[corn_sym].values, n_past=50, n_future=30, epochs=3)
        corn_pred = predict_lstm(model_corn, scaler_corn, corn_df[corn_sym].values[-50:], n_future=30)
    except Exception as e:
        corn_pred = None
    try:
        model_copper, scaler_copper = train_lstm(copper_df[copper_sym].values, n_past=50, n_future=30, epochs=3)
        copper_pred = predict_lstm(model_copper, scaler_copper, copper_df[copper_sym].values[-50:], n_future=30)
    except Exception as e:
        copper_pred = None

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
    if copper_pred is not None:
        future = pd.date_range(start=copper_df["Date"].iloc[-1] + timedelta(days=1), periods=len(copper_pred))
        fig.add_trace(go.Scatter(x=future, y=copper_pred, name="Copper prognose"))
    fig.add_trace(go.Scatter(x=corn_df["Date"], y=corn_df[corn_sym], name="Corn historik"))
    if corn_pred is not None:
        future2 = pd.date_range(start=corn_df["Date"].iloc[-1] + timedelta(days=1), periods=len(corn_pred))
        fig.add_trace(go.Scatter(x=future2, y=corn_pred, name="Corn prognose"))

    fig.add_trace(go.Scatter(x=corn_df["Date"], y=corn_df["SMA30"], name="Corn SMA30", line=dict(dash="dash")))
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
