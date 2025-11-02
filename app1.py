# 1️⃣ Installer nødvendige pakker
# pip install yfinance tensorflow scikit-learn pandas matplotlib streamlit

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st

# 2️⃣ Streamlit titel
st.title("Copper & Corn Swing Trading Dashboard")
st.write("Automatisk LSTM-prisforudsigelse og portefølje/rotationsplan")

# 3️⃣ Trading-parametre
capital = 2160
gearing = 2
risk_per_trade = 0.02
time_horizon_days = 60
look_back = 20
stop_loss_pct = 0.05
take_profit_mult = 2
symbols = ['HG=F','ZC=F']

# 4️⃣ Hent live data
@st.cache_data
def fetch_data(symbols):
    data = {}
    for sym in symbols:
        df = yf.download(sym, period='5y', interval='1d')
        data[sym] = df[['Close']].rename(columns={'Close': sym})
    df_all = pd.concat(data.values(), axis=1).dropna()
    return df_all

df_all = fetch_data(symbols)

# 5️⃣ Feature engineering
for sym in symbols:
    df_all[f'{sym}_pct'] = df_all[sym].pct_change()
    df_all[f'{sym}_ma_5'] = df_all[sym].rolling(5).mean()
    df_all[f'{sym}_ma_10'] = df_all[sym].rolling(10).mean()
df_all.dropna(inplace=True)

# 6️⃣ Skaler data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_all)

# 7️⃣ LSTM dataset funktion
def create_dataset(data, target_idx, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)

X, y_copper = create_dataset(scaled_data, 0, look_back)
_, y_corn = create_dataset(scaled_data, 1, look_back)
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train_copper, y_test_copper = y_copper[:split], y_copper[split:]
y_train_corn, y_test_corn = y_corn[:split], y_corn[split:]

# 8️⃣ LSTM-model funktion
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(60, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(60))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 9️⃣ Træn modeller
with st.spinner("Træner LSTM-modeller…"):
    model_copper = build_lstm((X_train.shape[1], X_train.shape[2]))
    model_copper.fit(X_train, y_train_copper, epochs=25, batch_size=32, verbose=0)
    model_corn = build_lstm((X_train.shape[1], X_train.shape[2]))
    model_corn.fit(X_train, y_train_corn, epochs=25, batch_size=32, verbose=0)

# 🔟 Forudsig fremtid
def predict_future(model, last_seq, days, scaler, target_idx):
    preds_scaled = []
    seq = last_seq.copy()
    for _ in range(days):
        pred = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0][0]
        preds_scaled.append(pred)
        new_row = seq[-1].copy()
        new_row[target_idx] = pred
        seq = np.vstack((seq[1:], new_row))
    preds = np.array([scaler.inverse_transform(np.array([[s]+[0]*(scaled_data.shape[1]-1)]))[0,target_idx] for s in preds_scaled])
    return preds

last_seq = scaled_data[-look_back:]
pred_copper = predict_future(model_copper, last_seq, time_horizon_days, scaler, 0)
pred_corn = predict_future(model_corn, last_seq, time_horizon_days, scaler, 1)

# 1️⃣1️⃣ Automatisk portefølje
future_dates = pd.date_range(df_all.index[-1]+pd.Timedelta(days=1), periods=time_horizon_days)
df_portfolio = pd.DataFrame(index=future_dates, columns=['Symbol','Signal','Price','Contracts','StopLoss','TakeProfit','ExpectedReturn','CumulativeP&L'])
last_prices = {'Copper': float(df_all['HG=F'].iloc[-1]), 'Corn': float(df_all['ZC=F'].iloc[-1])}
cumulative_pnl = 0

for i in range(time_horizon_days):
    copper_ret = (pred_copper[i]-last_prices['Copper'])/last_prices['Copper']
    corn_ret = (pred_corn[i]-last_prices['Corn'])/last_prices['Corn']

    # Fix: konverter til float
    copper_ret_val = float(copper_ret)
    corn_ret_val = float(corn_ret)

    if copper_ret_val > corn_ret_val:
        symbol = 'Copper'
        price = float(pred_copper[i])
        signal = 'LONG' if price > last_prices['Copper'] else 'SHORT'
        last_prices['Copper'] = price
        expected_return = copper_ret_val
    else:
        symbol = 'Corn'
        price = float(pred_corn[i])
        signal = 'LONG' if price > last_prices['Corn'] else 'SHORT'
        last_prices['Corn'] = price
        expected_return = corn_ret_val

    contracts = max(1,int((capital*risk_per_trade)/(price*stop_loss_pct)*gearing))
    pnl = expected_return*contracts*price*gearing
    cumulative_pnl += pnl
    df_portfolio.iloc[i] = [symbol, signal, price, contracts, stop_loss_pct, stop_loss_pct*take_profit_mult, expected_return, cumulative_pnl]

# 1️⃣2️⃣ Vis portefølje i Streamlit
st.subheader("Swing Trading Portefølje & Rotationsplan")
st.dataframe(df_portfolio)

# 1️⃣3️⃣ Visualisering
st.subheader("Prisforudsigelser")
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(df_all.index, df_all['HG=F'], label='Copper Actual', color='blue')
ax.plot(df_all.index, df_all['ZC=F'], label='Corn Actual', color='green')
ax.plot(future_dates, pred_copper, '--', label='Copper Predicted', color='red')
ax.plot(future_dates, pred_corn, '--', label='Corn Predicted', color='orange')
ax.set_title('Copper & Corn Prices with 1–3 Month Predictions')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

st.subheader("Simuleret Portefølje Cumulative P&L")
fig2, ax2 = plt.subplots(figsize=(14,5))
ax2.plot(df_portfolio.index, df_portfolio['CumulativeP&L'], color='purple', label='Cumulative P&L')
ax2.set_title('Simulated Portfolio P&L Over Time')
ax2.set_xlabel('Date')
ax2.set_ylabel('USD')
ax2.legend()
st.pyplot(fig2)
