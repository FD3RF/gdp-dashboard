import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta

# 自动刷新
st_autorefresh(interval=10000, key="refresh_v6_3")

st.set_page_config(page_title="ETH AI Trading System v6.3", layout="wide")

# 生成模拟数据
def generate_mock_data(n=300):
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    price = 2000.0
    times = pd.date_range(end=datetime.now(), periods=n, freq='5min')
    opens, highs, lows, closes, volumes = [], [], [], [], []
    
    for i in range(n):
        change = np.random.normal(0, 0.002)
        open_p = price
        close_p = price * (1 + change)
        vol = abs(np.random.normal(0, 0.005))
        high_p = max(open_p, close_p) * (1 + vol)
        low_p = min(open_p, close_p) * (1 - vol)
        volume = 1000 * (1 + np.random.exponential(0.5))
        opens.append(open_p)
        highs.append(high_p)
        lows.append(low_p)
        closes.append(close_p)
        volumes.append(volume)
        price = close_p
    
    df = pd.DataFrame({
        'time': times, 'open': opens, 'high': highs, 
        'low': lows, 'close': closes, 'volume': volumes
    })
    df['date'] = df['time'].dt.date
    return df

# 加载数据
@st.cache_data(ttl=10)
def load_data():
    url = "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit=300"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if not data or len(data) < 50:
            return None
        df = pd.DataFrame(data, columns=range(12)).iloc[:, 0:6]
        df.columns = ["time","open","high","low","close","volume"]
        df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
        df["time"] = pd.to_datetime(df["time"], unit='ms')
        df["date"] = df["time"].dt.date
        return df
    except:
        return None

df = load_data()
data_source = "实时数据 (币安)"

if df is None or len(df) < 50:
    df = generate_mock_data(300)
    data_source = "模拟数据"

# 指标计算
df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

df["vol_ma"] = df["volume"].rolling(20).mean()
df["high20"] = df["high"].rolling(20).max()
df["low20"] = df["low"].rolling(20).min()

tr = pd.concat([
    df["high"] - df["low"],
    abs(df["high"] - df["close"].shift(1)),
    abs(df["low"] - df["close"].shift(1))
], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean()

# 评分函数
def calc_score(row, df):
    score = 0
    trend_dir = 1 if row["close"] > row["ema200"] else -1
    score += 25 * trend_dir
    
    rsi_v = row["rsi"] if pd.notna(row["rsi"]) else 50
    if rsi_v > 55:
        score += 10
        rsi_dir = 1
    elif rsi_v < 45:
        score -= 10
        rsi_dir = -1
    else:
        rsi_dir = 0
    
    if row["volume"] > row["vol_ma"]:
        score += 15
    
    if row["close"] > row["high20"]:
        score += 15
        brk_dir = 1
    elif row["close"] < row["low20"]:
        score -= 15
        brk_dir = -1
    else:
        brk_dir = 0
    
    atr_v = row["atr"] if pd.notna(row["atr"]) else 0
    start_idx = max(0, row.name - 10)
    atr_mean = df["atr"].iloc[start_idx:row.name].mean() if row.name > 0 else atr_v
    
    if atr_v > atr_mean and trend_dir == brk_dir and brk_dir != 0:
        score += 10
    if trend_dir != brk_dir and brk_dir != 0:
        score -= 5
    if trend_dir != rsi_dir and rsi_dir != 0:
        score -= 5
    
    return score

df["score"] = df.apply(lambda r: calc_score(r, df), axis=1)

long_thr = max(15, np.nanpercentile(df["score"].tail(50), 55))
short_thr = min(-15, np.nanpercentile(df["score"].tail(50), 45))

# 信号生成
signals = []
for i, s in enumerate(df["score"].values):
    sig = "HOLD"
    if pd.isna(s):
        signals.append(sig)
        continue
    if s >= long_thr:
        sig = "LONG"
    elif s <= short_thr:
        sig = "SHORT"
    if i >= 2:
        rec = df["score"].values[i-2:i+1]
        rec = rec[~np.isnan(rec)]
        if len(rec) == 3:
            if np.all(rec >= 0) and np.mean(rec) >= long_thr * 0.6:
                sig = "LONG"
            if np.all(rec <= 0) and np.mean(rec) <= short_thr * 0.6:
                sig = "SHORT"
    signals.append(sig)

df["signal"] = signals

last = df.iloc[-1]
conf = min(abs(last["score"]) if pd.notna(last["score"]) else 0, 100)
daily_stats = df.groupby("date")["signal"].value_counts(normalize=True).unstack().fillna(0) * 100
recent = df.tail(50)
win_rate = (recent["signal"].isin(["LONG","SHORT"]).sum() / 50) * 100

# UI
st.title("ETH 5分钟 AI统一决策 v6.3")
st.markdown(f"**数据源:** {data_source} | **最新价格:** ${last['close']:.2f} | **时间:** {last['time'].strftime('%Y-%m-%d %H:%M')}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("ETH价格", f"${last['close']:.2f}")
c2.metric("AI评分", round(last["score"], 2) if pd.notna(last["score"]) else 0)
c3.metric("信心指数", f"{conf:.0f}%")
c4.metric("信号频率", f"{win_rate:.1f}%")

st.subheader("AI决策")
if last["signal"] == "LONG":
    st.success("🟢 做多信号")
elif last["signal"] == "SHORT":
    st.error("🔴 做空信号")
else:
    st.warning("⚪ 观望")

# 指标状态
st.subheader("指标状态")
c1, c2, c3, c4 = st.columns(4)
c1.metric("趋势", "多头" if last["close"] > last["ema200"] else "空头")
c2.metric("RSI", f"{last['rsi']:.1f}" if pd.notna(last['rsi']) else "N/A")
c3.metric("成交量", "放量" if last["volume"] > last["vol_ma"] else "缩量")
c4.metric("ATR", f"${last['atr']:.2f}" if pd.notna(last['atr']) else "N/A")

# K线图
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"], 
    low=df["low"], close=df["close"], name="K线"
))
fig.add_trace(go.Scatter(
    x=df["time"], y=df["ema200"], name="EMA200", 
    line=dict(color='blue', width=2)
))
fig.add_trace(go.Bar(
    x=df["time"], y=df["volume"], name="成交量", 
    marker_color='lightgray', yaxis="y2"
))

for sig_type, color, sym in [("LONG", "green", "triangle-up"), ("SHORT", "red", "triangle-down")]:
    mask = df["signal"] == sig_type
    if mask.any():
        fig.add_trace(go.Scatter(
            x=df["time"][mask], y=df["close"][mask], mode="markers",
            marker=dict(symbol=sym, size=12, color=color), name=sig_type
        ))

fig.update_layout(
    title="ETH 5分钟K线 + EMA200 + 成交量",
    xaxis_rangeslider_visible=False,
    yaxis2=dict(overlaying="y", side="right", showgrid=False),
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# RSI图
st.subheader("RSI 14")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color='orange', width=2)))
fig2.add_hline(y=70, line_dash="dash", line_color="red")
fig2.add_hline(y=30, line_dash="dash", line_color="green")
fig2.update_layout(yaxis_range=[0, 100], height=250)
st.plotly_chart(fig2, use_container_width=True)

# 每日统计
st.subheader("每日多空概率统计")
st.dataframe(daily_stats.tail(7))

# 最近信号
st.subheader("最近50根K线信号")
st.dataframe(recent[["time", "close", "score", "signal"]])
