import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import ta
from streamlit_autorefresh import st_autorefresh

# ------------------------
# 自动刷新
st_autorefresh(interval=10000, key="refresh_v6_2")

st.set_page_config(page_title="ETH AI Trading System v6.2", layout="wide")

symbol = "ETHUSDT"
interval = "5m"
limit = 300
url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

# ------------------------
# 数据加载
@st.cache_data(ttl=10)
def load_data():
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=range(12))
    df = df.iloc[:, 0:6]
    df.columns = ["time","open","high","low","close","volume"]
    df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
    df["time"] = pd.to_datetime(df["time"], unit='ms')
    df["date"] = df["time"].dt.date
    return df

df = load_data()

# ------------------------
# 指标计算
df["ema200"] = ta.trend.ema_indicator(df["close"], 200)
df["rsi"] = ta.momentum.rsi(df["close"], 14)
df["vol_ma"] = df["volume"].rolling(20).mean()
df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
df["high20"] = df["high"].rolling(20).max()
df["low20"] = df["low"].rolling(20).min()

# ------------------------
# 核心评分函数 - 完美排斥修复
def calc_score_v6_2(row, df):
    score = 0
    # 趋势
    trend_dir = 1 if row["close"] > row["ema200"] else -1
    score += 25*trend_dir
    # RSI
    if row["rsi"] > 55:
        score += 10
        rsi_dir = 1
    elif row["rsi"] < 45:
        score -= 10
        rsi_dir = -1
    else:
        rsi_dir = 0
    # 成交量
    if row["volume"] > row["vol_ma"]:
        score += 15
    # 突破
    if row["close"] > row["high20"]:
        score += 15
        brk_dir = 1
    elif row["close"] < row["low20"]:
        score -= 15
        brk_dir = -1
    else:
        brk_dir = 0
    # ATR方向加权
    recent_atr_mean = df["atr"].iloc[max(0,row.name-10):row.name].mean()
    if row["atr"] > recent_atr_mean:
        if trend_dir == brk_dir and brk_dir != 0:
            score += 10
        elif trend_dir != brk_dir and brk_dir != 0:
            score -= 5
    # 多维冲突扣分
    conflict_count = sum([trend_dir != brk_dir and brk_dir !=0, trend_dir != rsi_dir and rsi_dir !=0])
    score -= conflict_count*5
    return score

df["score"] = df.apply(lambda row: calc_score_v6_2(row, df), axis=1)

# ------------------------
# 动态阈值
long_threshold = max(15, np.percentile(df["score"].tail(50), 55))
short_threshold = min(-15, np.percentile(df["score"].tail(50), 45))

# ------------------------
# 信号生成
def generate_signal_v6_2(df):
    scores = df["score"].values
    signals = []
    for i, score in enumerate(scores):
        signal = "HOLD"
        weak_flag = False
        # 阈值判断
        if score >= long_threshold:
            signal = "LONG"
        elif score <= short_threshold:
            signal = "SHORT"
        else:
            weak_flag = True
        # 连续加权
        if i >= 2:
            recent = scores[i-2:i+1]
            if np.all(recent >= 0) and np.mean(recent) >= long_threshold*0.6:
                signal = "LONG"
            elif np.all(recent <= 0) and np.mean(recent) <= short_threshold*0.6:
                signal = "SHORT"
        # 标注弱信号
        if weak_flag and signal=="HOLD":
            signal = "WEAK"
        signals.append(signal)
    return signals

df["signal"] = generate_signal_v6_2(df)
last = df.iloc[-1]
confidence = min(abs(last["score"]),100)

# ------------------------
# 每日多空概率统计
daily_stats = df.groupby("date")["signal"].value_counts(normalize=True).unstack().fillna(0)*100

# 最近50根信号胜率与期望值
recent = df.tail(50)
signal_counts = recent["signal"].value_counts()
long_count = signal_counts.get("LONG",0)
short_count = signal_counts.get("SHORT",0)
total_trades = long_count + short_count
win_rate_est = (total_trades / 50)*100
avg_profit, avg_loss = 1, 0.7
expected_value = (win_rate_est/100)*avg_profit - ((100-win_rate_est)/100)*avg_loss

# ------------------------
# UI展示
st.title("ETH 5分钟 AI统一决策 v6.2 完美版")
st.subheader("市场状态")
c1,c2,c3,c4 = st.columns(4)
c1.metric("ETH价格", round(last["close"],2))
c2.metric("AI评分", round(last["score"],2))
c3.metric("信心指数", f"{confidence}%")
c4.metric("最近50根胜率(估)", f"{win_rate_est:.2f}%")

st.subheader("AI决策")
if last["signal"]=="LONG":
    st.success("🟢 做多信号")
elif last["signal"]=="SHORT":
    st.error("🔴 做空信号")
elif last["signal"]=="WEAK":
    st.info("🟡 弱信号 HOLD")
else:
    st.warning("⚪ HOLD 观望")

st.subheader("每日多空概率统计")
st.dataframe(daily_stats.tail(7))

# ------------------------
# K线图 + EMA + 成交量 + 信号标注
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K线"
))
fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], name="EMA200", line=dict(color='blue',width=2)))
fig.add_trace(go.Bar(x=df["time"], y=df["volume"], name="成交量", marker_color='lightgray', yaxis="y2"))
for sig_type,color,sym in [("LONG","green","triangle-up"),("SHORT","red","triangle-down"),("WEAK","orange","circle")]:
    mask = df["signal"]==sig_type
    fig.add_trace(go.Scatter(x=df["time"][mask], y=df["close"][mask], mode="markers",
                             marker=dict(symbol=sym,size=10,color=color), name=sig_type))
fig.update_layout(title="ETH 5分钟K线 + EMA200 + 成交量 + 信号标注",
                  xaxis_rangeslider_visible=False, yaxis2=dict(overlaying="y",side="right",showgrid=False))
st.plotly_chart(fig,use_container_width=True)

# RSI图
st.subheader("RSI 14")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color='orange',width=2)))
fig2.update_layout(yaxis_range=[0,100])
st.plotly_chart(fig2,use_container_width=True)

# 最近50根信号
st.subheader("最近50根K线信号")
st.dataframe(recent[["time","close","score","signal"]])
