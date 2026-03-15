import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# 自动刷新
st_autorefresh(interval=10000, key="refresh_v8_1")

st.set_page_config(page_title="ETH 5分钟交易系统 v8.1", layout="wide")

# ==================== 参数设置 (优化版) ====================
st.sidebar.header("⚙️ 参数设置")

st.sidebar.subheader("RSI阈值 (更严格)")
rsi_long = st.sidebar.slider("做多RSI阈值", 50, 70, 58, 1, help="RSI > 此值才有做多动量，建议58+")
rsi_short = st.sidebar.slider("做空RSI阈值", 30, 50, 42, 1, help="RSI < 此值才有做空动量，建议42-")

st.sidebar.subheader("成交量过滤")
vol_mult = st.sidebar.slider("成交量倍数", 1.0, 2.5, 1.5, 0.1, help="成交量必须 > 均量×倍数")

st.sidebar.subheader("波动率过滤")
min_volatility = st.sidebar.slider("最小波动率", 0.001, 0.005, 0.002, 0.0005, 
                                    help="ATR/价格 > 此值才交易，过滤低波动")

st.sidebar.subheader("止损止盈")
stop_atr_mult = st.sidebar.slider("止损ATR倍数", 1.0, 3.0, 2.0, 0.1)
take_atr_mult = st.sidebar.slider("止盈ATR倍数", 1.5, 5.0, 3.0, 0.1)

st.sidebar.subheader("突破确认")
require_close_confirm = st.sidebar.checkbox("要求收盘确认", True, help="收盘价必须突破，避免影线假突破")

# ==================== 数据获取 ====================
def generate_mock_data(n=300):
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    price = 2100.0
    times = pd.date_range(end=datetime.now(), periods=n, freq='5min')
    data = []
    
    for _ in range(n):
        change = np.random.normal(0, 0.003)  # 增加波动
        open_p = price
        close_p = price * (1 + change)
        vol = abs(np.random.normal(0, 0.008))
        high_p = max(open_p, close_p) * (1 + vol)
        low_p = min(open_p, close_p) * (1 - vol)
        volume = np.random.uniform(800, 3000)
        data.append([times[_], open_p, high_p, low_p, close_p, volume])
        price = close_p
    
    return pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])

def fetch_binance():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit=300"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if not data or len(data) < 50 or isinstance(data, dict):
            return None, None
        df = pd.DataFrame(data).iloc[:, 0:6]
        df.columns = ["time","open","high","low","close","volume"]
        df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
        df["time"] = pd.to_datetime(df["time"], unit='ms')
        return df, "Binance"
    except:
        return None, None

def fetch_okx():
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=5m&limit=300"
        resp = requests.get(url, timeout=10)
        data = resp.json().get('data', [])
        if not data or len(data) < 50:
            return None, None
        df = pd.DataFrame(data).iloc[:, 0:6]
        df.columns = ["time","open","high","low","close","volume"]
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["time"] = pd.to_datetime(df["time"].astype(float), unit='ms')
        df = df.sort_values('time').reset_index(drop=True).dropna()
        return df, "OKX" if len(df) >= 50 else None, None
    except:
        return None, None

def fetch_huobi():
    try:
        url = "https://api.huobi.pro/market/history/kline?symbol=ethusdt&period=5min&size=300"
        resp = requests.get(url, timeout=10)
        data = resp.json().get('data', [])
        if not data or len(data) < 50:
            return None, None
        df = pd.DataFrame(data)[['id','open','high','low','close','vol']]
        df.columns = ["time","open","high","low","close","volume"]
        df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
        df["time"] = pd.to_datetime(df["time"].astype(int), unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        return df, "Huobi"
    except:
        return None, None

@st.cache_data(ttl=10)
def load_data():
    for fetch_func in [fetch_binance, fetch_okx, fetch_huobi]:
        result = fetch_func()
        if result[0] is not None and len(result[0]) >= 50:
            return result[0], result[1]
    return generate_mock_data(), "模拟数据"

df, data_source = load_data()

if df is None or len(df) < 50:
    st.error("无法获取数据")
    st.stop()

# ==================== 核心指标计算 ====================
# EMA趋势
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

# RSI
delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# 成交量
df["vol_ma"] = df["volume"].rolling(20).mean()

# ATR
tr = pd.concat([df["high"]-df["low"], 
                abs(df["high"]-df["close"].shift(1)), 
                abs(df["low"]-df["close"].shift(1))], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean()

# 波动率
df["volatility"] = df["atr"] / df["close"]

# 支撑阻力
df["high20"] = df["high"].rolling(20).max()
df["low20"] = df["low"].rolling(20).min()

# ==================== 信号生成 (优化逻辑) ====================
def generate_signal(row):
    """
    信号触发顺序 (优化):
    1. 趋势过滤 - EMA50 vs EMA200
    2. 波动率过滤 - ATR/Price > 阈值
    3. 动量确认 - RSI
    4. 成交量确认 - 放量
    5. 突破触发 - 突破20周期高低点
    """
    signal = "HOLD"
    checks = {}
    
    # 1. 趋势过滤
    trend = "震荡"
    if row["ema50"] > row["ema200"]:
        trend = "多头"
    elif row["ema50"] < row["ema200"]:
        trend = "空头"
    checks["趋势"] = trend
    
    # 2. 波动率过滤
    volatility = row["volatility"] if pd.notna(row["volatility"]) else 0
    vol_ok = volatility > min_volatility
    checks["波动率"] = f"{volatility*100:.2f}% {'✓' if vol_ok else '✗'}"
    
    if not vol_ok:
        return signal, trend, checks, "波动率过低"
    
    # 3. RSI动量
    rsi = row["rsi"] if pd.notna(row["rsi"]) else 50
    checks["RSI"] = f"{rsi:.0f}"
    
    # 4. 成交量
    vol_ratio = row["volume"] / row["vol_ma"] if pd.notna(row["vol_ma"]) and row["vol_ma"] > 0 else 0
    vol_ok = vol_ratio > vol_mult
    checks["成交量"] = f"{vol_ratio:.1f}x {'✓' if vol_ok else '✗'}"
    
    # 5. 突破判断 (收盘确认)
    close = row["close"]
    if require_close_confirm:
        break_high = close > row["high20"]
        break_low = close < row["low20"]
    else:
        break_high = row["high"] > row["high20"]
        break_low = row["low"] < row["low20"]
    
    checks["突破高"] = "✓" if break_high else "✗"
    checks["突破低"] = "✓" if break_low else "✗"
    
    # ===== 做多判断 =====
    if trend == "多头":
        conditions = []
        
        # RSI动量确认
        if rsi > rsi_long:
            conditions.append("RSI✓")
        else:
            checks["RSI"] = f"{rsi:.0f}✗ (需>{rsi_long})"
        
        # 成交量确认
        if vol_ok:
            conditions.append("放量✓")
        
        # 突破确认
        if break_high:
            conditions.append("突破✓")
        
        # 至少满足2个条件 + 趋势
        if len(conditions) >= 2 and rsi > rsi_long and vol_ok:
            signal = "LONG"
    
    # ===== 做空判断 =====
    elif trend == "空头":
        conditions = []
        
        # RSI动量确认
        if rsi < rsi_short:
            conditions.append("RSI✓")
        else:
            checks["RSI"] = f"{rsi:.0f}✗ (需<{rsi_short})"
        
        # 成交量确认
        if vol_ok:
            conditions.append("放量✓")
        
        # 突破确认
        if break_low:
            conditions.append("跌破✓")
        
        # 至少满足2个条件 + 趋势
        if len(conditions) >= 2 and rsi < rsi_short and vol_ok:
            signal = "SHORT"
    
    block_reason = None if signal != "HOLD" else f"未满足条件"
    return signal, trend, checks, block_reason

# 应用信号
signals, trends, all_checks, block_reasons = [], [], [], []
for idx, row in df.iterrows():
    sig, trend, checks, reason = generate_signal(row)
    signals.append(sig)
    trends.append(trend)
    all_checks.append(checks)
    block_reasons.append(reason)

df["signal"] = signals
df["trend"] = trends
df["checks"] = all_checks

# ==================== 止损止盈 ====================
last = df.iloc[-1]
atr = last["atr"] if pd.notna(last["atr"]) else last["close"] * 0.01

if last["signal"] == "LONG":
    stop_loss = last["close"] - stop_atr_mult * atr
    take_profit = last["close"] + take_atr_mult * atr
    risk_reward = take_atr_mult / stop_atr_mult
elif last["signal"] == "SHORT":
    stop_loss = last["close"] + stop_atr_mult * atr
    take_profit = last["close"] - take_atr_mult * atr
    risk_reward = take_atr_mult / stop_atr_mult
else:
    stop_loss = take_profit = risk_reward = None

# ==================== 回测 ====================
def backtest(df, lookback=100, max_bars=30):
    recent = df.tail(lookback).copy()
    signals = recent[recent["signal"].isin(["LONG", "SHORT"])]
    
    if len(signals) == 0:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, 
                "signal_freq": 0, "avg_pnl": 0, "avg_win": 0, "avg_loss": 0}
    
    wins, losses = 0, 0
    win_pnls, loss_pnls = [], []
    
    for idx, row in signals.iterrows():
        pos = df.index.get_loc(idx)
        if pos + max_bars >= len(df):
            continue
        
        entry = row["close"]
        atr_val = row["atr"] if pd.notna(row["atr"]) else entry * 0.01
        
        if row["signal"] == "LONG":
            sl = entry - stop_atr_mult * atr_val
            tp = entry + take_atr_mult * atr_val
            for j in range(pos + 1, min(pos + max_bars + 1, len(df))):
                if df.iloc[j]["low"] <= sl:
                    loss_pnls.append((sl - entry) / entry * 100)
                    losses += 1
                    break
                if df.iloc[j]["high"] >= tp:
                    win_pnls.append((tp - entry) / entry * 100)
                    wins += 1
                    break
        else:
            sl = entry + stop_atr_mult * atr_val
            tp = entry - take_atr_mult * atr_val
            for j in range(pos + 1, min(pos + max_bars + 1, len(df))):
                if df.iloc[j]["high"] >= sl:
                    loss_pnls.append((entry - sl) / entry * 100)
                    losses += 1
                    break
                if df.iloc[j]["low"] <= tp:
                    win_pnls.append((entry - tp) / entry * 100)
                    wins += 1
                    break
    
    total = wins + losses
    return {
        "total": len(signals),
        "signal_freq": len(signals) / lookback * 100,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / total * 100) if total > 0 else 0,
        "avg_win": np.mean(win_pnls) if win_pnls else 0,
        "avg_loss": np.mean(loss_pnls) if loss_pnls else 0,
        "avg_pnl": np.mean(win_pnls + loss_pnls) if (win_pnls or loss_pnls) else 0
    }

bt = backtest(df, 100, 30)

# ==================== UI展示 ====================
st.title("📊 ETH 5分钟交易系统 v8.1 优化版")
st.markdown(f"**数据源:** {data_source} | **价格:** ${last['close']:.2f} | **时间:** {last['time'].strftime('%Y-%m-%d %H:%M')}")

# 核心指标
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ETH价格", f"${last['close']:.2f}")
c2.metric("趋势", df.iloc[-1]["trend"])
c3.metric("RSI", f"{last['rsi']:.0f}" if pd.notna(last['rsi']) else "N/A")
c4.metric("波动率", f"{last['volatility']*100:.2f}%")
c5.metric("信号频率", f"{bt['signal_freq']:.1f}%")

# 信号显示
st.subheader("🎯 交易信号")
if last["signal"] == "LONG":
    st.success("🟢 **做多信号**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", f"${stop_loss:.2f}", f"-{(last['close']-stop_loss)/last['close']*100:.2f}%")
    col3.metric("止盈", f"${take_profit:.2f}", f"+{(take_profit-last['close'])/last['close']*100:.2f}%")
    col4.metric("盈亏比", f"{risk_reward:.1f}:1")
elif last["signal"] == "SHORT":
    st.error("🔴 **做空信号**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", f"${stop_loss:.2f}", f"+{(stop_loss-last['close'])/last['close']*100:.2f}%")
    col3.metric("止盈", f"${take_profit:.2f}", f"-{(last['close']-take_profit)/last['close']*100:.2f}%")
    col4.metric("盈亏比", f"{risk_reward:.1f}:1")
else:
    st.warning("⚪ **观望**")
    last_checks = last["checks"]
    st.markdown(f"""
    **条件检查:**
    - 趋势: {last_checks.get('趋势', 'N/A')}
    - 波动率: {last_checks.get('波动率', 'N/A')}
    - RSI: {last_checks.get('RSI', 'N/A')}
    - 成交量: {last_checks.get('成交量', 'N/A')}
    - 突破高: {last_checks.get('突破高', 'N/A')}
    - 突破低: {last_checks.get('突破低', 'N/A')}
    """)

# 条件检查表
st.subheader("📋 条件检查")
checks = last["checks"]
cols = st.columns(6)

trend_ok = checks.get("趋势", "震荡") != "震荡"
cols[0].metric("趋势", checks.get("趋势", "N/A"), delta="✓" if trend_ok else "✗")

vol_ok = "✓" in checks.get("波动率", "")
cols[1].metric("波动率", checks.get("波动率", "N/A"), delta="✓" if vol_ok else "✗")

rsi = last["rsi"] if pd.notna(last["rsi"]) else 50
rsi_ok = (rsi > rsi_long) if checks.get("趋势") == "多头" else (rsi < rsi_short) if checks.get("趋势") == "空头" else False
cols[2].metric("RSI动量", checks.get("RSI", "N/A"), delta="✓" if rsi_ok else "✗")

vol_ratio = last["volume"] / last["vol_ma"] if pd.notna(last["vol_ma"]) and last["vol_ma"] > 0 else 0
vol_str_ok = vol_ratio > vol_mult
cols[3].metric("成交量", f"{vol_ratio:.1f}x", delta="✓" if vol_str_ok else "✗")

cols[4].metric("突破高", checks.get("突破高", "N/A"))
cols[5].metric("突破低", checks.get("突破低", "N/A"))

# K线图
st.subheader("📈 价格走势")
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    name="K线", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
))

fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", line=dict(color='orange', width=1.5)))
fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], name="EMA200", line=dict(color='blue', width=2)))

fig.add_hline(y=last["high20"], line_dash="dash", line_color="red", opacity=0.6, annotation_text="阻力")
fig.add_hline(y=last["low20"], line_dash="dash", line_color="green", opacity=0.6, annotation_text="支撑")

for sig, color, sym in [("LONG", "green", "triangle-up"), ("SHORT", "red", "triangle-down")]:
    mask = df["signal"] == sig
    if mask.any():
        fig.add_trace(go.Scatter(
            x=df["time"][mask], y=df["close"][mask], mode="markers",
            marker=dict(symbol=sym, size=12, color=color), name=sig
        ))

fig.update_layout(
    title="ETH 5分钟 K线 + EMA趋势",
    xaxis_rangeslider_visible=False, height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# 成交量
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=df["time"], y=df["volume"], name="成交量", marker_color='lightgray'))
fig_vol.add_trace(go.Scatter(x=df["time"], y=df["vol_ma"] * vol_mult, name=f"均量×{vol_mult}", line=dict(color='red', dash='dash')))
fig_vol.add_trace(go.Scatter(x=df["time"], y=df["vol_ma"], name="均量20", line=dict(color='blue')))
fig_vol.update_layout(title="成交量", height=120, showlegend=True)
st.plotly_chart(fig_vol, use_container_width=True)

# RSI
st.subheader("📉 RSI 14")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color='purple')))
fig_rsi.add_hline(y=rsi_long, line_dash="dash", line_color="green", annotation_text=f"做多{rsi_long}")
fig_rsi.add_hline(y=rsi_short, line_dash="dash", line_color="red", annotation_text=f"做空{rsi_short}")
fig_rsi.update_layout(yaxis_range=[0, 100], height=150)
st.plotly_chart(fig_rsi, use_container_width=True)

# 回测统计
st.subheader("📊 回测统计")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("交易次数", bt["total"])
c2.metric("盈利/亏损", f"{bt['wins']}/{bt['losses']}")
c3.metric("胜率", f"{bt['win_rate']:.1f}%")
c4.metric("平均盈利", f"{bt['avg_win']:.2f}%")
c5.metric("平均亏损", f"{bt['avg_loss']:.2f}%")

# 最近信号
st.subheader("📋 最近信号")
recent_sig = df[df["signal"].isin(["LONG", "SHORT"])].tail(10)[["time", "close", "signal", "trend"]]
if len(recent_sig) > 0:
    recent_sig = recent_sig.copy()
    recent_sig["time"] = recent_sig["time"].dt.strftime('%Y-%m-%d %H:%M')
    recent_sig["close"] = recent_sig["close"].round(2)
    st.dataframe(recent_sig, use_container_width=True)
else:
    st.info("暂无历史信号")

# 策略说明
st.sidebar.markdown("---")
st.sidebar.subheader("📖 策略说明")
st.sidebar.info(f"""
**信号触发顺序:**
1. 趋势过滤 (EMA50 vs EMA200)
2. 波动率过滤 (>{min_volatility*100:.1f}%)
3. RSI动量确认 (>{rsi_long} / <{rsi_short})
4. 成交量确认 (>{vol_mult}x均量)
5. 突破触发 (20周期高低点)

**风控:**
- 止损 = {stop_atr_mult} × ATR
- 止盈 = {take_atr_mult} × ATR
- 盈亏比 = 1:{take_atr_mult/stop_atr_mult:.1f}

**当前胜率:** {bt['win_rate']:.1f}%
""")
