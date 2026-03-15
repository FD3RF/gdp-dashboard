import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# 自动刷新
st_autorefresh(interval=10000, key="refresh_v9_0")

st.set_page_config(page_title="ETH 5分钟量化交易系统 v9.0", layout="wide")

# ==================== 固定参数 (不可调) ====================
RSI_LONG = 55      # 做多RSI阈值
RSI_SHORT = 45     # 做空RSI阈值
VOL_MULT = 1.5     # 成交量倍数
STOP_ATR = 1.8     # 止损ATR倍数
TAKE_ATR = 2.8     # 止盈ATR倍数
BREAKOUT = 20      # 突破周期

# ==================== 数据获取 ====================
def fetch_data():
    """获取数据，优先Binance"""
    # Binance
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit=300"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data and len(data) >= 50 and not isinstance(data, dict):
            df = pd.DataFrame(data).iloc[:, 0:6]
            df.columns = ["time","open","high","low","close","volume"]
            df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
            df["time"] = pd.to_datetime(df["time"], unit='ms')
            return df, "Binance"
    except:
        pass
    
    # OKX
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=5m&limit=300"
        resp = requests.get(url, timeout=10)
        data = resp.json().get('data', [])
        if data and len(data) >= 50:
            df = pd.DataFrame(data).iloc[:, 0:6]
            df.columns = ["time","open","high","low","close","volume"]
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df["time"] = pd.to_datetime(df["time"].astype(float), unit='ms')
            df = df.sort_values('time').reset_index(drop=True).dropna()
            return df, "OKX"
    except:
        pass
    
    # 模拟数据
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    price = 2100.0
    times = pd.date_range(end=datetime.now(), periods=300, freq='5min')
    data = []
    for _ in range(300):
        change = np.random.normal(0, 0.003)
        o, c = price, price * (1 + change)
        h = max(o, c) * (1 + abs(np.random.normal(0, 0.005)))
        l = min(o, c) * (1 - abs(np.random.normal(0, 0.005)))
        data.append([times[_], o, h, l, c, np.random.uniform(800, 3000)])
        price = c
    return pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume']), "模拟数据"

@st.cache_data(ttl=10)
def load_data():
    return fetch_data()

df, data_source = load_data()

if df is None or len(df) < 50:
    st.error("无法获取数据")
    st.stop()

# ==================== 指标计算 ====================
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
df["avg_vol"] = df["volatility"].rolling(50).mean()

# 突破
df["high_n"] = df["high"].rolling(BREAKOUT).max()
df["low_n"] = df["low"].rolling(BREAKOUT).min()

# ==================== 量化信号计算 ====================
def calc_signal(row):
    """
    精确量化信号计算
    输出: (signal, sl, tp, details)
    
    信号类型:
    - LONG: 强做多 (全条件满足)
    - WEAK_LONG: 弱做多 (成交量不足)
    - SHORT: 强做空
    - WEAK_SHORT: 弱做空
    - HOLD: 观望
    """
    signal = "HOLD"
    sl = tp = 0
    details = {}
    
    # 提取数据
    close = row["close"]
    atr = row["atr"] if pd.notna(row["atr"]) else close * 0.01
    rsi = row["rsi"] if pd.notna(row["rsi"]) else 50
    vol_ratio = row["volume"] / row["vol_ma"] if pd.notna(row["vol_ma"]) and row["vol_ma"] > 0 else 0
    vol = row["volatility"] if pd.notna(row["volatility"]) else 0
    avg_vol = row["avg_vol"] if pd.notna(row["avg_vol"]) else vol
    
    # 1. 趋势判断
    if row["ema50"] > row["ema200"]:
        trend = "多头"
        trend_score = 1
    elif row["ema50"] < row["ema200"]:
        trend = "空头"
        trend_score = -1
    else:
        trend = "震荡"
        trend_score = 0
    details["趋势"] = trend
    
    # 2. 波动率过滤 (动态阈值: max(0.12%, avg_vol*0.8))
    vol_threshold = max(0.0012, avg_vol * 0.8)
    vol_ok = vol >= vol_threshold
    details["波动率"] = f"{vol*100:.2f}% {'✓' if vol_ok else '✗'}"
    
    # 3. RSI动量
    if trend_score > 0:
        rsi_ok = rsi >= RSI_LONG
        details["RSI"] = f"{rsi:.0f} {'✓' if rsi_ok else '✗'} (>={RSI_LONG})"
    elif trend_score < 0:
        rsi_ok = rsi <= RSI_SHORT
        details["RSI"] = f"{rsi:.0f} {'✓' if rsi_ok else '✗'} (<={RSI_SHORT})"
    else:
        rsi_ok = False
        details["RSI"] = f"{rsi:.0f} (震荡)"
    
    # 4. 成交量
    vol_ok = vol_ratio >= VOL_MULT
    details["成交量"] = f"{vol_ratio:.2f}x {'✓' if vol_ok else '✗'} (>={VOL_MULT}x)"
    
    # 5. 突破确认
    break_high = close > row["high_n"]
    break_low = close < row["low_n"]
    details["突破高"] = "✓" if break_high else "✗"
    details["突破低"] = "✓" if break_low else "✗"
    
    # ===== 做多判断 =====
    if trend_score > 0 and vol_ok and rsi_ok:
        # 强做多: 全条件满足
        signal = "LONG"
        sl = close - STOP_ATR * atr
        tp = close + TAKE_ATR * atr
        details["信号"] = "强做多"
        details["止损"] = f"${sl:.2f}"
        details["止盈"] = f"${tp:.2f}"
        details["盈亏比"] = f"1:{TAKE_ATR/STOP_ATR:.1f}"
        
    elif trend_score > 0 and rsi_ok and not vol_ok:
        # 弱做多: RSI满足但缩量
        signal = "WEAK_LONG"
        sl = close - STOP_ATR * atr
        tp = close + TAKE_ATR * atr
        details["信号"] = "弱做多(缩量)"
        details["止损"] = f"${sl:.2f}"
        details["止盈"] = f"${tp:.2f}"
        details["建议"] = "轻仓试探"
    
    # ===== 做空判断 =====
    elif trend_score < 0 and vol_ok and rsi_ok:
        # 强做空
        signal = "SHORT"
        sl = close + STOP_ATR * atr
        tp = close - TAKE_ATR * atr
        details["信号"] = "强做空"
        details["止损"] = f"${sl:.2f}"
        details["止盈"] = f"${tp:.2f}"
        details["盈亏比"] = f"1:{TAKE_ATR/STOP_ATR:.1f}"
        
    elif trend_score < 0 and rsi_ok and not vol_ok:
        # 弱做空
        signal = "WEAK_SHORT"
        sl = close + STOP_ATR * atr
        tp = close - TAKE_ATR * atr
        details["信号"] = "弱做空(缩量)"
        details["止损"] = f"${sl:.2f}"
        details["止盈"] = f"${tp:.2f}"
        details["建议"] = "轻仓试探"
    
    else:
        details["信号"] = "观望"
        # 分析原因
        reasons = []
        if trend_score == 0:
            reasons.append("趋势震荡")
        if not vol_ok:
            reasons.append("波动率低")
        if not rsi_ok:
            reasons.append("RSI未达标")
        if not vol_ok:
            reasons.append("缩量")
        details["原因"] = ", ".join(reasons) if reasons else "条件不满足"
    
    return signal, sl, tp, details

# 应用信号
signals, sls, tps, all_details = [], [], [], []
for idx, row in df.iterrows():
    sig, sl, tp, details = calc_signal(row)
    signals.append(sig)
    sls.append(sl)
    tps.append(tp)
    all_details.append(details)

df["signal"] = signals
df["sl"] = sls
df["tp"] = tps

# ==================== 回测 ====================
def backtest(df, lookback=100, max_bars=30):
    recent = df.tail(lookback).copy()
    trades = recent[recent["signal"].isin(["LONG", "SHORT"])]
    
    if len(trades) == 0:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, 
                "freq": 0, "avg_win": 0, "avg_loss": 0, "expectancy": 0}
    
    wins, losses = 0, 0
    win_pnls, loss_pnls = [], []
    
    for idx, row in trades.iterrows():
        pos = df.index.get_loc(idx)
        if pos + max_bars >= len(df):
            continue
        
        entry = row["close"]
        sl, tp = row["sl"], row["tp"]
        
        if row["signal"] == "LONG":
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
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    win_rate = wins / total * 100 if total > 0 else 0
    
    # 期望值 = 胜率 × 平均盈利 - (1-胜率) × 平均亏损
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * abs(avg_loss)) if total > 0 else 0
    
    return {
        "total": len(trades),
        "freq": len(trades) / lookback * 100,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy
    }

bt = backtest(df, 100, 30)

# ==================== UI展示 ====================
last = df.iloc[-1]
last_details = all_details[-1]
atr = last["atr"] if pd.notna(last["atr"]) else last["close"] * 0.01

st.title("📊 ETH 5分钟量化交易系统 v9.0")
st.markdown(f"**{data_source}** | **${last['close']:.2f}** | **{last['time'].strftime('%Y-%m-%d %H:%M')}**")

# 核心指标
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("价格", f"${last['close']:.2f}")
c2.metric("趋势", last_details["趋势"])
c3.metric("RSI", f"{last['rsi']:.0f}")
c4.metric("波动率", f"{last['volatility']*100:.2f}%")
c5.metric("信号频率", f"{bt['freq']:.1f}%")

# ===== 信号显示 =====
st.subheader("🎯 交易信号")

if last["signal"] == "LONG":
    st.success(f"🟢 **强做多信号**")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", last_details.get("止损", "N/A"))
    col3.metric("止盈", last_details.get("止盈", "N/A"))
    col4.metric("盈亏比", last_details.get("盈亏比", "N/A"))
    col5.metric("距离止损", f"{(last['close']-last['sl'])/last['close']*100:.2f}%")
    
elif last["signal"] == "SHORT":
    st.error(f"🔴 **强做空信号**")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", last_details.get("止损", "N/A"))
    col3.metric("止盈", last_details.get("止盈", "N/A"))
    col4.metric("盈亏比", last_details.get("盈亏比", "N/A"))
    col5.metric("距离止损", f"{(last['sl']-last['close'])/last['close']*100:.2f}%")
    
elif last["signal"] == "WEAK_LONG":
    st.info(f"🟡 **弱做多信号** - 缩量确认")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", last_details.get("止损", "N/A"))
    col3.metric("止盈", last_details.get("止盈", "N/A"))
    col4.metric("建议", last_details.get("建议", "轻仓"))
    
elif last["signal"] == "WEAK_SHORT":
    st.info(f"🟡 **弱做空信号** - 缩量确认")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", last_details.get("止损", "N/A"))
    col3.metric("止盈", last_details.get("止盈", "N/A"))
    col4.metric("建议", last_details.get("建议", "轻仓"))
    
else:
    st.warning(f"⚪ **观望**")
    st.markdown(f"**原因:** {last_details.get('原因', '条件不满足')}")

# ===== 条件检查 =====
st.subheader("📋 条件检查")
cols = st.columns(6)
cols[0].metric("趋势", last_details["趋势"])
cols[1].metric("波动率", last_details["波动率"])
cols[2].metric("RSI", last_details["RSI"])
cols[3].metric("成交量", last_details["成交量"])
cols[4].metric("突破高", last_details["突破高"])
cols[5].metric("突破低", last_details["突破低"])

# ===== K线图 =====
st.subheader("📈 价格走势")
fig = go.Figure()

# K线
fig.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
    increasing_line_color='#26a69a', decreasing_line_color='#ef5350', name="K线"
))

# EMA
fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", line=dict(color='orange', width=1)))
fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], name="EMA200", line=dict(color='blue', width=2)))

# 支撑阻力
fig.add_hline(y=last["high_n"], line_dash="dash", line_color="red", opacity=0.5, annotation_text="阻力")
fig.add_hline(y=last["low_n"], line_dash="dash", line_color="green", opacity=0.5, annotation_text="支撑")

# 信号标记
colors = {"LONG": ("green", 14), "SHORT": ("red", 14), 
          "WEAK_LONG": ("lightgreen", 8), "WEAK_SHORT": ("lightcoral", 8)}
for sig, (color, size) in colors.items():
    mask = df["signal"] == sig
    if mask.any():
        sym = "triangle-up" if "LONG" in sig else "triangle-down"
        fig.add_trace(go.Scatter(
            x=df["time"][mask], y=df["close"][mask], mode="markers",
            marker=dict(symbol=sym, size=size, color=color), name=sig
        ))

# 止损止盈线 (当前信号)
if last["signal"] in ["LONG", "WEAK_LONG"]:
    fig.add_hline(y=last["sl"], line_color="red", line_width=2, annotation_text="止损")
    fig.add_hline(y=last["tp"], line_color="green", line_width=2, annotation_text="止盈")
elif last["signal"] in ["SHORT", "WEAK_SHORT"]:
    fig.add_hline(y=last["sl"], line_color="red", line_width=2, annotation_text="止损")
    fig.add_hline(y=last["tp"], line_color="green", line_width=2, annotation_text="止盈")

fig.update_layout(xaxis_rangeslider_visible=False, height=450,
                  legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
st.plotly_chart(fig, use_container_width=True)

# ===== 成交量 =====
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=df["time"], y=df["volume"], marker_color='lightgray', name="成交量"))
fig_vol.add_trace(go.Scatter(x=df["time"], y=df["vol_ma"], line=dict(color='blue'), name="均量20"))
fig_vol.add_trace(go.Scatter(x=df["time"], y=df["vol_ma"]*VOL_MULT, line=dict(color='red', dash='dash'), name=f"阈值{VOL_MULT}x"))
fig_vol.update_layout(height=120, showlegend=False)
st.plotly_chart(fig_vol, use_container_width=True)

# ===== RSI =====
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["rsi"], line=dict(color='purple'), name="RSI"))
fig_rsi.add_hline(y=RSI_LONG, line_dash="dash", line_color="green", annotation_text=f"多{RSI_LONG}")
fig_rsi.add_hline(y=RSI_SHORT, line_dash="dash", line_color="red", annotation_text=f"空{RSI_SHORT}")
fig_rsi.update_layout(yaxis_range=[0, 100], height=120)
st.plotly_chart(fig_rsi, use_container_width=True)

# ===== 回测统计 =====
st.subheader("📊 回测统计")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("交易次数", bt["total"])
c2.metric("胜/负", f"{bt['wins']}/{bt['losses']}")
c3.metric("胜率", f"{bt['win_rate']:.1f}%")
c4.metric("平均盈利", f"{bt['avg_win']:.2f}%")
c5.metric("平均亏损", f"{bt['avg_loss']:.2f}%")
c6.metric("期望值", f"{bt['expectancy']:.3f}%")

# ===== 最近信号 =====
st.subheader("📋 最近信号")
cols = ["time", "close", "signal", "sl", "tp"]
recent = df[df["signal"].isin(["LONG", "SHORT", "WEAK_LONG", "WEAK_SHORT"])].tail(10)[cols]
if len(recent) > 0:
    recent = recent.copy()
    recent["time"] = recent["time"].dt.strftime('%m-%d %H:%M')
    recent["close"] = recent["close"].round(2)
    recent["sl"] = recent["sl"].round(2)
    recent["tp"] = recent["tp"].round(2)
    st.dataframe(recent, use_container_width=True, hide_index=True)
else:
    st.info("暂无历史信号")

# ===== 策略参数 =====
st.sidebar.header("📖 策略参数")
st.sidebar.info(f"""
**固定参数:**
- RSI阈值: {RSI_LONG}/{RSI_SHORT}
- 成交量: ≥{VOL_MULT}x均量
- 突破周期: {BREAKOUT}
- 止损: {STOP_ATR} ATR
- 止盈: {TAKE_ATR} ATR
- 盈亏比: 1:{TAKE_ATR/STOP_ATR:.1f}

**信号类型:**
- 强做多/空: 全条件满足
- 弱做多/空: 缩量确认
- 观望: 条件不满足

**当前胜率:** {bt['win_rate']:.1f}%
**期望值:** {bt['expectancy']:.3f}%
""")
