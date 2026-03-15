import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# 自动刷新
st_autorefresh(interval=10000, key="refresh_v9_2")

st.set_page_config(page_title="ETH 5分钟量化交易系统 v9.2", layout="wide")

# ==================== 核心信号函数 ====================
def get_eth_signal(kline, atr, avg_volume, params=None):
    """
    精确量化信号计算 v9.2
    
    关键改进: 突破时忽略波动率过滤
    逻辑顺序: 趋势 → 突破 → 波动率 → RSI → 成交量
    
    核心逻辑:
    - 如果突破成立，波动率过滤失效
    - 突破本身就是波动释放
    """
    if params is None:
        params = {
            "rsi_long": 55,
            "rsi_short": 45,
            "volume_mult": 1.5,
            "volatility_thresh": 0.0015,
            "stop_loss_mult": 1.8,
            "take_profit_mult": 2.8
        }
    
    close = kline["close"]
    ema50 = kline["ema50"]
    ema200 = kline["ema200"]
    rsi = kline["rsi"]
    volume = kline["volume"]
    high20 = kline.get("high20", None)
    low20 = kline.get("low20", None)
    
    # 1. 趋势判断
    if ema50 > ema200:
        trend = "多头"
        trend_dir = 1
    elif ema50 < ema200:
        trend = "空头"
        trend_dir = -1
    else:
        trend = "横盘"
        trend_dir = 0
    
    # 2. 突破判断 (提前计算)
    breakout_long = high20 is not None and close > high20
    breakout_short = low20 is not None and close < low20
    
    # 3. 波动率判断
    volatility = atr / close if close > 0 else 0
    vol_ok = volatility >= params["volatility_thresh"]
    
    # ★ 关键: 如果突破成立，波动率过滤失效
    vol_filter_ok_long = vol_ok or breakout_long
    vol_filter_ok_short = vol_ok or breakout_short
    
    # 4. RSI动量判断
    rsi_ok_long = rsi >= params["rsi_long"]
    rsi_ok_short = rsi <= params["rsi_short"]
    
    # 5. 成交量判断
    vol_ratio = volume / avg_volume if avg_volume > 0 else 0
    volume_ok = vol_ratio >= params["volume_mult"]
    
    # 6. 信号计算
    signal = "观望"
    confidence = 0.0
    stop_loss = 0
    take_profit = 0
    reasons = []
    
    # ===== 做多判断 =====
    if trend_dir > 0:
        # 强做多: 突破成立 (波动率过滤自动通过)
        if rsi_ok_long and vol_filter_ok_long and breakout_long and volume_ok:
            signal = "强做多"
            confidence = 1.0
            stop_loss = close - params["stop_loss_mult"] * atr
            take_profit = close + params["take_profit_mult"] * atr
            reasons.append("突破+" if not vol_ok else "突破")
        # 强做多: 放量趋势 (无突破但波动率足够)
        elif rsi_ok_long and vol_ok and volume_ok:
            signal = "强做多"
            confidence = 0.9
            stop_loss = close - params["stop_loss_mult"] * atr
            take_profit = close + params["take_profit_mult"] * atr
            reasons.append("放量趋势")
        # 弱做多: 突破但缩量
        elif rsi_ok_long and vol_filter_ok_long and breakout_long:
            signal = "弱做多"
            confidence = 0.8
            stop_loss = close - params["stop_loss_mult"] * atr
            take_profit = close + params["take_profit_mult"] * atr
            reasons.append("突破缩量")
        # 弱做多: 无突破缩量
        elif rsi_ok_long and vol_ok:
            signal = "弱做多"
            confidence = 0.6
            stop_loss = close - params["stop_loss_mult"] * atr
            take_profit = close + params["take_profit_mult"] * atr
            reasons.append("缩量")
        else:
            if not rsi_ok_long:
                reasons.append(f"RSI={rsi:.0f}<55")
            if not vol_filter_ok_long:
                reasons.append(f"波动率低{volatility*100:.2f}%")
            if not volume_ok:
                reasons.append(f"缩量{vol_ratio:.1f}x")
            if not breakout_long:
                reasons.append("未突破")
    
    # ===== 做空判断 =====
    elif trend_dir < 0:
        # 强做空: 突破成立
        if rsi_ok_short and vol_filter_ok_short and breakout_short and volume_ok:
            signal = "强做空"
            confidence = 1.0
            stop_loss = close + params["stop_loss_mult"] * atr
            take_profit = close - params["take_profit_mult"] * atr
            reasons.append("突破-" if not vol_ok else "突破")
        # 强做空: 放量趋势
        elif rsi_ok_short and vol_ok and volume_ok:
            signal = "强做空"
            confidence = 0.9
            stop_loss = close + params["stop_loss_mult"] * atr
            take_profit = close - params["take_profit_mult"] * atr
            reasons.append("放量趋势")
        # 弱做空: 突破但缩量
        elif rsi_ok_short and vol_filter_ok_short and breakout_short:
            signal = "弱做空"
            confidence = 0.8
            stop_loss = close + params["stop_loss_mult"] * atr
            take_profit = close - params["take_profit_mult"] * atr
            reasons.append("突破缩量")
        # 弱做空: 无突破缩量
        elif rsi_ok_short and vol_ok:
            signal = "弱做空"
            confidence = 0.6
            stop_loss = close + params["stop_loss_mult"] * atr
            take_profit = close - params["take_profit_mult"] * atr
            reasons.append("缩量")
        else:
            if not rsi_ok_short:
                reasons.append(f"RSI={rsi:.0f}>45")
            if not vol_filter_ok_short:
                reasons.append(f"波动率低{volatility*100:.2f}%")
            if not volume_ok:
                reasons.append(f"缩量{vol_ratio:.1f}x")
            if not breakout_short:
                reasons.append("未突破")
    
    else:
        reasons.append("横盘趋势")
    
    return {
        "signal": signal,
        "trend": trend,
        "entry_price": close,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": confidence,
        "volatility": volatility,
        "vol_ratio": vol_ratio,
        "breakout_long": breakout_long,
        "breakout_short": breakout_short,
        "reasons": reasons
    }

# ==================== 数据获取 ====================
def fetch_data():
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
    
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    price, times = 2100.0, pd.date_range(end=datetime.now(), periods=300, freq='5min')
    data = []
    for _ in range(300):
        change = np.random.normal(0, 0.003)
        o, c = price, price * (1 + change)
        h, l = max(o,c) * (1 + abs(np.random.normal(0,0.005))), min(o,c) * (1 - abs(np.random.normal(0,0.005)))
        data.append([times[_], o, h, l, c, np.random.uniform(800, 3000)])
        price = c
    return pd.DataFrame(data, columns=['time','open','high','low','close','volume']), "模拟数据"

@st.cache_data(ttl=10)
def load_data():
    return fetch_data()

df, data_source = load_data()

if df is None or len(df) < 50:
    st.error("无法获取数据")
    st.stop()

# ==================== 指标计算 ====================
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

df["vol_ma"] = df["volume"].rolling(20).mean()

tr = pd.concat([df["high"]-df["low"], 
                abs(df["high"]-df["close"].shift(1)), 
                abs(df["low"]-df["close"].shift(1))], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean()

df["high20"] = df["high"].rolling(20).max()
df["low20"] = df["low"].rolling(20).min()

# ==================== 应用信号函数 ====================
params = {
    "rsi_long": 55,
    "rsi_short": 45,
    "volume_mult": 1.5,
    "volatility_thresh": 0.0015,
    "stop_loss_mult": 1.8,
    "take_profit_mult": 2.8
}

signals, trends, sls, tps, confs, vols, vol_ratios, reasons_list = [], [], [], [], [], [], [], []

for idx, row in df.iterrows():
    kline = {
        "close": row["close"],
        "ema50": row["ema50"],
        "ema200": row["ema200"],
        "rsi": row["rsi"] if pd.notna(row["rsi"]) else 50,
        "volume": row["volume"],
        "high20": row["high20"] if pd.notna(row["high20"]) else None,
        "low20": row["low20"] if pd.notna(row["low20"]) else None
    }
    atr = row["atr"] if pd.notna(row["atr"]) else row["close"] * 0.01
    avg_vol = row["vol_ma"] if pd.notna(row["vol_ma"]) else row["volume"]
    
    result = get_eth_signal(kline, atr, avg_vol, params)
    
    signals.append(result["signal"])
    trends.append(result["trend"])
    sls.append(result["stop_loss"])
    tps.append(result["take_profit"])
    confs.append(result["confidence"])
    vols.append(result["volatility"])
    vol_ratios.append(result["vol_ratio"])
    reasons_list.append(result["reasons"])

df["signal"] = signals
df["trend"] = trends
df["sl"] = sls
df["tp"] = tps
df["confidence"] = confs

# ==================== 回测 ====================
def backtest(df, lookback=100, max_bars=30):
    recent = df.tail(lookback).copy()
    trades = recent[recent["signal"].isin(["强做多", "强做空", "弱做多", "弱做空"])]
    
    if len(trades) == 0:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, 
                "freq": 0, "avg_win": 0, "avg_loss": 0, "expectancy": 0}
    
    wins, losses = 0, 0
    win_pnls, loss_pnls = [], []
    
    for idx, row in trades.iterrows():
        pos = df.index.get_loc(idx)
        if pos + max_bars >= len(df):
            continue
        
        entry, sl, tp = row["close"], row["sl"], row["tp"]
        
        if "多" in row["signal"]:
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
    expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * abs(avg_loss)) if total > 0 else 0
    
    return {
        "total": len(trades), "freq": len(trades) / lookback * 100,
        "wins": wins, "losses": losses, "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss, "expectancy": expectancy
    }

bt = backtest(df, 100, 30)

# ==================== UI ====================
# ★ 使用已完成K线 (iloc[-2])，避免最后一根未收盘K线的volume=0问题
last = df.iloc[-2]
last_reasons = reasons_list[-2]
last_vol = vol_ratios[-2]
last_volatility = vols[-2]
last_conf = confs[-2]

st.title("📊 ETH 5分钟量化交易系统 v9.2")
st.markdown(f"**{data_source}** | **${last['close']:.2f}** | **{last['time'].strftime('%Y-%m-%d %H:%M')}**")

# 指标
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("价格", f"${last['close']:.2f}")
c2.metric("趋势", last["trend"])
c3.metric("RSI", f"{last['rsi']:.0f}")
c4.metric("波动率", f"{last_volatility*100:.2f}%")
c5.metric("信号频率", f"{bt['freq']:.1f}%")

# 信号
st.subheader("🎯 交易信号")
sig = last["signal"]

if sig == "强做多":
    st.success(f"🟢 **强做多信号** | 信心度: {last_conf*100:.0f}%")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", f"${last['sl']:.2f}")
    col3.metric("止盈", f"${last['tp']:.2f}")
    col4.metric("盈亏比", "1:1.6")
    st.info(f"**原因:** {', '.join(last_reasons)}")
elif sig == "强做空":
    st.error(f"🔴 **强做空信号** | 信心度: {last_conf*100:.0f}%")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", f"${last['sl']:.2f}")
    col3.metric("止盈", f"${last['tp']:.2f}")
    col4.metric("盈亏比", "1:1.6")
    st.info(f"**原因:** {', '.join(last_reasons)}")
elif sig == "弱做多":
    st.info(f"🟡 **弱做多信号** | 信心度: {last_conf*100:.0f}%")
    col1, col2, col3 = st.columns(3)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", f"${last['sl']:.2f}")
    col3.metric("止盈", f"${last['tp']:.2f}")
    st.warning(f"**原因:** {', '.join(last_reasons)}")
elif sig == "弱做空":
    st.info(f"🟡 **弱做空信号** | 信心度: {last_conf*100:.0f}%")
    col1, col2, col3 = st.columns(3)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", f"${last['sl']:.2f}")
    col3.metric("止盈", f"${last['tp']:.2f}")
    st.warning(f"**原因:** {', '.join(last_reasons)}")
else:
    st.warning("⚪ **观望**")
    st.markdown(f"**原因:** {', '.join(last_reasons) if last_reasons else '条件不满足'}")

# 条件检查
st.subheader("📋 条件检查")
cols = st.columns(6)
cols[0].metric("趋势", last["trend"])
cols[1].metric("波动率", f"{last_volatility*100:.2f}% {'✓' if last_volatility >= 0.0015 else '✗'}")
cols[2].metric("RSI", f"{last['rsi']:.0f}")
cols[3].metric("成交量", f"{last_vol:.2f}x {'✓' if last_vol >= 1.5 else '✗'}")
cols[4].metric("突破高", "✓" if last["close"] > last["high20"] else "✗")
cols[5].metric("突破低", "✓" if last["close"] < last["low20"] else "✗")

# K线图
st.subheader("📈 价格走势")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                              increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], line=dict(color='orange',width=1), name="EMA50"))
fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], line=dict(color='blue',width=2), name="EMA200"))
fig.add_hline(y=last["high20"], line_dash="dash", line_color="red", opacity=0.5, annotation_text="阻力")
fig.add_hline(y=last["low20"], line_dash="dash", line_color="green", opacity=0.5, annotation_text="支撑")

if sig in ["强做多", "弱做多", "强做空", "弱做空"]:
    fig.add_hline(y=last["sl"], line_color="red", line_width=2, annotation_text="止损")
    fig.add_hline(y=last["tp"], line_color="green", line_width=2, annotation_text="止盈")

for s, color, sz in [("强做多","green",14),("强做空","red",14),("弱做多","lightgreen",8),("弱做空","lightcoral",8)]:
    mask = df["signal"]==s
    if mask.any():
        sym = "triangle-up" if "多" in s else "triangle-down"
        fig.add_trace(go.Scatter(x=df["time"][mask], y=df["close"][mask], mode="markers",
                                 marker=dict(symbol=sym,size=sz,color=color), name=s))

fig.update_layout(xaxis_rangeslider_visible=False, height=400, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# 成交量
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=df["time"], y=df["volume"], marker_color='lightgray'))
fig_vol.add_trace(go.Scatter(x=df["time"], y=df["vol_ma"], line=dict(color='blue'), name="均量"))
fig_vol.add_trace(go.Scatter(x=df["time"], y=df["vol_ma"]*1.5, line=dict(color='red',dash='dash'), name="阈值"))
fig_vol.update_layout(height=100, showlegend=False)
st.plotly_chart(fig_vol, use_container_width=True)

# RSI
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df["time"], y=df["rsi"], line=dict(color='purple')))
fig_rsi.add_hline(y=55, line_dash="dash", line_color="green")
fig_rsi.add_hline(y=45, line_dash="dash", line_color="red")
fig_rsi.update_layout(yaxis_range=[0,100], height=100)
st.plotly_chart(fig_rsi, use_container_width=True)

# 回测
st.subheader("📊 回测统计")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("交易次数", bt["total"])
c2.metric("胜/负", f"{bt['wins']}/{bt['losses']}")
c3.metric("胜率", f"{bt['win_rate']:.1f}%")
c4.metric("平均盈利", f"{bt['avg_win']:.2f}%")
c5.metric("平均亏损", f"{bt['avg_loss']:.2f}%")
c6.metric("期望值", f"{bt['expectancy']:.3f}%")

# 参数
st.sidebar.header("📖 固定参数")
st.sidebar.info(f"""
**RSI:** 55/45
**成交量:** ≥1.5x均量
**波动率:** ≥0.15%
**止损:** 1.8 ATR
**止盈:** 2.8 ATR
**盈亏比:** 1:1.6

★ **关键改进:**
突破时忽略波动率过滤

**胜率:** {bt['win_rate']:.1f}%
**期望值:** {bt['expectancy']:.3f}%
""")
