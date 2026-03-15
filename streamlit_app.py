"""
ETH 5分钟量化交易系统 v9.7 - 优化版UI
简化页面布局，更清晰易读
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# 自动刷新
st_autorefresh(interval=10000, key="refresh_v9_7")

st.set_page_config(page_title="ETH 5分钟量化交易系统", layout="wide", initial_sidebar_state="collapsed")

# ==================== 核心信号函数 ====================
def get_eth_signal(kline, atr, avg_volume, params=None):
    if params is None:
        params = {"rsi_long": 55, "rsi_short": 45, "volume_mult": 1.5, 
                  "volatility_thresh": 0.0015, "stop_loss_mult": 1.8, "take_profit_mult": 2.8}
    
    close = kline["close"]
    ema50 = kline["ema50"]
    ema200 = kline["ema200"]
    rsi = kline["rsi"]
    volume = kline["volume"]
    high20 = kline.get("high20", None)
    low20 = kline.get("low20", None)
    
    # 趋势判断
    trend = "多头" if ema50 > ema200 else "空头" if ema50 < ema200 else "横盘"
    trend_dir = 1 if ema50 > ema200 else -1 if ema50 < ema200 else 0
    
    # 突破判断
    breakout_long = high20 is not None and close > high20
    breakout_short = low20 is not None and close < low20
    
    # 波动率
    volatility = atr / close if close > 0 else 0
    vol_ok = volatility >= params["volatility_thresh"]
    vol_filter_ok_long = vol_ok or breakout_long
    vol_filter_ok_short = vol_ok or breakout_short
    
    # RSI & 成交量
    rsi_ok_long = rsi >= params["rsi_long"]
    rsi_ok_short = rsi <= params["rsi_short"]
    vol_ratio = volume / avg_volume if avg_volume > 0 else 0
    volume_ok = vol_ratio >= params["volume_mult"]
    
    # 信号计算
    signal, confidence, stop_loss, take_profit = "观望", 0.0, 0, 0
    reasons = []
    
    if trend_dir > 0:  # 做多
        if rsi_ok_long and vol_filter_ok_long and breakout_long and volume_ok:
            signal, confidence = "强做多", 1.0
            stop_loss = close - params["stop_loss_mult"] * atr
            take_profit = close + params["take_profit_mult"] * atr
            reasons.append("突破+")
        elif rsi_ok_long and vol_ok and volume_ok:
            signal, confidence = "强做多", 0.9
            stop_loss = close - params["stop_loss_mult"] * atr
            take_profit = close + params["take_profit_mult"] * atr
            reasons.append("放量趋势")
        elif rsi_ok_long and vol_filter_ok_long and breakout_long and vol_ratio >= 0.8:
            signal, confidence = "弱做多", 0.8
            stop_loss = close - params["stop_loss_mult"] * atr
            take_profit = close + params["take_profit_mult"] * atr
            reasons.append("突破缩量")
        elif rsi_ok_long and vol_ok and vol_ratio >= 0.8:
            signal, confidence = "弱做多", 0.6
            stop_loss = close - params["stop_loss_mult"] * atr
            take_profit = close + params["take_profit_mult"] * atr
            reasons.append("缩量趋势")
        else:
            if not rsi_ok_long: reasons.append(f"RSI={rsi:.0f}<55")
            if not vol_filter_ok_long: reasons.append(f"波动率低")
            if not volume_ok: reasons.append(f"缩量{vol_ratio:.1f}x")
            if not breakout_long: reasons.append("未突破")
    elif trend_dir < 0:  # 做空
        if rsi_ok_short and vol_filter_ok_short and breakout_short and volume_ok:
            signal, confidence = "强做空", 1.0
            stop_loss = close + params["stop_loss_mult"] * atr
            take_profit = close - params["take_profit_mult"] * atr
            reasons.append("突破-")
        elif rsi_ok_short and vol_ok and volume_ok:
            signal, confidence = "强做空", 0.9
            stop_loss = close + params["stop_loss_mult"] * atr
            take_profit = close - params["take_profit_mult"] * atr
            reasons.append("放量趋势")
        elif rsi_ok_short and vol_filter_ok_short and breakout_short and vol_ratio >= 0.8:
            signal, confidence = "弱做空", 0.8
            stop_loss = close + params["stop_loss_mult"] * atr
            take_profit = close - params["take_profit_mult"] * atr
            reasons.append("突破缩量")
        elif rsi_ok_short and vol_ok and vol_ratio >= 0.8:
            signal, confidence = "弱做空", 0.6
            stop_loss = close + params["stop_loss_mult"] * atr
            take_profit = close - params["take_profit_mult"] * atr
            reasons.append("缩量趋势")
        else:
            if not rsi_ok_short: reasons.append(f"RSI={rsi:.0f}>45")
            if not vol_filter_ok_short: reasons.append(f"波动率低")
            if not volume_ok: reasons.append(f"缩量{vol_ratio:.1f}x")
            if not breakout_short: reasons.append("未突破")
    else:
        reasons.append("横盘趋势")
    
    # 突破概率
    breakout_prob = 0
    if trend_dir > 0:
        prob = 0
        if rsi >= 80: prob += 30
        elif rsi >= 70: prob += 20
        elif rsi >= 60: prob += 10
        if vol_ratio >= 3.0: prob += 35
        elif vol_ratio >= 2.0: prob += 25
        elif vol_ratio >= 1.5: prob += 15
        if volatility < 0.001: prob += 35
        elif volatility < 0.0015: prob += 25
        elif volatility < 0.002: prob += 15
        breakout_prob = min(prob, 95)
    elif trend_dir < 0:
        prob = 0
        if rsi <= 20: prob += 30
        elif rsi <= 30: prob += 20
        elif rsi <= 40: prob += 10
        if vol_ratio >= 3.0: prob += 35
        elif vol_ratio >= 2.0: prob += 25
        elif vol_ratio >= 1.5: prob += 15
        if volatility < 0.001: prob += 35
        elif volatility < 0.0015: prob += 25
        elif volatility < 0.002: prob += 15
        breakout_prob = min(prob, 95)
    
    # 仓位计算
    position_size, position_reason = 0, "观望"
    if signal in ["强做多", "强做空"]:
        if breakout_prob >= 70: position_size, position_reason = 45, "高概率适度仓45%"
        elif breakout_prob >= 60: position_size, position_reason = 70, "中高概率70%"
        elif breakout_prob >= 50: position_size, position_reason = 95, "最佳概率满仓95%"
        else: position_size, position_reason = 50, "低概率50%"
    elif signal in ["弱做多", "弱做空"]:
        if breakout_prob >= 70: position_size, position_reason = 40, "弱高概率40%"
        elif breakout_prob >= 60: position_size, position_reason = 60, "弱中概率60%"
        elif breakout_prob >= 50: position_size, position_reason = 80, "弱最佳80%"
        else: position_size, position_reason = 0, "弱低概率观望"
    else:
        if breakout_prob >= 70: position_size, position_reason = 20, "高概率试探20%"
        elif breakout_prob >= 50: position_size, position_reason = 10, "中概率试探10%"
    
    if volatility < 0.001: position_size = int(position_size * 0.5)
    elif volatility < 0.0015: position_size = int(position_size * 0.7)
    
    return {
        "signal": signal, "trend": trend, "entry_price": close, "stop_loss": stop_loss,
        "take_profit": take_profit, "confidence": confidence, "volatility": volatility,
        "vol_ratio": vol_ratio, "breakout_long": breakout_long, "breakout_short": breakout_short,
        "breakout_prob": breakout_prob, "position_size": position_size, "position_reason": position_reason,
        "reasons": reasons
    }

# ==================== 数据获取 ====================
def get_data(limit=500):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit={limit}"
        r = requests.get(url, timeout=10)
        data = r.json()
        df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","qvol","trades","tb","tq","ignore"])
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open","high","low","close","volume"]: df[col] = pd.to_numeric(df[col])
        return df.sort_values("time").reset_index(drop=True), "Binance"
    except:
        try:
            url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=5m&limit={limit}"
            r = requests.get(url, timeout=10)
            result = r.json()
            if result.get("code") == "0" and result.get("data"):
                df = pd.DataFrame(result["data"], columns=["time","open","high","low","close","volume","vc","vq","conf"])
                df["time"] = pd.to_datetime(df["time"])
                for col in ["open","high","low","close","volume"]: df[col] = pd.to_numeric(df[col])
                return df.sort_values("time").reset_index(drop=True), "OKX"
        except: pass
    return None, None

# ==================== 指标计算 ====================
def calc_indicators(df):
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain/loss))
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"] - df["close"].shift())
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["high20"] = df["high"].rolling(20).max()
    df["low20"] = df["low"].rolling(20).min()
    df["high15"] = df["high"].rolling(15).max()
    df["low15"] = df["low"].rolling(15).min()
    df["breakout_up"] = df["close"] > df["high15"].shift(1)
    return df

# ==================== 获取数据 ====================
df, data_source = get_data(500)
if df is None or len(df) < 200:
    st.error("无法获取足够数据，请稍后重试")
    st.stop()

df = calc_indicators(df)
df = df.dropna().reset_index(drop=True)

if len(df) < 50:
    st.error("数据不足，请稍后重试")
    st.stop()
params = {"rsi_long": 55, "rsi_short": 45, "volume_mult": 1.5, "volatility_thresh": 0.0015, "stop_loss_mult": 1.8, "take_profit_mult": 2.8}

# 应用信号
signals, trends, sls, tps, confs, vols, vol_ratios, reasons_list, breakout_probs, position_sizes, position_reasons = [], [], [], [], [], [], [], [], [], [], []
for idx, row in df.iterrows():
    kline = {"close": row["close"], "ema50": row["ema50"], "ema200": row["ema200"], "rsi": row["rsi"] if pd.notna(row["rsi"]) else 50,
             "volume": row["volume"], "high20": row["high20"] if pd.notna(row["high20"]) else None, "low20": row["low20"] if pd.notna(row["low20"]) else None}
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
    breakout_probs.append(result["breakout_prob"])
    position_sizes.append(result["position_size"])
    position_reasons.append(result["position_reason"])

df["signal"] = signals
df["sl"] = sls
df["tp"] = tps

# 回测
def backtest(df, lookback=100, max_bars=30):
    recent = df.tail(lookback).copy()
    trades = recent[recent["signal"].isin(["强做多", "强做空", "弱做多", "弱做空"])]
    if len(trades) == 0:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "freq": 0, "avg_win": 0, "avg_loss": 0, "expectancy": 0, "records": []}
    wins, losses, win_pnls, loss_pnls, records = 0, 0, [], [], []
    for idx, row in trades.iterrows():
        pos = df.index.get_loc(idx)
        if pos + max_bars >= len(df): continue
        entry, sl, tp = row["close"], row["sl"], row["tp"]
        is_long = "多" in row["signal"]
        for j in range(pos + 1, min(pos + max_bars + 1, len(df))):
            hit_sl = df.iloc[j]["low"] <= sl if is_long else df.iloc[j]["high"] >= sl
            hit_tp = df.iloc[j]["high"] >= tp if is_long else df.iloc[j]["low"] <= tp
            if hit_sl:
                pnl = (sl - entry) / entry * 100 if is_long else (entry - sl) / entry * 100
                loss_pnls.append(pnl); losses += 1
                records.append({"时间": row["time"].strftime("%m-%d %H:%M"), "信号": row["signal"], "入场": entry, "出场": sl, "盈亏%": round(pnl,2), "结果": "❌"})
                break
            if hit_tp:
                pnl = (tp - entry) / entry * 100 if is_long else (entry - tp) / entry * 100
                win_pnls.append(pnl); wins += 1
                records.append({"时间": row["time"].strftime("%m-%d %H:%M"), "信号": row["signal"], "入场": entry, "出场": tp, "盈亏%": round(pnl,2), "结果": "✅"})
                break
    total = wins + losses
    return {"total": len(trades), "freq": len(trades)/lookback*100, "wins": wins, "losses": losses,
            "win_rate": wins/total*100 if total > 0 else 0, "avg_win": np.mean(win_pnls) if win_pnls else 0,
            "avg_loss": np.mean(loss_pnls) if loss_pnls else 0, "expectancy": (wins/total*np.mean(win_pnls) - losses/total*abs(np.mean(loss_pnls))) if total > 0 else 0,
            "records": records}

bt = backtest(df, 100, 30)

# 当前状态
last = df.iloc[-2]
sig = last["signal"]
last_vol = vol_ratios[-2]
last_volatility = vols[-2]
last_prob = breakout_probs[-2]
last_pos = position_sizes[-2]
last_reasons = reasons_list[-2]

# ==================== UI ====================
st.markdown(f"""
<style>
.metric-card {{background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 5px;}}
.signal-strong {{background: linear-gradient(90deg, #00b894, #00cec9); padding: 20px; border-radius: 15px;}}
.signal-weak {{background: linear-gradient(90deg, #fdcb6e, #e17055); padding: 20px; border-radius: 15px;}}
.signal-hold {{background: linear-gradient(90deg, #636e72, #2d3436); padding: 20px; border-radius: 15px;}}
</style>
""", unsafe_allow_html=True)

# 标题栏
col1, col2, col3 = st.columns([2, 3, 2])
with col1:
    st.markdown(f"### 📊 ETH 5分钟量化系统")
with col2:
    st.markdown(f"<h2 style='text-align:center'>${last['close']:.2f}</h2>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<p style='text-align:right'>{data_source} | {last['time'].strftime('%m-%d %H:%M')}</p>", unsafe_allow_html=True)

st.divider()

# 核心指标
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("趋势", last["trend"], delta="多头" if last["trend"]=="多头" else "空头")
c2.metric("RSI", f"{last['rsi']:.0f}", delta="✓" if 45 <= last['rsi'] <= 55 else "⚠")
c3.metric("波动率", f"{last_volatility*100:.2f}%", delta="✓" if last_volatility >= 0.0015 else "低")
c4.metric("成交量", f"{last_vol:.1f}x", delta="✓" if last_vol >= 1.5 else "缩量")
c5.metric("突破概率", f"{last_prob:.0f}%")

# 信号显示
st.divider()
if sig == "强做多":
    st.success(f"### 🟢 {sig} | 仓位 {last_pos}% | 信心 {confs[-2]*100:.0f}%")
elif sig == "强做空":
    st.error(f"### 🔴 {sig} | 仓位 {last_pos}% | 信心 {confs[-2]*100:.0f}%")
elif sig in ["弱做多", "弱做空"]:
    st.info(f"### 🟡 {sig} | 仓位 {last_pos}% | 信心 {confs[-2]*100:.0f}%")
else:
    st.warning(f"### ⚪ {sig}")

# 交易计划
if sig in ["强做多", "弱做多", "强做空", "弱做空"]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", f"${last['sl']:.2f}")
    col3.metric("止盈", f"${last['tp']:.2f}")
    risk = abs(last['close'] - last['sl'])
    reward = abs(last['tp'] - last['close'])
    col4.metric("盈亏比", f"1:{reward/risk:.1f}" if risk > 0 else "N/A")
else:
    st.info(f"**原因:** {', '.join(last_reasons)}")

st.divider()

# 图表和记录用Tabs
tab1, tab2, tab3 = st.tabs(["📈 价格走势", "📊 回测统计", "📝 交易记录"])

with tab1:
    # K线图
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                  increasing_line_color='#26a69a', decreasing_line_color='#ef5350', name="ETH"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], line=dict(color='orange', width=1), name="EMA50"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], line=dict(color='blue', width=2), name="EMA200"))
    fig.add_hline(y=last["high20"], line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=last["low20"], line_dash="dash", line_color="green", opacity=0.5)
    
    # 信号标记
    for s, color, sym in [("强做多","lime",15),("强做空","red",15),("弱做多","lightgreen",8),("弱做空","lightcoral",8)]:
        mask = df["signal"] == s
        if mask.any():
            fig.add_trace(go.Scatter(x=df["time"][mask], y=df["close"][mask], mode="markers",
                                     marker=dict(symbol="triangle-up" if "多" in s else "triangle-down", size=sym, color=color), name=s))
    
    if sig in ["强做多", "弱做多", "强做空", "弱做空"] and last["sl"] > 0:
        fig.add_hline(y=last["sl"], line_dash="dash", line_color="red", line_width=2)
        fig.add_hline(y=last["tp"], line_dash="dash", line_color="green", line_width=2)
    
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("交易次数", bt["total"])
    c2.metric("胜/负", f"{bt['wins']}/{bt['losses']}")
    c3.metric("胜率", f"{bt['win_rate']:.1f}%")
    c4.metric("平均盈利", f"{bt['avg_win']:.2f}%")
    c5.metric("平均亏损", f"{bt['avg_loss']:.2f}%")
    c6.metric("期望值", f"{bt['expectancy']:.3f}%")

with tab3:
    records = bt.get("records", [])
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    else:
        st.info("暂无交易记录")

# 侧边栏参数
with st.sidebar:
    st.markdown("### 📖 策略参数")
    st.info(f"""
**RSI阈值:** 55/45
**成交量:** ≥1.5x
**波动率:** ≥0.15%
**止损:** 1.8 ATR
**止盈:** 2.8 ATR
**突破概率:** {last_prob:.0f}%
**当前仓位:** {last_pos}%
""")
