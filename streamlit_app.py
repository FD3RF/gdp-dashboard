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

# ==================== v10.1 优化版信号系统 ====================
def get_eth_signal(kline, atr, avg_volume, params=None):
    """
    v10.1优化版: 
    - 增加60-70分信号仓位
    - 过滤弱信号(<60分)
    - 优化做空逻辑
    - 改进震荡策略
    """
    close = kline["close"]
    ema50 = kline["ema50"]
    ema200 = kline["ema200"]
    rsi = kline["rsi"]
    volume = kline["volume"]
    high20 = kline.get("high20", None)
    low20 = kline.get("low20", None)

    # ========== 基础指标 ==========
    trend = "多头" if ema50 > ema200 else "空头" if ema50 < ema200 else "横盘"
    trend_dir = 1 if ema50 > ema200 else -1 if ema50 < ema200 else 0
    volatility = atr / close if close > 0 else 0
    vol_ratio = volume / avg_volume if avg_volume > 0 else 0
    breakout_long = high20 is not None and close > high20
    breakout_short = low20 is not None and close < low20

    # ========== AI评分模型 (0-100) ==========
    score = 0
    score_details = []

    # 趋势分 (20分)
    if trend_dir > 0:
        score += 20
        score_details.append("趋势+20")
    elif trend_dir < 0:
        score += 10
        score_details.append("趋势+10")

    # RSI分 (20分) - 多头更宽松，空头更严格
    if trend_dir > 0:  # 多头
        if rsi >= 70: score += 20
        elif rsi >= 60: score += 15
        elif rsi >= 55: score += 10
        elif rsi >= 50: score += 5
    elif trend_dir < 0:  # 空头 - 需要更极端的RSI
        if rsi <= 25: score += 20
        elif rsi <= 30: score += 15
        elif rsi <= 35: score += 10
        elif rsi <= 40: score += 5

    # 成交量分 (20分)
    if vol_ratio >= 2.0: score += 20
    elif vol_ratio >= 1.5: score += 15
    elif vol_ratio >= 1.2: score += 10
    elif vol_ratio >= 1.0: score += 5

    # 波动率分 (20分)
    if volatility >= 0.002: score += 20
    elif volatility >= 0.0015: score += 15
    elif volatility >= 0.001: score += 10
    elif volatility >= 0.0008: score += 5

    # 突破分 (20分) - 多头给更多分
    if trend_dir > 0 and breakout_long:
        score += 20
        score_details.append("突破+20")
    elif trend_dir < 0 and breakout_short:
        score += 15
        score_details.append("突破+15")
    elif trend_dir > 0 and high20 and close > high20 * 0.998:
        score += 10
        score_details.append("接近突破+10")
    elif trend_dir < 0 and low20 and close < low20 * 1.002:
        score += 8
        score_details.append("接近突破+8")

    # ========== 信号类型判断 ==========
    signal, signal_type = "观望", "none"
    stop_loss, take_profit = 0, 0
    position_size = 0
    reasons = []

    # --- 趋势策略 (过滤<60分) ---
    if score >= 80:  # 强信号
        if trend_dir > 0:
            signal = "强做多"
            signal_type = "strong"
            stop_loss = close - 1.8 * atr
            take_profit = close + 3.0 * atr
            position_size = 60
            reasons = ["强信号", f"评分{score}", "突破确认"]
        elif trend_dir < 0:
            # 强做空需要额外条件
            if breakout_short and vol_ratio >= 1.5:
                signal = "强做空"
                signal_type = "strong"
                stop_loss = close + 2.0 * atr
                take_profit = close - 2.5 * atr
                position_size = 30
                reasons = ["强做空", f"评分{score}", "突破+放量"]
            else:
                # 不满足强做空条件，降级
                signal = "中做空"
                signal_type = "medium"
                stop_loss = close + 1.8 * atr
                take_profit = close - 2.0 * atr
                position_size = 25
                reasons = ["中做空", f"评分{score}"]

    elif score >= 70:  # 中强信号 - 最佳区间
        if trend_dir > 0:
            signal = "中做多"
            signal_type = "medium"
            stop_loss = close - 1.5 * atr
            take_profit = close + 2.5 * atr
            position_size = 80  # ★ 增加仓位
            reasons = ["中强信号", f"评分{score}", "最佳区间"]
        elif trend_dir < 0:
            signal = "中做空"
            signal_type = "medium"
            stop_loss = close + 1.8 * atr
            take_profit = close - 2.0 * atr
            position_size = 40
            reasons = ["中做空", f"评分{score}"]

    elif score >= 60:  # 中等信号 - 次佳区间
        if trend_dir > 0:
            signal = "中做多"
            signal_type = "medium"
            stop_loss = close - 1.5 * atr
            take_profit = close + 2.0 * atr
            position_size = 70  # ★ 增加仓位
            reasons = ["中信号", f"评分{score}"]
        elif trend_dir < 0:
            signal = "中做空"
            signal_type = "medium"
            stop_loss = close + 1.5 * atr
            take_profit = close - 2.0 * atr
            position_size = 30
            reasons = ["中做空", f"评分{score}"]

    # ★ 过滤50-59分弱信号

    # --- 改进震荡策略 ---
    if signal == "观望" and volatility <= 0.0012:
        # 多头震荡：RSI超卖 + 接近支撑
        if rsi <= 30 and (low20 is None or close <= low20 * 1.01):
            signal = "震荡多"
            signal_type = "range"
            stop_loss = close - 0.8 * atr
            take_profit = close + 1.2 * atr
            position_size = 20
            reasons = ["震荡超卖", f"RSI={rsi:.0f}", "接近支撑"]
        # 空头震荡：RSI超买 + 接近阻力
        elif rsi >= 70 and (high20 is None or close >= high20 * 0.99):
            signal = "震荡空"
            signal_type = "range"
            stop_loss = close + 0.8 * atr
            take_profit = close - 1.2 * atr
            position_size = 15
            reasons = ["震荡超买", f"RSI={rsi:.0f}", "接近阻力"]

    # ========== 动态仓位调整 ==========
    if signal != "观望":
        if score >= 80: position_size = min(position_size, 70)
        elif score >= 70: position_size = min(position_size, 80)
        elif score >= 60: position_size = min(position_size, 70)
        if volatility < 0.001:
            position_size = int(position_size * 0.7)
    else:
        reasons = [f"评分{score}<60"]
        if volatility < 0.001: reasons.append("低波动")
        if vol_ratio < 1.0: reasons.append(f"缩量{vol_ratio:.1f}x")
        if not breakout_long and trend_dir > 0: reasons.append("未突破")

    return {
        "signal": signal,
        "signal_type": signal_type,
        "trend": trend,
        "entry_price": close,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": score / 100,
        "volatility": volatility,
        "vol_ratio": vol_ratio,
        "breakout_long": breakout_long,
        "breakout_short": breakout_short,
        "breakout_prob": score,
        "position_size": position_size,
        "position_reason": f"评分{score}→{position_size}%仓",
        "reasons": reasons,
        "score": score,
        "score_details": score_details
    }

# ==================== 数据获取 ====================
def get_data(limit=500):
    """获取K线数据，多数据源备用"""
    import time
    import os

    # 1. 尝试Binance (最多3次)
    for attempt in range(3):
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit={limit}"
            r = requests.get(url, timeout=15)
            data = r.json()
            if isinstance(data, list) and len(data) > 300:
                df = pd.DataFrame(data, columns=["open_time","open","high","low","close","volume","close_time","qvol","trades","tb","tq","ignore"])
                df["time"] = pd.to_datetime(df["open_time"], unit="ms")
                for col in ["open","high","low","close","volume"]: df[col] = pd.to_numeric(df[col])
                return df.sort_values("time").reset_index(drop=True), "Binance"
        except:
            if attempt < 2: time.sleep(1)

    # 2. 尝试OKX (最多3次)
    for attempt in range(3):
        try:
            url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=5m&limit={limit}"
            r = requests.get(url, timeout=15)
            result = r.json()
            if result.get("code") == "0" and result.get("data") and len(result["data"]) > 300:
                df = pd.DataFrame(result["data"], columns=["time","open","high","low","close","volume","vc","vq","conf"])
                df["time"] = pd.to_datetime(df["time"])
                for col in ["open","high","low","close","volume"]: df[col] = pd.to_numeric(df[col])
                return df.sort_values("time").reset_index(drop=True), "OKX"
        except:
            if attempt < 2: time.sleep(1)

    # 3. 尝试CoinGecko (免费API)
    try:
        url = "https://api.coingecko.com/api/v3/coins/ethereum/ohlc?vs_currency=usd&days=1"
        r = requests.get(url, timeout=15)
        data = r.json()
        if isinstance(data, list) and len(data) > 50:
            # CoinGecko返回 [timestamp, open, high, low, close]
            df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df["volume"] = 0  # CoinGecko OHLC不含成交量
            for col in ["open","high","low","close"]: df[col] = pd.to_numeric(df[col])
            return df.sort_values("time").reset_index(drop=True), "CoinGecko"
    except:
        pass

    # 4. 尝试CryptoCompare
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym=ETH&tsym=USD&limit={min(limit, 2000)}"
        r = requests.get(url, timeout=15)
        result = r.json()
        if result.get("Response") == "Success" and result.get("Data", {}).get("Data"):
            data = result["Data"]["Data"]
            df = pd.DataFrame(data)
            df = df[["time", "open", "high", "low", "close", "volumefrom"]].copy()
            df.columns = ["time", "open", "high", "low", "close", "volume"]
            df["time"] = pd.to_datetime(df["time"], unit="s")
            for col in ["open","high","low","close","volume"]: df[col] = pd.to_numeric(df[col])
            df = df[df["close"] > 0]
            return df.sort_values("time").reset_index(drop=True), "CryptoCompare"
    except:
        pass

    # 5. 尝试CoinCap
    try:
        url = f"https://api.coincap.io/v2/assets/ethereum/history?interval=m5&limit={limit}"
        r = requests.get(url, timeout=15)
        result = r.json()
        if result.get("data"):
            data = result["data"]
            df = pd.DataFrame(data)
            df = df[["time", "priceUsd"]].copy()
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df["priceUsd"] = pd.to_numeric(df["priceUsd"], errors="coerce")
            df["open"] = df["high"] = df["low"] = df["close"] = df["priceUsd"]
            df["volume"] = 0
            return df.sort_values("time").reset_index(drop=True), "CoinCap"
    except:
        pass
    
    # 使用本地历史数据 (尝试多个路径)
    local_paths = [
        "以太坊合约/ETHUSDT_5m_1y_okx.csv",
        "./以太坊合约/ETHUSDT_5m_1y_okx.csv",
        "/mount/src/gdp-dashboard/以太坊合约/ETHUSDT_5m_1y_okx.csv",
        "/workspace/gdp-dashboard/以太坊合约/ETHUSDT_5m_1y_okx.csv"
    ]
    
    for path in local_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if len(df) > 300:
                    # 处理时间列（可能是time或datetime）
                    if "datetime" in df.columns:
                        df["time"] = pd.to_datetime(df["datetime"])
                    elif "time" in df.columns:
                        df["time"] = pd.to_datetime(df["time"])
                    for col in ["open","high","low","close","volume"]: 
                        if col in df.columns: 
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.dropna(subset=["open","high","low","close"]).tail(500).reset_index(drop=True)
                    return df, "本地数据"
        except:
            continue
    
    # 最后备用：生成模拟数据
    return generate_mock_data(limit), "模拟数据"


def generate_mock_data(limit=500):
    """生成模拟ETH价格数据"""
    np.random.seed(42)
    times = pd.date_range(end=datetime.now(), periods=limit, freq="5min")
    price = 2100
    prices = []
    
    for i in range(limit):
        change = np.random.normal(0, 0.002)
        price = price * (1 + change)
        high = price * (1 + abs(np.random.normal(0, 0.001)))
        low = price * (1 - abs(np.random.normal(0, 0.001)))
        open_p = price * (1 + np.random.normal(0, 0.0005))
        volume = 1000 * (1 + np.random.normal(0, 0.5))
        prices.append([times[i], open_p, high, low, price, volume])
    
    df = pd.DataFrame(prices, columns=["time", "open", "high", "low", "close", "volume"])
    return df

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

# 数据验证
if df is None or len(df) < 50:
    st.error("❌ 无法获取数据，请刷新页面重试")
    st.stop()

# 显示数据源
if data_source == "模拟数据":
    st.warning("⚠️ 使用模拟数据（API不可用）")
elif data_source == "本地数据":
    st.info("📁 使用本地历史数据")

# ★ 计算指标
df = calc_indicators(df)
df = df.dropna().reset_index(drop=True)

if len(df) < 50:
    st.error("❌ 有效数据不足")
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
df["trend"] = trends
df["sl"] = sls
df["tp"] = tps

# 回测
def backtest(df, lookback=100, max_bars=30):
    recent = df.tail(lookback).copy()
    # v10: 包含所有信号类型
    trade_signals = ["强做多", "强做空", "中做多", "中做空", "弱做多", "弱做空", "震荡多", "震荡空"]
    trades = recent[recent["signal"].isin(trade_signals)]
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

# 当前状态 - 使用最后一根完整K线 (倒数第二行，因为最后一根可能未完成)
if len(df) >= 2:
    last = df.iloc[-2]
    sig = last["signal"]
    last_vol = vol_ratios[-2] if len(vol_ratios) >= 2 else 1.0
    last_volatility = vols[-2] if len(vols) >= 2 else 0.001
    last_prob = breakout_probs[-2] if len(breakout_probs) >= 2 else 0
    last_pos = position_sizes[-2] if len(position_sizes) >= 2 else 0
    last_reasons = reasons_list[-2] if len(reasons_list) >= 2 else ["数据不足"]
else:
    st.error("❌ 数据不足，无法生成交易信号")
    st.stop()

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
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("趋势", last["trend"], delta="多头" if last["trend"]=="多头" else "空头")
c2.metric("RSI", f"{last['rsi']:.0f}", delta="✓" if 45 <= last['rsi'] <= 55 else "⚠")
c3.metric("波动率", f"{last_volatility*100:.2f}%", delta="✓" if last_volatility >= 0.0015 else "低")
c4.metric("成交量", f"{last_vol:.1f}x", delta="✓" if last_vol >= 1.5 else "缩量")
c5.metric("AI评分", f"{last_prob}", delta="强" if last_prob >= 80 else "中" if last_prob >= 60 else "弱")
c6.metric("建议仓位", f"{last_pos}%")

# 信号显示 (v10更新)
st.divider()
trade_signals = ["强做多", "强做空", "中做多", "中做空", "弱做多", "弱做空", "震荡多", "震荡空"]
if sig == "强做多":
    st.success(f"### 🟢 {sig} | AI评分 {last_prob} | 仓位 {last_pos}%")
elif sig == "强做空":
    st.error(f"### 🔴 {sig} | AI评分 {last_prob} | 仓位 {last_pos}%")
elif sig in ["中做多", "中做空"]:
    st.info(f"### 🔵 {sig} | AI评分 {last_prob} | 仓位 {last_pos}%")
elif sig in ["弱做多", "弱做空"]:
    st.warning(f"### 🟡 {sig} | AI评分 {last_prob} | 仓位 {last_pos}%")
elif sig in ["震荡多", "震荡空"]:
    st.info(f"### 🔄 {sig} | 震荡策略 | 仓位 {last_pos}%")
else:
    st.warning(f"### ⚪ {sig} | AI评分 {last_prob}")

# 交易计划
if sig in trade_signals:
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
    
    # 信号标记 (v10更新)
    signal_colors = [
        ("强做多", "lime", 15), ("强做空", "red", 15),
        ("中做多", "cyan", 12), ("中做空", "orange", 12),
        ("弱做多", "lightgreen", 8), ("弱做空", "lightcoral", 8),
        ("震荡多", "yellow", 10), ("震荡空", "magenta", 10)
    ]
    for s, color, sym in signal_colors:
        mask = df["signal"] == s
        if mask.any():
            fig.add_trace(go.Scatter(x=df["time"][mask], y=df["close"][mask], mode="markers",
                                     marker=dict(symbol="triangle-up" if "多" in s else "triangle-down", size=sym, color=color), name=s))
    
    if sig in ["强做多", "弱做多", "强做空", "弱做空"] and last["sl"] > 0:
        fig.add_hline(y=last["sl"], line_dash="dash", line_color="red", line_width=2)
        fig.add_hline(y=last["tp"], line_dash="dash", line_color="green", line_width=2)
    
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

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
        st.dataframe(pd.DataFrame(records), width='stretch', hide_index=True)
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
