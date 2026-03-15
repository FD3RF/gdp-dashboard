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

# ==================== v10.2 四大策略整合系统 ====================
def get_eth_signal(kline, atr, avg_volume, params=None):
    """
    v10.2完整策略体系:
    1️⃣ 突破趋势策略 - 价格突破+成交量放大
    2️⃣ 抄底逃顶策略 - RSI极值+放量确认
    3️⃣ 区间震荡策略 - 支撑阻力反向开仓
    4️⃣ 趋势波段策略 - 均线方向+回调买入
    """
    close = kline["close"]
    ema20 = kline.get("ema20", kline.get("ema50", close))
    ema50 = kline.get("ema50", close)
    ema60 = kline.get("ema60", kline.get("ema50", close))
    ema200 = kline.get("ema200", close)
    rsi = kline["rsi"]
    rsi9 = kline.get("rsi9", rsi)
    macd = kline.get("macd", 0)
    macd_signal = kline.get("macd_signal", 0)
    volume = kline["volume"]
    high20 = kline.get("high20", None)
    low20 = kline.get("low20", None)
    breakout_up = kline.get("breakout_up", False)
    breakout_down = kline.get("breakout_down", False)
    ema_cross_up = kline.get("ema_cross_up", False)
    ema_cross_down = kline.get("ema_cross_down", False)

    # ========== 基础指标 ==========
    trend = "多头" if ema50 > ema200 else "空头" if ema50 < ema200 else "横盘"
    trend_dir = 1 if ema50 > ema200 else -1 if ema50 < ema200 else 0
    volatility = atr / close if close > 0 else 0
    vol_ratio = volume / avg_volume if avg_volume > 0 else 0

    # ========== AI评分模型 (0-100) ==========
    score = 0
    strategy_tags = []

    # ----- 1️⃣ 突破趋势策略 (最高40分) -----
    if trend_dir > 0 and breakout_up and vol_ratio >= 1.5:
        score += 40
        strategy_tags.append("突破趋势")
    elif trend_dir < 0 and breakout_down and vol_ratio >= 1.5:
        score += 35
        strategy_tags.append("突破趋势")
    elif trend_dir > 0 and breakout_up:
        score += 25
        strategy_tags.append("缩量突破")
    elif ema_cross_up and vol_ratio >= 1.5:
        score += 30
        strategy_tags.append("金叉突破")
    elif ema_cross_down and vol_ratio >= 1.5:
        score += 25
        strategy_tags.append("死叉突破")

    # ----- 2️⃣ 抄底逃顶策略 (最高30分) -----
    if rsi9 <= 25 and vol_ratio >= 1.8:  # 极度超卖+放量
        score += 30
        strategy_tags.append("抄底")
    elif rsi9 >= 75 and vol_ratio >= 1.8:  # 极度超买+放量
        score += 25
        strategy_tags.append("逃顶")
    elif rsi9 <= 30 and vol_ratio >= 1.5:
        score += 20
        strategy_tags.append("超卖")
    elif rsi9 >= 70 and vol_ratio >= 1.5:
        score += 15
        strategy_tags.append("超买")
    # MACD背离
    if macd > macd_signal and rsi < 40:
        score += 10
        strategy_tags.append("底背离")
    elif macd < macd_signal and rsi > 60:
        score += 8
        strategy_tags.append("顶背离")

    # ----- 3️⃣ 区间震荡策略 (最高20分) -----
    if volatility <= 0.0012:  # 低波动
        if low20 and close <= low20 * 1.005 and rsi < 35:
            score += 20
            strategy_tags.append("支撑反弹")
        elif high20 and close >= high20 * 0.995 and rsi > 65:
            score += 18
            strategy_tags.append("阻力回落")

    # ----- 4️⃣ 趋势波段策略 (最高20分) -----
    if trend_dir > 0:  # 多头趋势
        # 回调至EMA20/60附近
        if ema20 > ema60 and close <= ema20 * 1.005 and close >= ema20 * 0.995:
            score += 20
            strategy_tags.append("回调买入")
        elif ema20 > ema60 and close <= ema60 * 1.01 and close >= ema60 * 0.99:
            score += 18
            strategy_tags.append("深调买入")
    elif trend_dir < 0:  # 空头趋势
        if ema20 < ema60 and close >= ema20 * 0.995 and close <= ema20 * 1.005:
            score += 15
            strategy_tags.append("回调做空")

    # ----- 基础趋势分 (最高10分) -----
    if trend_dir > 0:
        score += 10
    elif trend_dir < 0:
        score += 5

    # ========== 信号类型判断 ==========
    signal, signal_type = "观望", "none"
    stop_loss, take_profit = 0, 0
    position_size = 0
    reasons = []

    # ===== 信号分级 =====
    if score >= 70:  # 强信号
        if trend_dir > 0 or "抄底" in strategy_tags or "支撑反弹" in strategy_tags or "回调买入" in strategy_tags:
            signal = "强做多"
            signal_type = "strong"
            stop_loss = close - 1.8 * atr
            take_profit = close + 3.0 * atr
            position_size = 70
            reasons = strategy_tags[:3] + [f"评分{score}"]
        elif trend_dir < 0 or "逃顶" in strategy_tags or "阻力回落" in strategy_tags:
            signal = "强做空"
            signal_type = "strong"
            stop_loss = close + 2.0 * atr
            take_profit = close - 2.5 * atr
            position_size = 50
            reasons = strategy_tags[:3] + [f"评分{score}"]

    elif score >= 50:  # 中等信号
        if trend_dir > 0 or "支撑反弹" in strategy_tags or "回调买入" in strategy_tags:
            signal = "中做多"
            signal_type = "medium"
            stop_loss = close - 1.5 * atr
            take_profit = close + 2.0 * atr
            position_size = 50
            reasons = strategy_tags[:2] + [f"评分{score}"]
        elif trend_dir < 0 or "阻力回落" in strategy_tags:
            signal = "中做空"
            signal_type = "medium"
            stop_loss = close + 1.5 * atr
            take_profit = close - 2.0 * atr
            position_size = 30
            reasons = strategy_tags[:2] + [f"评分{score}"]

    # ===== 过滤<50分信号 =====
    if signal == "观望":
        reasons = [f"评分{score}<50"]
        if volatility < 0.001: reasons.append("低波动")
        if vol_ratio < 1.0: reasons.append(f"缩量{vol_ratio:.1f}x")

    # ========== 动态仓位调整 ==========
    if signal != "观望":
        if score >= 80: position_size = min(position_size, 80)
        elif score >= 70: position_size = min(position_size, 70)
        elif score >= 60: position_size = min(position_size, 60)
        elif score >= 50: position_size = min(position_size, 40)
        if volatility < 0.001:
            position_size = int(position_size * 0.7)

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
        "breakout_long": breakout_up,
        "breakout_short": breakout_down,
        "breakout_prob": score,
        "position_size": position_size,
        "position_reason": f"评分{score}→{position_size}%仓",
        "reasons": reasons,
        "score": score,
        "strategy_tags": strategy_tags
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

# ==================== 指标计算 (v10.2 增强版) ====================
def calc_indicators(df):
    # ===== 均线系统 =====
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema60"] = df["close"].ewm(span=60, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # ===== RSI系统 =====
    # RSI(14) 标准周期
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain/loss))

    # RSI(9) 短周期 - 提高敏感度
    gain9 = delta.where(delta > 0, 0).rolling(9).mean()
    loss9 = (-delta.where(delta < 0, 0)).rolling(9).mean()
    df["rsi9"] = 100 - (100 / (1 + gain9/loss9))

    # ===== MACD =====
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ===== ATR =====
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"] - df["close"].shift())
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # ===== 成交量 =====
    df["vol_ma"] = df["volume"].rolling(20).mean()

    # ===== 支撑/阻力 =====
    df["high20"] = df["high"].rolling(20).max()
    df["low20"] = df["low"].rolling(20).min()
    df["high15"] = df["high"].rolling(15).max()
    df["low15"] = df["low"].rolling(15).min()

    # ===== 突破判断 =====
    df["breakout_up"] = df["close"] > df["high15"].shift(1)
    df["breakout_down"] = df["close"] < df["low15"].shift(1)

    # ===== 均线交叉 =====
    df["ema_cross_up"] = (df["ema20"] > df["ema60"]) & (df["ema20"].shift(1) <= df["ema60"].shift(1))
    df["ema_cross_down"] = (df["ema20"] < df["ema60"]) & (df["ema20"].shift(1) >= df["ema60"].shift(1))

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
    kline = {
        "close": row["close"],
        "ema20": row.get("ema20", row.get("ema50", row["close"])),
        "ema50": row["ema50"],
        "ema60": row.get("ema60", row.get("ema50", row["close"])),
        "ema200": row["ema200"],
        "rsi": row["rsi"] if pd.notna(row["rsi"]) else 50,
        "rsi9": row.get("rsi9", row["rsi"] if pd.notna(row["rsi"]) else 50),
        "macd": row.get("macd", 0),
        "macd_signal": row.get("macd_signal", 0),
        "volume": row["volume"],
        "high20": row["high20"] if pd.notna(row["high20"]) else None,
        "low20": row["low20"] if pd.notna(row["low20"]) else None,
        "breakout_up": row.get("breakout_up", False),
        "breakout_down": row.get("breakout_down", False),
        "ema_cross_up": row.get("ema_cross_up", False),
        "ema_cross_down": row.get("ema_cross_down", False)
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
