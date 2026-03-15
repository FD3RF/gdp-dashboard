import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta

# 自动刷新
st_autorefresh(interval=10000, key="refresh_v7_2")

st.set_page_config(page_title="ETH AI Trading System v7.2", layout="wide")

# ==================== 参数设置侧边栏 ====================
st.sidebar.header("⚙️ 参数设置")

# 阈值设置 - 优化默认值
st.sidebar.subheader("信号阈值")
long_threshold = st.sidebar.slider("做多阈值", 10, 100, 30, 5, help="分数>=此值才发出做多信号")
short_threshold = st.sidebar.slider("做空阈值", -100, -10, -30, 5, help="分数<=此值才发出做空信号")
weak_threshold = st.sidebar.slider("弱信号阈值", 5, 25, 15, 5, help="分数>=此值显示弱信号提示")

# 权重设置 - 标准化为满分100
st.sidebar.subheader("指标权重 (满分100)")
w_trend = st.sidebar.slider("趋势权重", 0, 50, 35, 5)
w_rsi = st.sidebar.slider("RSI权重", 0, 30, 20, 5)
w_volume = st.sidebar.slider("成交量权重", 0, 30, 25, 5)
w_breakout = st.sidebar.slider("突破权重", 0, 30, 20, 5)

# 过滤条件 - 默认全部启用
st.sidebar.subheader("过滤条件")
require_volume_confirm = st.sidebar.checkbox("要求放量确认", True, help="做空必须放量")
require_rsi_confirm = st.sidebar.checkbox("要求RSI确认", True, help="趋势和RSI方向一致")
use_sar_filter = st.sidebar.checkbox("启用SAR过滤", True)
use_sr_filter = st.sidebar.checkbox("启用支撑阻力过滤", True)

# 止损止盈
st.sidebar.subheader("止损止盈")
stop_atr_mult = st.sidebar.slider("止损ATR倍数", 0.5, 3.0, 1.5, 0.1)
take_atr_mult = st.sidebar.slider("止盈ATR倍数", 1.0, 6.0, 3.0, 0.1)

# ==================== 数据获取函数 ====================
def generate_mock_data(n=300):
    np.random.seed(int(datetime.now().timestamp()) % 10000)
    price = 2100.0
    times = pd.date_range(end=datetime.now(), periods=n, freq='5min')
    opens, highs, lows, closes, volumes = [], [], [], [], []
    
    for i in range(n):
        change = np.random.normal(0, 0.002)
        open_p = price
        close_p = price * (1 + change)
        vol = abs(np.random.normal(0, 0.005))
        high_p = max(open_p, close_p) * (1 + vol)
        low_p = min(open_p, close_p) * (1 - vol)
        volume = np.random.uniform(500, 2000)  # 模拟真实成交量范围
        opens.append(open_p)
        highs.append(high_p)
        lows.append(low_p)
        closes.append(close_p)
        volumes.append(volume)
        price = close_p
    
    return pd.DataFrame({
        'time': times, 'open': opens, 'high': highs, 
        'low': lows, 'close': closes, 'volume': volumes
    })

def fetch_binance():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit=300"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if not data or len(data) < 50 or isinstance(data, dict):
            return None, None
        df = pd.DataFrame(data, columns=range(12)).iloc[:, 0:6]
        df.columns = ["time","open","high","low","close","volume"]
        df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
        df["time"] = pd.to_datetime(df["time"], unit='ms')
        return df, "Binance"
    except Exception as e:
        return None, None

def fetch_okx():
    try:
        url = "https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=5m&limit=300"
        resp = requests.get(url, timeout=10)
        data = resp.json().get('data', [])
        if not data or len(data) < 50:
            return None, None
        df = pd.DataFrame(data)
        # OKX返回: [ts, open, high, low, close, volume, volumeCcy, ...]
        df = df.iloc[:, 0:6]
        df.columns = ["time","open","high","low","close","volume"]
        # 强制转换为数值类型 - 修复成交量解析问题
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df["time"] = pd.to_datetime(df["time"].astype(float), unit='ms')
        df = df.sort_values('time').reset_index(drop=True)
        # 删除异常数据
        df = df.dropna()
        if len(df) < 50:
            return None, None
        return df, "OKX"
    except Exception as e:
        return None, None

def fetch_huobi():
    try:
        url = "https://api.huobi.pro/market/history/kline?symbol=ethusdt&period=5min&size=300"
        resp = requests.get(url, timeout=10)
        data = resp.json().get('data', [])
        if not data or len(data) < 50:
            return None, None
        df = pd.DataFrame(data)
        df = df[['id','open','high','low','close','vol']]
        df.columns = ["time","open","high","low","close","volume"]
        df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
        df["time"] = pd.to_datetime(df["time"].astype(int), unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        return df, "Huobi"
    except Exception as e:
        return None, None

@st.cache_data(ttl=10)
def load_data():
    for fetch_func in [fetch_binance, fetch_okx, fetch_huobi]:
        df, source = fetch_func()
        if df is not None and len(df) >= 50:
            df['date'] = df['time'].dt.date
            return df, source
    df = generate_mock_data(300)
    df['date'] = df['time'].dt.date
    return df, "模拟数据"

df, data_source = load_data()

if df is None or len(df) < 50:
    st.error("无法获取数据")
    st.stop()

# ==================== 指标计算 ====================
# EMA
df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

# RSI
delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

# 成交量 - 修复计算
df["vol_ma"] = df["volume"].rolling(20).mean()
# 防止除零
df["vol_ratio"] = np.where(df["vol_ma"] > 0, df["volume"] / df["vol_ma"], 1.0)

# 支撑阻力
df["high20"] = df["high"].rolling(20).max()
df["low20"] = df["low"].rolling(20).min()

# ATR
tr = pd.concat([df["high"]-df["low"], 
                abs(df["high"]-df["close"].shift(1)), 
                abs(df["low"]-df["close"].shift(1))], axis=1).max(axis=1)
df["atr"] = tr.rolling(14).mean()

# SAR (抛物线转向)
def calc_sar(df, af=0.02, max_af=0.2):
    sar = np.zeros(len(df))
    if len(df) < 2:
        return sar, True
    
    ep = df["high"].iloc[0]
    af_val = af
    is_long = True
    sar[0] = df["low"].iloc[0]
    
    for i in range(1, len(df)):
        if is_long:
            sar[i] = sar[i-1] + af_val * (ep - sar[i-1])
            sar[i] = min(sar[i], df["low"].iloc[i-1], df["low"].iloc[i])
            if df["low"].iloc[i] < sar[i]:
                is_long = False
                sar[i] = ep
                ep = df["low"].iloc[i]
                af_val = af
            elif df["high"].iloc[i] > ep:
                ep = df["high"].iloc[i]
                af_val = min(af_val + af, max_af)
        else:
            sar[i] = sar[i-1] - af_val * (sar[i-1] - ep)
            sar[i] = max(sar[i], df["high"].iloc[i-1], df["high"].iloc[i])
            if df["high"].iloc[i] > sar[i]:
                is_long = True
                sar[i] = ep
                ep = df["high"].iloc[i]
                af_val = af
            elif df["low"].iloc[i] < ep:
                ep = df["low"].iloc[i]
                af_val = min(af_val + af, max_af)
    
    return sar, is_long

df["sar"], last_is_long_sar = calc_sar(df)

# ==================== 评分函数 v7.2 (标准化满分100) ====================
def calc_score_v72(row, df):
    score = 0
    details = {}
    
    # 1. 趋势方向 (满分 ±40)
    trend_dir = 1 if row["close"] > row["ema200"] else -1
    trend_score = w_trend * trend_dir
    score += trend_score
    details["趋势"] = f"{'多头' if trend_dir > 0 else '空头'} ({trend_score:+d})"
    
    # 2. RSI (满分 ±20)
    rsi_v = row["rsi"] if pd.notna(row["rsi"]) else 50
    if rsi_v > 70:
        rsi_score = w_rsi
        rsi_dir = 1
    elif rsi_v > 55:
        rsi_score = w_rsi // 2
        rsi_dir = 1
    elif rsi_v < 30:
        rsi_score = -w_rsi
        rsi_dir = -1
    elif rsi_v < 45:
        rsi_score = -w_rsi // 2
        rsi_dir = -1
    else:
        rsi_score = 0
        rsi_dir = 0
    score += rsi_score
    details["RSI"] = f"{rsi_v:.1f} ({rsi_score:+d})"
    
    # 3. 成交量 (满分 ±25)
    vol_ratio = row["vol_ratio"] if pd.notna(row["vol_ratio"]) else 1.0
    if vol_ratio > 2.0:
        vol_score = w_volume
    elif vol_ratio > 1.5:
        vol_score = int(w_volume * 0.7)
    elif vol_ratio > 1.2:
        vol_score = int(w_volume * 0.4)
    elif vol_ratio < 0.5:
        vol_score = -int(w_volume * 0.5)  # 极度缩量扣分
    elif vol_ratio < 0.8:
        vol_score = -int(w_volume * 0.2)  # 缩量小幅扣分
    else:
        vol_score = 0
    score += vol_score
    details["成交量"] = f"{vol_ratio:.2f}x ({vol_score:+d})"
    
    # 4. 突破 (满分 ±20)
    if row["close"] > row["high20"]:
        brk_score = w_breakout
        brk_dir = 1
    elif row["close"] < row["low20"]:
        brk_score = -w_breakout
        brk_dir = -1
    else:
        brk_score = 0
        brk_dir = 0
    score += brk_score
    details["突破"] = f"{'上破' if brk_dir > 0 else '下破' if brk_dir < 0 else '无'} ({brk_score:+d})"
    
    # 5. 冲突扣分
    conflict = 0
    if trend_dir != rsi_dir and rsi_dir != 0:
        conflict += 5
    if trend_dir != brk_dir and brk_dir != 0:
        conflict += 5
    score -= conflict
    if conflict > 0:
        details["冲突"] = f"-{conflict}"
    
    return score, details

# 计算所有评分
scores = []
for idx, row in df.iterrows():
    s, d = calc_score_v72(row, df)
    scores.append(s)
df["score"] = scores

# ==================== 信号生成 v7.2 ====================
def generate_signal_v72(df):
    signals = []
    reasons = []
    
    for i, row in df.iterrows():
        sig = "HOLD"
        reason = []
        score = row["score"]
        
        if pd.isna(score):
            signals.append(sig)
            reasons.append([])
            continue
        
        trend_dir = 1 if row["close"] > row["ema200"] else -1
        rsi_v = row["rsi"] if pd.notna(row["rsi"]) else 50
        vol_ratio = row["vol_ratio"] if pd.notna(row["vol_ratio"]) else 1.0
        is_above_sar = row["close"] > row["sar"]
        dist_to_resistance = row["high20"] - row["close"]
        dist_to_support = row["close"] - row["low20"]
        atr = row["atr"] if pd.notna(row["atr"]) else 0
        
        # 做多条件
        if score >= long_threshold:
            can_long = True
            
            if require_rsi_confirm and rsi_v < 50:
                can_long = False
                reason.append("RSI未确认(<50)")
            
            if require_volume_confirm and vol_ratio < 1.0:
                can_long = False
                reason.append("未放量")
            
            if use_sar_filter and not is_above_sar:
                can_long = False
                reason.append("SAR空头")
            
            if use_sr_filter and atr > 0 and dist_to_resistance < atr * 0.5:
                can_long = False
                reason.append("接近阻力")
            
            if can_long:
                sig = "LONG"
                reason.append(f"分数{score:.0f}>={long_threshold}")
        
        # 做空条件
        elif score <= short_threshold:
            can_short = True
            
            if require_rsi_confirm and rsi_v > 50:
                can_short = False
                reason.append("RSI未确认(>50)")
            
            if require_volume_confirm and vol_ratio < 1.0:
                can_short = False
                reason.append("未放量(做空需放量)")
            
            if use_sar_filter and is_above_sar:
                can_short = False
                reason.append("SAR多头")
            
            if use_sr_filter and atr > 0 and dist_to_support < atr * 0.5:
                can_short = False
                reason.append("接近支撑")
            
            if can_short:
                sig = "SHORT"
                reason.append(f"分数{score:.0f}<={short_threshold}")
        
        # 弱信号标注
        if sig == "HOLD":
            if score >= weak_threshold and score < long_threshold:
                sig = "WEAK_LONG"
                reason.append(f"弱多({score:.0f}<{long_threshold})")
            elif score <= -weak_threshold and score > short_threshold:
                sig = "WEAK_SHORT"
                reason.append(f"弱空({score:.0f}>{short_threshold})")
        
        signals.append(sig)
        reasons.append(reason)
    
    return signals, reasons

df["signal"], df["reason"] = generate_signal_v72(df)

# ==================== 止损止盈计算 ====================
last = df.iloc[-1]
atr = last["atr"] if pd.notna(last["atr"]) else 0

if last["signal"] == "LONG":
    stop_loss = last["close"] - stop_atr_mult * atr
    take_profit = last["close"] + take_atr_mult * atr
    risk_reward = (take_profit - last["close"]) / (last["close"] - stop_loss) if stop_loss < last["close"] else 0
elif last["signal"] == "SHORT":
    stop_loss = last["close"] + stop_atr_mult * atr
    take_profit = last["close"] - take_atr_mult * atr
    risk_reward = (last["close"] - take_profit) / (stop_loss - last["close"]) if stop_loss > last["close"] else 0
else:
    stop_loss = take_profit = risk_reward = None

# ==================== 回测统计 (使用ATR止损止盈) ====================
def backtest_with_sl_tp(df, lookback=100, max_bars=30):
    """使用ATR止损止盈计算真实胜率"""
    recent = df.tail(lookback).copy()
    signals = recent[recent["signal"].isin(["LONG", "SHORT"])]
    
    total_trades = len(signals)
    if total_trades == 0:
        return {"total": 0, "win_rate": 0, "avg_pnl": 0, "signal_freq": 0, 
                "wins": 0, "losses": 0, "avg_win": 0, "avg_loss": 0}
    
    wins = 0
    losses = 0
    win_pnls = []
    loss_pnls = []
    
    for idx, row in signals.iterrows():
        pos = df.index.get_loc(idx)
        if pos + max_bars >= len(df):
            continue
            
        entry = row["close"]
        atr_val = row["atr"] if pd.notna(row["atr"]) else entry * 0.01
        
        if row["signal"] == "LONG":
            sl = entry - stop_atr_mult * atr_val
            tp = entry + take_atr_mult * atr_val
            # 检查未来K线谁先触发
            for j in range(pos + 1, min(pos + max_bars + 1, len(df))):
                future_low = df.iloc[j]["low"]
                future_high = df.iloc[j]["high"]
                if future_low <= sl:  # 止损触发
                    pnl = (sl - entry) / entry * 100
                    loss_pnls.append(pnl)
                    losses += 1
                    break
                elif future_high >= tp:  # 止盈触发
                    pnl = (tp - entry) / entry * 100
                    win_pnls.append(pnl)
                    wins += 1
                    break
        else:  # SHORT
            sl = entry + stop_atr_mult * atr_val
            tp = entry - take_atr_mult * atr_val
            for j in range(pos + 1, min(pos + max_bars + 1, len(df))):
                future_low = df.iloc[j]["low"]
                future_high = df.iloc[j]["high"]
                if future_high >= sl:  # 止损触发
                    pnl = (entry - sl) / entry * 100
                    loss_pnls.append(pnl)
                    losses += 1
                    break
                elif future_low <= tp:  # 止盈触发
                    pnl = (entry - tp) / entry * 100
                    win_pnls.append(pnl)
                    wins += 1
                    break
    
    total = wins + losses
    return {
        "total": total_trades,
        "signal_freq": total_trades / lookback * 100,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / total * 100) if total > 0 else 0,
        "avg_win": np.mean(win_pnls) if win_pnls else 0,
        "avg_loss": np.mean(loss_pnls) if loss_pnls else 0,
        "avg_pnl": (np.mean(win_pnls + loss_pnls)) if (win_pnls or loss_pnls) else 0
    }

bt_result = backtest_with_sl_tp(df, 100, 30)

# ==================== UI展示 ====================
st.title("📊 ETH 5分钟 AI统一决策 v7.2")
st.markdown(f"**数据源:** {data_source} | **最新价格:** ${last['close']:.2f} | **时间:** {last['time'].strftime('%Y-%m-%d %H:%M')}")

# 主要指标
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ETH价格", f"${last['close']:.2f}")
c2.metric("AI评分", round(last["score"], 2) if pd.notna(last["score"]) else 0)
c3.metric("信号频率", f"{bt_result['signal_freq']:.1f}%")
c4.metric("回测胜率", f"{bt_result['win_rate']:.1f}%")
c5.metric("平均盈亏", f"{bt_result['avg_pnl']:.2f}%")

# AI决策
st.subheader("🎯 AI决策")
if last["signal"] == "LONG":
    st.success(f"🟢 **做多信号**")
    col1, col2, col3 = st.columns(3)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", f"${stop_loss:.2f}", f"-{(last['close']-stop_loss)/last['close']*100:.2f}%")
    col3.metric("止盈", f"${take_profit:.2f}", f"+{(take_profit-last['close'])/last['close']*100:.2f}%")
    st.info(f"📈 盈亏比: {risk_reward:.2f}:1")
elif last["signal"] == "SHORT":
    st.error(f"🔴 **做空信号**")
    col1, col2, col3 = st.columns(3)
    col1.metric("入场价", f"${last['close']:.2f}")
    col2.metric("止损", f"${stop_loss:.2f}", f"+{(stop_loss-last['close'])/last['close']*100:.2f}%")
    col3.metric("止盈", f"${take_profit:.2f}", f"-{(last['close']-take_profit)/last['close']*100:.2f}%")
    st.info(f"📈 盈亏比: {risk_reward:.2f}:1")
elif last["signal"] == "WEAK_LONG":
    st.info(f"🟡 **弱做多信号** - 接近阈值，建议观望")
    st.markdown(f"当前评分 **{last['score']:.0f}** < 做多阈值 **{long_threshold}**，差 **{long_threshold - last['score']:.0f}** 分")
elif last["signal"] == "WEAK_SHORT":
    st.info(f"🟡 **弱做空信号** - 接近阈值，建议观望")
    st.markdown(f"当前评分 **{last['score']:.0f}** > 做空阈值 **{short_threshold}**，差 **{last['score'] - short_threshold:.0f}** 分")
else:
    st.warning("⚪ **观望** - 无明确信号")
    reasons = df.iloc[-1]["reason"]
    if reasons:
        st.info(f"原因: {', '.join(reasons)}")

# 四维指标状态
st.subheader("📍 四维指标状态")
c1, c2, c3, c4 = st.columns(4)
trend_status = "🟢多头" if last["close"] > last["ema200"] else "🔴空头"
rsi_v = last["rsi"] if pd.notna(last["rsi"]) else 50
rsi_status = "🟢超买" if rsi_v > 70 else "🔴超卖" if rsi_v < 30 else "🟡中性"
vol_ratio = last["vol_ratio"] if pd.notna(last["vol_ratio"]) else 1.0
vol_status = "🟢放量" if vol_ratio > 1.2 else "🔴缩量" if vol_ratio < 0.8 else "🟡正常"
sar_status = "🟢之上" if last["close"] > last["sar"] else "🔴之下"

c1.metric("趋势(EMA200)", trend_status)
c2.metric("RSI 14", f"{rsi_v:.1f}", rsi_status)
c3.metric("成交量", f"{vol_ratio:.2f}x", vol_status)
c4.metric("SAR", sar_status)

# K线图
st.subheader("📈 价格走势与信号")
fig = go.Figure()

# K线
fig.add_trace(go.Candlestick(
    x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], 
    name="K线", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
))

# EMA
fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], name="EMA200", line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], name="EMA50", line=dict(color='orange', width=1)))

# SAR点
fig.add_trace(go.Scatter(x=df["time"], y=df["sar"], name="SAR", mode='markers', 
                         marker=dict(size=3, color='purple', symbol='diamond'), opacity=0.6))

# 支撑阻力线
fig.add_hline(y=last["high20"], line_dash="dash", line_color="red", annotation_text="阻力", opacity=0.7)
fig.add_hline(y=last["low20"], line_dash="dash", line_color="green", annotation_text="支撑", opacity=0.7)

# 信号标记 - 强信号
for sig_type, color, sym in [("LONG", "green", "triangle-up"), ("SHORT", "red", "triangle-down")]:
    mask = df["signal"] == sig_type
    if mask.any():
        fig.add_trace(go.Scatter(
            x=df["time"][mask], y=df["close"][mask], mode="markers",
            marker=dict(symbol=sym, size=15, color=color), name=sig_type
        ))

# 信号标记 - 弱信号
for sig_type, color, sym in [("WEAK_LONG", "lightgreen", "triangle-up"), ("WEAK_SHORT", "lightcoral", "triangle-down")]:
    mask = df["signal"] == sig_type
    if mask.any():
        fig.add_trace(go.Scatter(
            x=df["time"][mask], y=df["close"][mask], mode="markers",
            marker=dict(symbol=sym, size=8, color=color, opacity=0.5), name=sig_type
        ))

fig.update_layout(
    title="ETH 5分钟K线 + EMA + SAR + 支撑阻力",
    xaxis_rangeslider_visible=False, 
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# 成交量图
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=df["time"], y=df["volume"], name="成交量", marker_color='lightgray'))
fig_vol.add_trace(go.Scatter(x=df["time"], y=df["vol_ma"], name="均量20", line=dict(color='blue', width=1)))
fig_vol.update_layout(title="成交量", height=150, showlegend=False)
st.plotly_chart(fig_vol, use_container_width=True)

# RSI图
st.subheader("📉 RSI 14")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["time"], y=df["rsi"], name="RSI", line=dict(color='purple', width=2)))
fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买")
fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖")
fig2.add_hline(y=50, line_dash="dot", line_color="gray")
fig2.update_layout(yaxis_range=[0, 100], height=200)
st.plotly_chart(fig2, use_container_width=True)

# 回测统计
st.subheader("📊 回测统计 (ATR止损止盈)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("交易次数", bt_result["total"])
c2.metric("盈利次数", bt_result["wins"])
c3.metric("亏损次数", bt_result["losses"])
c4.metric("胜率", f"{bt_result['win_rate']:.1f}%")
c5.metric("平均盈亏", f"{bt_result['avg_pnl']:.2f}%")

# 最近信号历史
st.subheader("📋 最近20条信号")
recent_signals = df[df["signal"].isin(["LONG", "SHORT", "WEAK_LONG", "WEAK_SHORT"])].tail(20)[["time", "close", "score", "signal"]]
if len(recent_signals) > 0:
    recent_signals = recent_signals.copy()
    recent_signals["time"] = recent_signals["time"].dt.strftime('%Y-%m-%d %H:%M')
    recent_signals["close"] = recent_signals["close"].round(2)
    recent_signals["score"] = recent_signals["score"].round(1)
    st.dataframe(recent_signals, use_container_width=True)
else:
    st.info("暂无历史信号")

# 参数建议
st.sidebar.markdown("---")
st.sidebar.subheader("💡 参数建议")
st.sidebar.info(f"""
**当前设置:**
- 做多阈值: {long_threshold}
- 做空阈值: {short_threshold}
- 弱信号阈值: ±{weak_threshold}
- 信号频率: {bt_result['signal_freq']:.1f}%
- 胜率: {bt_result['win_rate']:.1f}%

**建议:**
- 频率 > 40%: 提高阈值
- 频率 < 10%: 降低阈值
- 胜率 < 50%: 启用过滤条件
- 胜率 > 70%: 可适当降低阈值
""")
