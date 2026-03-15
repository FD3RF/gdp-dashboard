"""
ETH 5分钟历史数据回测脚本
使用 v9.6 策略逻辑进行1年历史回测
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time

# ==================== 配置 ====================
SYMBOL = "ETH-USDT"
TIMEFRAME = "5m"
DAYS = 365  # 1年历史数据

# 固定参数 (v9.6)
PARAMS = {
    "rsi_long": 55,
    "rsi_short": 45,
    "volume_mult": 1.5,
    "volatility_thresh": 0.0015,
    "stop_loss_mult": 1.8,
    "take_profit_mult": 2.8
}


# ==================== 数据获取 ====================
def generate_simulated_data(days=365):
    """生成模拟的ETH价格数据 (用于API不可用时)"""
    print(f"📊 生成 {days} 天模拟数据...")
    
    np.random.seed(42)
    bars = int(days * 288)  # 每天288根5分钟K线
    
    # 初始价格
    price = 2000
    prices = []
    volumes = []
    times = []
    
    start_time = datetime.now() - timedelta(days=days)
    
    for i in range(bars):
        # 趋势 + 噪声
        trend = 0.0001 * np.sin(i / 500)  # 长期趋势
        noise = np.random.normal(0, 0.002)  # 短期噪声
        
        change = trend + noise
        price = price * (1 + change)
        
        # OHLC
        high = price * (1 + abs(np.random.normal(0, 0.001)))
        low = price * (1 - abs(np.random.normal(0, 0.001)))
        open_price = price * (1 + np.random.normal(0, 0.0005))
        close_price = price
        
        # 成交量 (带波动)
        base_vol = 1000
        vol = base_vol * (1 + np.random.normal(0, 0.5))
        vol = max(vol, 100)
        
        times.append(start_time + timedelta(minutes=5*i))
        prices.append([open_price, high, low, close_price])
        volumes.append(vol)
    
    df = pd.DataFrame(prices, columns=["open", "high", "low", "close"])
    df["time"] = times
    df["volume"] = volumes
    
    return df


def fetch_klines(days=30):
    """获取K线数据 - 优先使用Binance,备选OKX"""
    bars_per_day = 288  # 5分钟K线
    limit = min(int(bars_per_day * days), 1000)
    
    # 尝试Binance
    try:
        print(f"📥 从Binance获取ETH数据 ({days}天)...")
        url = f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=5m&limit={limit}"
        r = requests.get(url, timeout=20)
        data = r.json()
        
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])
            df["time"] = pd.to_datetime(df["open_time"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            df = df.sort_values("time").reset_index(drop=True)
            print(f"✅ Binance: {len(df)} 根K线")
            return df, "Binance"
    except Exception as e:
        print(f"Binance失败: {e}")
    
    # 尝试OKX
    try:
        print(f"📥 从OKX获取ETH数据...")
        url = f"https://www.okx.com/api/v5/market/candles?instId=ETH-USDT&bar=5m&limit={limit}"
        r = requests.get(url, timeout=20)
        result = r.json()
        
        if result.get("code") == "0" and result.get("data"):
            data = result["data"]
            df = pd.DataFrame(data, columns=[
                "time", "open", "high", "low", "close", "volume",
                "volCcy", "volCcyQuote", "confirm"
            ])
            df["time"] = pd.to_datetime(df["time"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            df = df.sort_values("time").reset_index(drop=True)
            print(f"✅ OKX: {len(df)} 根K线")
            return df, "OKX"
    except Exception as e:
        print(f"OKX失败: {e}")
    
    # 生成模拟数据
    print("⚠️ API不可用，生成模拟数据...")
    return generate_simulated_data(days), "模拟"


# ==================== 指标计算 ====================
def calculate_indicators(df):
    """计算技术指标"""
    print("📊 计算技术指标...")
    
    # EMA
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    
    # 成交量
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"]
    
    # 波动率
    df["volatility"] = df["atr"] / df["close"]
    
    # 高低点
    df["high20"] = df["high"].rolling(20).max()
    df["low20"] = df["low"].rolling(20).min()
    df["high15"] = df["high"].rolling(15).max()
    df["low15"] = df["low"].rolling(15).min()
    
    # 突破
    df["breakout_up"] = df["close"] > df["high15"].shift(1)
    df["breakout_down"] = df["close"] < df["low15"].shift(1)
    
    # 趋势
    df["trend"] = np.where(df["ema50"] > df["ema200"], "多头", "空头")
    
    return df


# ==================== 信号生成 (v9.6) ====================
def generate_signals(df):
    """生成交易信号"""
    print("🎯 生成交易信号...")
    
    signals = []
    sls = []
    tps = []
    
    for i in range(200, len(df)):  # 从200开始，确保指标稳定
        row = df.iloc[i]
        
        close = row["close"]
        ema50 = row["ema50"]
        ema200 = row["ema200"]
        rsi = row["rsi"] if pd.notna(row["rsi"]) else 50
        atr = row["atr"] if pd.notna(row["atr"]) else close * 0.01
        vol_ratio = row["vol_ratio"] if pd.notna(row["vol_ratio"]) else 0
        volatility = row["volatility"] if pd.notna(row["volatility"]) else 0
        breakout_up = row["breakout_up"] if pd.notna(row["breakout_up"]) else False
        breakout_down = row["breakout_down"] if pd.notna(row["breakout_down"]) else False
        
        signal = "HOLD"
        sl = 0
        tp = 0
        
        # 趋势判断
        trend_up = ema50 > ema200
        trend_down = ema50 < ema200
        
        # 条件判断
        rsi_ok_long = rsi >= PARAMS["rsi_long"]
        rsi_ok_short = rsi <= PARAMS["rsi_short"]
        vol_ok = volatility >= PARAMS["volatility_thresh"]
        volume_ok = vol_ratio >= PARAMS["volume_mult"]
        volume_weak = vol_ratio >= 0.8  # 弱信号成交量阈值
        
        # 突破时忽略波动率过滤
        vol_filter_long = vol_ok or breakout_up
        vol_filter_short = vol_ok or breakout_down
        
        # ===== 做多判断 =====
        if trend_up:
            if rsi_ok_long and vol_filter_long and breakout_up and volume_ok:
                signal = "STRONG_LONG"
                sl = close - PARAMS["stop_loss_mult"] * atr
                tp = close + PARAMS["take_profit_mult"] * atr
            elif rsi_ok_long and vol_ok and volume_ok:
                signal = "STRONG_LONG"
                sl = close - PARAMS["stop_loss_mult"] * atr
                tp = close + PARAMS["take_profit_mult"] * atr
            elif rsi_ok_long and vol_filter_long and breakout_up and volume_weak:
                signal = "WEAK_LONG"
                sl = close - PARAMS["stop_loss_mult"] * atr
                tp = close + PARAMS["take_profit_mult"] * atr
            elif rsi_ok_long and vol_ok and volume_weak:
                signal = "WEAK_LONG"
                sl = close - PARAMS["stop_loss_mult"] * atr
                tp = close + PARAMS["take_profit_mult"] * atr
        
        # ===== 做空判断 =====
        elif trend_down:
            if rsi_ok_short and vol_filter_short and breakout_down and volume_ok:
                signal = "STRONG_SHORT"
                sl = close + PARAMS["stop_loss_mult"] * atr
                tp = close - PARAMS["take_profit_mult"] * atr
            elif rsi_ok_short and vol_ok and volume_ok:
                signal = "STRONG_SHORT"
                sl = close + PARAMS["stop_loss_mult"] * atr
                tp = close - PARAMS["take_profit_mult"] * atr
            elif rsi_ok_short and vol_filter_short and breakout_down and volume_weak:
                signal = "WEAK_SHORT"
                sl = close + PARAMS["stop_loss_mult"] * atr
                tp = close - PARAMS["take_profit_mult"] * atr
            elif rsi_ok_short and vol_ok and volume_weak:
                signal = "WEAK_SHORT"
                sl = close + PARAMS["stop_loss_mult"] * atr
                tp = close - PARAMS["take_profit_mult"] * atr
        
        signals.append(signal)
        sls.append(sl)
        tps.append(tp)
    
    # 填充前200个
    signals = ["HOLD"] * 200 + signals
    sls = [0] * 200 + sls
    tps = [0] * 200 + tps
    
    df["signal"] = signals
    df["sl"] = sls
    df["tp"] = tps
    
    return df


# ==================== 回测引擎 ====================
def run_backtest(df, max_bars=30):
    """运行回测"""
    print("🔄 运行回测...")
    
    trades = df[df["signal"].isin(["STRONG_LONG", "WEAK_LONG", "STRONG_SHORT", "WEAK_SHORT"])]
    print(f"   发现 {len(trades)} 个交易信号")
    
    if len(trades) == 0:
        return None
    
    results = []
    
    for idx, row in trades.iterrows():
        pos = df.index.get_loc(idx)
        if pos + max_bars >= len(df):
            continue
        
        entry = row["close"]
        sl = row["sl"]
        tp = row["tp"]
        signal = row["signal"]
        
        # 模拟交易
        if "LONG" in signal:
            for j in range(pos + 1, min(pos + max_bars + 1, len(df))):
                if df.iloc[j]["low"] <= sl:
                    pnl = (sl - entry) / entry * 100
                    results.append({
                        "time": row["time"],
                        "signal": signal,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "exit": sl,
                        "pnl": pnl,
                        "result": "LOSS"
                    })
                    break
                if df.iloc[j]["high"] >= tp:
                    pnl = (tp - entry) / entry * 100
                    results.append({
                        "time": row["time"],
                        "signal": signal,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "exit": tp,
                        "pnl": pnl,
                        "result": "WIN"
                    })
                    break
        else:  # SHORT
            for j in range(pos + 1, min(pos + max_bars + 1, len(df))):
                if df.iloc[j]["high"] >= sl:
                    pnl = (entry - sl) / entry * 100
                    results.append({
                        "time": row["time"],
                        "signal": signal,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "exit": sl,
                        "pnl": pnl,
                        "result": "LOSS"
                    })
                    break
                if df.iloc[j]["low"] <= tp:
                    pnl = (entry - tp) / entry * 100
                    results.append({
                        "time": row["time"],
                        "signal": signal,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "exit": tp,
                        "pnl": pnl,
                        "result": "WIN"
                    })
                    break
    
    return pd.DataFrame(results)


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("📊 ETH 5分钟 历史回测系统 v9.6")
    print("=" * 60)
    
    # 获取数据
    df, source = fetch_klines(DAYS)
    print(f"   数据源: {source}")
    
    # 计算指标
    df = calculate_indicators(df)
    
    # 生成信号
    df = generate_signals(df)
    
    # 运行回测
    results = run_backtest(df)
    
    if results is None or len(results) == 0:
        print("\n❌ 没有完成的交易")
        return
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📈 回测结果")
    print("=" * 60)
    
    total = len(results)
    wins = len(results[results["result"] == "WIN"])
    losses = len(results[results["result"] == "LOSS"])
    win_rate = wins / total * 100 if total > 0 else 0
    
    total_pnl = results["pnl"].sum()
    avg_win = results[results["result"] == "WIN"]["pnl"].mean() if wins > 0 else 0
    avg_loss = results[results["result"] == "LOSS"]["pnl"].mean() if losses > 0 else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss) if total > 0 else 0
    
    # 按信号类型统计
    strong_trades = results[results["signal"].str.contains("STRONG")]
    weak_trades = results[results["signal"].str.contains("WEAK")]
    
    print(f"\n📊 总体统计:")
    print(f"   交易次数: {total}")
    print(f"   胜/负: {wins}/{losses}")
    print(f"   胜率: {win_rate:.1f}%")
    print(f"   总盈亏: {total_pnl:.2f}%")
    print(f"   平均盈利: {avg_win:.2f}%")
    print(f"   平均亏损: {avg_loss:.2f}%")
    print(f"   期望值: {expectancy:.3f}%")
    print(f"   盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   盈亏比: N/A")
    
    if len(strong_trades) > 0:
        strong_wins = len(strong_trades[strong_trades["result"] == "WIN"])
        print(f"\n🟢 强信号统计:")
        print(f"   交易次数: {len(strong_trades)}")
        print(f"   胜率: {strong_wins/len(strong_trades)*100:.1f}%")
        print(f"   总盈亏: {strong_trades['pnl'].sum():.2f}%")
    
    if len(weak_trades) > 0:
        weak_wins = len(weak_trades[weak_trades["result"] == "WIN"])
        print(f"\n🟡 弱信号统计:")
        print(f"   交易次数: {len(weak_trades)}")
        print(f"   胜率: {weak_wins/len(weak_trades)*100:.1f}%")
        print(f"   总盈亏: {weak_trades['pnl'].sum():.2f}%")
    
    # 按月统计
    results["month"] = pd.to_datetime(results["time"]).dt.to_period("M")
    monthly = results.groupby("month").agg({
        "pnl": ["count", "sum", lambda x: (x > 0).sum() / len(x) * 100]
    }).round(2)
    monthly.columns = ["交易次数", "总盈亏%", "胜率%"]
    
    print(f"\n📅 月度统计:")
    print(monthly.to_string())
    
    # 保存结果
    results.to_csv("/workspace/gdp-dashboard/backtest_results.csv", index=False)
    print(f"\n💾 结果已保存到: backtest_results.csv")
    
    return results


if __name__ == "__main__":
    main()
