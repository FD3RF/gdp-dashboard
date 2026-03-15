"""
ETH 5分钟量化策略 v10 回测
使用1年历史数据回测AI评分模型 + 两级信号 + 震荡策略
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==================== v10.1 优化版信号系统 ====================
def get_eth_signal_v10(close, ema50, ema200, rsi, volume, avg_volume, atr, high20, low20):
    """
    v10.1优化版: 
    - 增加60-70分信号仓位
    - 过滤弱信号(<60分)
    - 优化做空逻辑
    - 改进震荡策略
    """
    # 基础指标
    trend_dir = 1 if ema50 > ema200 else -1 if ema50 < ema200 else 0
    volatility = atr / close if close > 0 else 0
    vol_ratio = volume / avg_volume if avg_volume > 0 else 0
    breakout_long = high20 is not None and close > high20
    breakout_short = low20 is not None and close < low20

    # ========== AI评分模型 (0-100) ==========
    score = 0

    # 趋势分 (20分)
    if trend_dir > 0:
        score += 20
    elif trend_dir < 0:
        score += 10

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
    if trend_dir > 0 and breakout_long: score += 20
    elif trend_dir < 0 and breakout_short: score += 15  # 空头突破分降低
    elif trend_dir > 0 and high20 and close > high20 * 0.998: score += 10
    elif trend_dir < 0 and low20 and close < low20 * 1.002: score += 8

    # ========== 信号类型判断 ==========
    signal = "观望"
    stop_loss, take_profit = 0, 0
    position_size = 0
    signal_type = "none"

    # --- 趋势策略 (过滤<60分) ---
    if score >= 80:  # 强信号
        if trend_dir > 0:
            signal = "强做多"
            stop_loss = close - 1.8 * atr
            take_profit = close + 3.0 * atr
            position_size = 60
            signal_type = "strong"
        elif trend_dir < 0:
            # 强做空需要额外条件
            if breakout_short and vol_ratio >= 1.5:
                signal = "强做空"
                stop_loss = close + 2.0 * atr  # 扩大止损
                take_profit = close - 2.5 * atr  # 降低止盈目标
                position_size = 30  # 降低仓位
                signal_type = "strong"

    elif score >= 70:  # 中强信号 - 最佳区间
        if trend_dir > 0:
            signal = "中做多"
            stop_loss = close - 1.5 * atr
            take_profit = close + 2.5 * atr
            position_size = 80  # ★ 增加80%仓位
            signal_type = "medium"
        elif trend_dir < 0:
            signal = "中做空"
            stop_loss = close + 1.8 * atr
            take_profit = close - 2.0 * atr
            position_size = 40
            signal_type = "medium"

    elif score >= 60:  # 中等信号 - 次佳区间
        if trend_dir > 0:
            signal = "中做多"
            stop_loss = close - 1.5 * atr
            take_profit = close + 2.0 * atr
            position_size = 70  # ★ 增加70%仓位
            signal_type = "medium"
        elif trend_dir < 0:
            signal = "中做空"
            stop_loss = close + 1.5 * atr
            take_profit = close - 2.0 * atr
            position_size = 30
            signal_type = "medium"

    # ★ 过滤50-59分弱信号 - 不再交易

    # --- 改进震荡策略 ---
    if signal == "观望" and volatility <= 0.0012:  # 放宽波动率条件
        # 多头震荡：RSI超卖 + 接近支撑
        if rsi <= 30 and (low20 is None or close <= low20 * 1.01):
            signal = "震荡多"
            stop_loss = close - 0.8 * atr  # 更紧止损
            take_profit = close + 1.2 * atr
            position_size = 20
            signal_type = "range"
        # 空头震荡：RSI超买 + 接近阻力
        elif rsi >= 70 and (high20 is None or close >= high20 * 0.99):
            signal = "震荡空"
            stop_loss = close + 0.8 * atr
            take_profit = close - 1.2 * atr
            position_size = 15
            signal_type = "range"

    # ========== 动态仓位调整 ==========
    if signal != "观望":
        # 根据评分调整仓位
        if score >= 80: position_size = min(position_size, 70)
        elif score >= 70: position_size = min(position_size, 80)  # ★ 最佳区间
        elif score >= 60: position_size = min(position_size, 70)  # ★ 次佳区间
        
        # 低波动减仓
        if volatility < 0.001:
            position_size = int(position_size * 0.7)

    return signal, signal_type, stop_loss, take_profit, position_size, score


# ==================== 计算指标 ====================
def calc_indicators(df):
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))
    
    # ATR
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"] - df["close"].shift())
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # 成交量均线
    df["vol_ma"] = df["volume"].rolling(20).mean()
    
    # 突破
    df["high20"] = df["high"].rolling(20).max()
    df["low20"] = df["low"].rolling(20).min()
    
    return df


# ==================== 回测主函数 ====================
def backtest_v10(df, max_bars=30):
    """v10回测"""
    print("🔍 正在计算信号...")
    
    # 预分配列表
    signals = []
    
    for idx, row in df.iterrows():
        if pd.isna(row["ema200"]) or pd.isna(row["rsi"]) or pd.isna(row["atr"]):
            signals.append({"signal": "观望", "score": 0})
            continue
            
        signal, signal_type, sl, tp, pos, score = get_eth_signal_v10(
            row["close"], row["ema50"], row["ema200"], row["rsi"],
            row["volume"], row["vol_ma"], row["atr"], row["high20"], row["low20"]
        )
        signals.append({"signal": signal, "sl": sl, "tp": tp, "pos": pos, "score": score, "type": signal_type})
    
    df["signal"] = [s["signal"] for s in signals]
    df["sl"] = [s["sl"] for s in signals]
    df["tp"] = [s["tp"] for s in signals]
    df["pos"] = [s["pos"] for s in signals]
    df["score"] = [s["score"] for s in signals]
    df["signal_type"] = [s["type"] for s in signals]
    
    print(f"✅ 信号计算完成")
    
    # 交易统计
    trade_signals = ["强做多", "强做空", "中做多", "中做空", "弱做多", "弱做空", "震荡多", "震荡空"]
    trades_df = df[df["signal"].isin(trade_signals)].copy()
    
    print(f"📊 总信号数: {len(trades_df)}")
    
    # 模拟交易
    results = []
    wins, losses = 0, 0
    total_pnl = 0
    total_weighted_pnl = 0
    max_drawdown = 0
    peak_pnl = 0
    pnl_history = []
    
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    signal_stats = {
        "强做多": {"wins": 0, "losses": 0, "pnl": 0},
        "强做空": {"wins": 0, "losses": 0, "pnl": 0},
        "中做多": {"wins": 0, "losses": 0, "pnl": 0},
        "中做空": {"wins": 0, "losses": 0, "pnl": 0},
        "弱做多": {"wins": 0, "losses": 0, "pnl": 0},
        "弱做空": {"wins": 0, "losses": 0, "pnl": 0},
        "震荡多": {"wins": 0, "losses": 0, "pnl": 0},
        "震荡空": {"wins": 0, "losses": 0, "pnl": 0},
    }
    
    score_ranges = {
        "80+": {"wins": 0, "losses": 0, "pnl": 0},
        "70-79": {"wins": 0, "losses": 0, "pnl": 0},
        "60-69": {"wins": 0, "losses": 0, "pnl": 0},
        "50-59": {"wins": 0, "losses": 0, "pnl": 0},
        "<50": {"wins": 0, "losses": 0, "pnl": 0},
    }
    
    for idx, row in trades_df.iterrows():
        pos = df.index.get_loc(idx)
        if pos + max_bars >= len(df):
            continue
            
        entry = row["close"]
        sl = row["sl"]
        tp = row["tp"]
        position = row["pos"] / 100
        signal_type = row["signal"]
        score = row["score"]
        is_long = "多" in signal_type
        
        # 模拟后续走势
        result_pnl = None
        for j in range(pos + 1, min(pos + max_bars + 1, len(df))):
            if is_long:
                if df.iloc[j]["low"] <= sl:  # 止损
                    result_pnl = (sl - entry) / entry * 100 * position
                    losses += 1
                    signal_stats[signal_type]["losses"] += 1
                    consecutive_wins = 0
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    break
                if df.iloc[j]["high"] >= tp:  # 止盈
                    result_pnl = (tp - entry) / entry * 100 * position
                    wins += 1
                    signal_stats[signal_type]["wins"] += 1
                    consecutive_losses = 0
                    consecutive_wins += 1
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    break
            else:
                if df.iloc[j]["high"] >= sl:  # 止损
                    result_pnl = (entry - sl) / entry * 100 * position
                    losses += 1
                    signal_stats[signal_type]["losses"] += 1
                    consecutive_wins = 0
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    break
                if df.iloc[j]["low"] <= tp:  # 止盈
                    result_pnl = (entry - tp) / entry * 100 * position
                    wins += 1
                    signal_stats[signal_type]["wins"] += 1
                    consecutive_losses = 0
                    consecutive_wins += 1
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    break
        
        if result_pnl is not None:
            total_pnl += result_pnl
            signal_stats[signal_type]["pnl"] += result_pnl
            
            # 按评分统计
            if score >= 80: score_ranges["80+"]["pnl"] += result_pnl; score_ranges["80+"]["wins" if result_pnl > 0 else "losses"] += 1
            elif score >= 70: score_ranges["70-79"]["pnl"] += result_pnl; score_ranges["70-79"]["wins" if result_pnl > 0 else "losses"] += 1
            elif score >= 60: score_ranges["60-69"]["pnl"] += result_pnl; score_ranges["60-69"]["wins" if result_pnl > 0 else "losses"] += 1
            elif score >= 50: score_ranges["50-59"]["pnl"] += result_pnl; score_ranges["50-59"]["wins" if result_pnl > 0 else "losses"] += 1
            else: score_ranges["<50"]["pnl"] += result_pnl; score_ranges["<50"]["wins" if result_pnl > 0 else "losses"] += 1
            
            # 计算回撤
            pnl_history.append(total_pnl)
            if total_pnl > peak_pnl:
                peak_pnl = total_pnl
            drawdown = peak_pnl - total_pnl
            max_drawdown = max(max_drawdown, drawdown)
    
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    
    return {
        "total_signals": len(trades_df),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "max_drawdown": max_drawdown,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "signal_stats": signal_stats,
        "score_ranges": score_ranges
    }


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ETH 5分钟量化策略 v10 回测")
    print("=" * 60)
    
    # 加载数据
    print("\n📥 加载历史数据...")
    df = pd.read_csv("以太坊合约/ETHUSDT_5m_1y_okx.csv")
    
    # 处理时间列
    if "datetime" in df.columns:
        df["time"] = pd.to_datetime(df["datetime"])
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    
    # 数据类型转换
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    
    print(f"✅ 数据加载完成: {len(df)} 条记录")
    print(f"   时间范围: {df['time'].min()} ~ {df['time'].max()}")
    
    # 计算指标
    print("\n📊 计算技术指标...")
    df = calc_indicators(df)
    df = df.dropna().reset_index(drop=True)
    print(f"✅ 有效数据: {len(df)} 条")
    
    # 运行回测
    print("\n🔄 运行回测...")
    results = backtest_v10(df)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("📈 v10 回测结果")
    print("=" * 60)
    
    print(f"\n📊 总体统计:")
    print(f"   信号总数: {results['total_signals']}")
    print(f"   交易次数: {results['total_trades']}")
    print(f"   胜/负: {results['wins']}/{results['losses']}")
    print(f"   胜率: {results['win_rate']:.1f}%")
    print(f"   总盈亏: {results['total_pnl']:.2f}%")
    print(f"   最大回撤: {results['max_drawdown']:.2f}%")
    print(f"   最大连续盈利: {results['max_consecutive_wins']}")
    print(f"   最大连续亏损: {results['max_consecutive_losses']}")
    
    print(f"\n📊 按信号类型统计:")
    print(f"   {'信号类型':<8} {'胜':<6} {'负':<6} {'胜率':<8} {'盈亏%':<10}")
    print("   " + "-" * 40)
    for sig, stats in results["signal_stats"].items():
        total = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total * 100 if total > 0 else 0
        if total > 0:
            print(f"   {sig:<8} {stats['wins']:<6} {stats['losses']:<6} {wr:.1f}%{'':<4} {stats['pnl']:.2f}%")
    
    print(f"\n📊 按AI评分统计:")
    print(f"   {'评分区间':<10} {'胜':<6} {'负':<6} {'胜率':<8} {'盈亏%':<10}")
    print("   " + "-" * 40)
    for score_range, stats in results["score_ranges"].items():
        total = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total * 100 if total > 0 else 0
        if total > 0:
            print(f"   {score_range:<10} {stats['wins']:<6} {stats['losses']:<6} {wr:.1f}%{'':<4} {stats['pnl']:.2f}%")
    
    # 计算期望值
    if results['total_trades'] > 0:
        avg_win = results['total_pnl'] / results['wins'] if results['wins'] > 0 else 0
        avg_loss = results['total_pnl'] / results['losses'] if results['losses'] > 0 else 0
        expectancy = results['total_pnl'] / results['total_trades']
        print(f"\n📊 风险指标:")
        print(f"   平均盈利: {avg_win:.3f}%")
        print(f"   平均亏损: {avg_loss:.3f}%")
        print(f"   期望值: {expectancy:.4f}%/笔")
        print(f"   盈亏比: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "   盈亏比: N/A")
        if results['max_drawdown'] > 0:
            sharpe = results['total_pnl'] / results['max_drawdown']
            print(f"   收益回撤比: {sharpe:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ 回测完成!")
    print("=" * 60)
