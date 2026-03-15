"""
ETH 5分钟历史数据回测 - 优化版 v10.1
使用真实OKX历史数据 + 动态仓位/止盈止损优化
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载 ====================
def load_real_data(filepath):
    """加载真实历史数据"""
    print(f"📥 加载历史数据: {filepath}")
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"✅ 加载 {len(df)} 根K线")
    print(f"   时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    return df


# ==================== 指标计算 ====================
def calculate_indicators(df):
    """计算技术指标"""
    print("📊 计算技术指标...")
    
    # EMA
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # 成交量
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    
    # 波动率
    df['volatility'] = df['atr'] / df['close']
    
    # 高低点
    df['high15'] = df['high'].rolling(15).max()
    df['low15'] = df['low'].rolling(15).min()
    df['high20'] = df['high'].rolling(20).max()
    df['low20'] = df['low'].rolling(20).min()
    
    # 突破
    df['breakout_up'] = df['close'] > df['high15'].shift(1)
    df['breakout_down'] = df['close'] < df['low15'].shift(1)
    
    # 趋势
    df['trend'] = np.where(df['ema50'] > df['ema200'], '多头', '空头')
    
    return df


# ==================== 突破概率计算 ====================
def calculate_breakout_prob(row):
    """计算突破概率"""
    prob = 0
    
    # RSI贡献 (0-30%)
    rsi = row['rsi']
    if pd.notna(rsi):
        if rsi >= 80:
            prob += 30
        elif rsi >= 70:
            prob += 20
        elif rsi >= 60:
            prob += 10
        elif rsi <= 20:
            prob += 30
        elif rsi <= 30:
            prob += 20
        elif rsi <= 40:
            prob += 10
    
    # 成交量贡献 (0-35%)
    vol_ratio = row['vol_ratio']
    if pd.notna(vol_ratio):
        if vol_ratio >= 3.0:
            prob += 35
        elif vol_ratio >= 2.0:
            prob += 25
        elif vol_ratio >= 1.5:
            prob += 15
    
    # 波动率压缩贡献 (0-35%)
    volatility = row['volatility']
    if pd.notna(volatility):
        if volatility < 0.001:
            prob += 35
        elif volatility < 0.0015:
            prob += 25
        elif volatility < 0.002:
            prob += 15
    
    return min(prob, 95)


# ==================== 动态止盈止损 v2 ====================
def get_dynamic_sl_tp(close, atr, volatility, signal_type, breakout_prob):
    """根据波动率和突破概率动态计算止盈止损 - v10.2优化版"""
    
    # 波动率调整
    if volatility >= 0.002:  # 高波动
        sl_mult = 1.0
        tp_mult = 2.5
    elif volatility >= 0.0015:  # 中波动
        sl_mult = 1.2
        tp_mult = 2.0
    elif volatility >= 0.001:  # 低波动
        sl_mult = 1.5
        tp_mult = 1.5
    else:  # 极低波动
        sl_mult = 1.8
        tp_mult = 1.2
    
    # ★ 优化：高概率信号给予更大止盈空间
    if breakout_prob >= 70:
        # 高概率信号：扩大止盈到1.5-2x
        tp_mult *= 1.8
    elif breakout_prob >= 60:
        tp_mult *= 1.3
    elif breakout_prob >= 50:
        # 最佳区间：正常止盈
        tp_mult *= 1.1
    
    # 强信号可以更激进
    if signal_type == "STRONG":
        sl_mult *= 0.9
        tp_mult *= 1.1
    
    if "LONG" in signal_type:
        sl = close - sl_mult * atr
        tp = close + tp_mult * atr
    else:
        sl = close + sl_mult * atr
        tp = close - tp_mult * atr
    
    return sl, tp, sl_mult, tp_mult


# ==================== 动态仓位计算 v2 ====================
def get_dynamic_position(breakout_prob, signal_type, vol_ratio, volatility):
    """根据突破概率动态计算仓位 - v10.2优化版"""
    
    # 低波动率时减仓
    vol_adj = 1.0
    if volatility < 0.001:
        vol_adj = 0.5
    elif volatility < 0.0015:
        vol_adj = 0.7
    
    if signal_type == "STRONG":
        # ★ 优化：50-60%概率仓位最高
        if breakout_prob >= 70:
            # 高概率：减仓到40-50%，避免止盈过早
            return int(45 * vol_adj), "高概率适度仓45%"
        elif breakout_prob >= 60:
            return int(70 * vol_adj), "中高概率70%"
        elif breakout_prob >= 50:
            # ★ 最佳区间：加仓到90-100%
            return int(95 * vol_adj), "最佳概率满仓95%"
        else:
            return int(60 * vol_adj), "低概率60%"
    
    elif signal_type == "WEAK":
        # 弱信号：同样50-60%最优
        if breakout_prob >= 70:
            return int(40 * vol_adj), "弱高概率40%"
        elif breakout_prob >= 60:
            return int(60 * vol_adj), "弱中概率60%"
        elif breakout_prob >= 50:
            # ★ 最佳区间
            return int(80 * vol_adj), "弱最佳80%"
        else:
            return 0, "弱低概率观望"
    
    return 0, "观望"


# ==================== 信号生成 v10.1 ====================
def generate_signals_optimized(df):
    """生成优化信号 v10.1"""
    print("🎯 生成优化信号 (v10.1)...")
    
    signals = []
    
    for i in range(200, len(df)):
        row = df.iloc[i]
        
        close = row['close']
        ema50 = row['ema50']
        ema200 = row['ema200']
        rsi = row['rsi'] if pd.notna(row['rsi']) else 50
        atr = row['atr'] if pd.notna(row['atr']) else close * 0.01
        vol_ratio = row['vol_ratio'] if pd.notna(row['vol_ratio']) else 0
        volatility = row['volatility'] if pd.notna(row['volatility']) else 0
        breakout_up = row['breakout_up'] if pd.notna(row['breakout_up']) else False
        breakout_down = row['breakout_down'] if pd.notna(row['breakout_down']) else False
        
        signal = {
            'signal': 'HOLD',
            'type': '',
            'sl': 0,
            'tp': 0,
            'sl_mult': 0,
            'tp_mult': 0,
            'position': 0,
            'position_reason': '',
            'breakout_prob': 0,
            'reason': ''
        }
        
        # 趋势判断
        trend_up = ema50 > ema200
        trend_down = ema50 < ema200
        
        # 条件判断
        rsi_ok_long = rsi >= 55
        rsi_ok_short = rsi <= 45
        vol_ok = volatility >= 0.001
        volume_ok = vol_ratio >= 0.8  # 降低到0.8x
        volume_strong = vol_ratio >= 1.5
        
        # 突破时忽略波动率
        vol_filter_long = vol_ok or breakout_up
        vol_filter_short = vol_ok or breakout_down
        
        # 计算突破概率
        breakout_prob = calculate_breakout_prob(row)
        signal['breakout_prob'] = breakout_prob
        
        # ★ 关键优化：过滤掉40-50%概率信号
        if breakout_prob < 50:
            signals.append(signal)
            continue
        
        # ===== 做多判断 =====
        if trend_up:
            # 强做多: 多条件满足
            if rsi_ok_long and vol_filter_long and breakout_up and volume_strong:
                signal['signal'] = 'LONG'
                signal['type'] = 'STRONG'
                sl, tp, sl_m, tp_m = get_dynamic_sl_tp(close, atr, volatility, 'STRONG_LONG', breakout_prob)
                signal['sl'] = sl
                signal['tp'] = tp
                signal['sl_mult'] = sl_m
                signal['tp_mult'] = tp_m
                pos, pos_reason = get_dynamic_position(breakout_prob, 'STRONG', vol_ratio, volatility)
                signal['position'] = pos
                signal['position_reason'] = pos_reason
                signal['reason'] = '强突破'
            
            elif rsi_ok_long and vol_ok and volume_strong:
                signal['signal'] = 'LONG'
                signal['type'] = 'STRONG'
                sl, tp, sl_m, tp_m = get_dynamic_sl_tp(close, atr, volatility, 'STRONG_LONG', breakout_prob)
                signal['sl'] = sl
                signal['tp'] = tp
                signal['sl_mult'] = sl_m
                signal['tp_mult'] = tp_m
                pos, pos_reason = get_dynamic_position(breakout_prob, 'STRONG', vol_ratio, volatility)
                signal['position'] = pos
                signal['position_reason'] = pos_reason
                signal['reason'] = '放量趋势'
            
            # 弱做多: 成交量>=0.8x 且概率>=40%
            elif rsi_ok_long and vol_filter_long and volume_ok:
                signal['signal'] = 'LONG'
                signal['type'] = 'WEAK'
                sl, tp, sl_m, tp_m = get_dynamic_sl_tp(close, atr, volatility, 'WEAK_LONG', breakout_prob)
                signal['sl'] = sl
                signal['tp'] = tp
                signal['sl_mult'] = sl_m
                signal['tp_mult'] = tp_m
                pos, pos_reason = get_dynamic_position(breakout_prob, 'WEAK', vol_ratio, volatility)
                signal['position'] = pos
                signal['position_reason'] = pos_reason
                signal['reason'] = '弱趋势'
        
        # ===== 做空判断 =====
        elif trend_down:
            # 强做空
            if rsi_ok_short and vol_filter_short and breakout_down and volume_strong:
                signal['signal'] = 'SHORT'
                signal['type'] = 'STRONG'
                sl, tp, sl_m, tp_m = get_dynamic_sl_tp(close, atr, volatility, 'STRONG_SHORT', breakout_prob)
                signal['sl'] = sl
                signal['tp'] = tp
                signal['sl_mult'] = sl_m
                signal['tp_mult'] = tp_m
                pos, pos_reason = get_dynamic_position(breakout_prob, 'STRONG', vol_ratio, volatility)
                signal['position'] = pos
                signal['position_reason'] = pos_reason
                signal['reason'] = '强突破'
            
            elif rsi_ok_short and vol_ok and volume_strong:
                signal['signal'] = 'SHORT'
                signal['type'] = 'STRONG'
                sl, tp, sl_m, tp_m = get_dynamic_sl_tp(close, atr, volatility, 'STRONG_SHORT', breakout_prob)
                signal['sl'] = sl
                signal['tp'] = tp
                signal['sl_mult'] = sl_m
                signal['tp_mult'] = tp_m
                pos, pos_reason = get_dynamic_position(breakout_prob, 'STRONG', vol_ratio, volatility)
                signal['position'] = pos
                signal['position_reason'] = pos_reason
                signal['reason'] = '放量趋势'
            
            # 弱做空
            elif rsi_ok_short and vol_filter_short and volume_ok:
                signal['signal'] = 'SHORT'
                signal['type'] = 'WEAK'
                sl, tp, sl_m, tp_m = get_dynamic_sl_tp(close, atr, volatility, 'WEAK_SHORT', breakout_prob)
                signal['sl'] = sl
                signal['tp'] = tp
                signal['sl_mult'] = sl_m
                signal['tp_mult'] = tp_m
                pos, pos_reason = get_dynamic_position(breakout_prob, 'WEAK', vol_ratio, volatility)
                signal['position'] = pos
                signal['position_reason'] = pos_reason
                signal['reason'] = '弱趋势'
        
        signals.append(signal)
    
    # 填充前200个
    empty_signal = {
        'signal': 'HOLD', 'type': '', 'sl': 0, 'tp': 0,
        'sl_mult': 0, 'tp_mult': 0, 'position': 0,
        'position_reason': '', 'breakout_prob': 0, 'reason': ''
    }
    signals = [empty_signal] * 200 + signals
    
    # 添加到DataFrame
    for key in ['signal', 'type', 'sl', 'tp', 'sl_mult', 'tp_mult', 'position', 'position_reason', 'breakout_prob', 'reason']:
        df[key] = [s[key] for s in signals]
    
    return df


# ==================== 回测引擎 ====================
def run_backtest(df, max_bars=30):
    """运行回测 - 考虑仓位权重"""
    print("🔄 运行回测...")
    
    trades = df[df['signal'].isin(['LONG', 'SHORT'])]
    print(f"   发现 {len(trades)} 个交易信号")
    
    if len(trades) == 0:
        return None
    
    results = []
    
    for idx, row in trades.iterrows():
        pos_idx = df.index.get_loc(idx)
        if pos_idx + max_bars >= len(df):
            continue
        
        entry = row['close']
        sl = row['sl']
        tp = row['tp']
        signal = row['signal']
        signal_type = row['type']
        position = row['position']
        breakout_prob = row['breakout_prob']
        reason = row['reason']
        volatility = row['volatility']
        
        if position <= 0:  # 跳过0仓位
            continue
        
        # 模拟交易
        if signal == 'LONG':
            for j in range(pos_idx + 1, min(pos_idx + max_bars + 1, len(df))):
                if df.iloc[j]['low'] <= sl:
                    pnl_raw = (sl - entry) / entry * 100
                    pnl_weighted = pnl_raw * position / 100
                    results.append({
                        'time': row['datetime'],
                        'signal': signal,
                        'type': signal_type,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'exit': sl,
                        'pnl_raw': pnl_raw,
                        'pnl_weighted': pnl_weighted,
                        'position': position,
                        'breakout_prob': breakout_prob,
                        'volatility': volatility,
                        'reason': reason,
                        'result': 'LOSS'
                    })
                    break
                if df.iloc[j]['high'] >= tp:
                    pnl_raw = (tp - entry) / entry * 100
                    pnl_weighted = pnl_raw * position / 100
                    results.append({
                        'time': row['datetime'],
                        'signal': signal,
                        'type': signal_type,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'exit': tp,
                        'pnl_raw': pnl_raw,
                        'pnl_weighted': pnl_weighted,
                        'position': position,
                        'breakout_prob': breakout_prob,
                        'volatility': volatility,
                        'reason': reason,
                        'result': 'WIN'
                    })
                    break
        else:  # SHORT
            for j in range(pos_idx + 1, min(pos_idx + max_bars + 1, len(df))):
                if df.iloc[j]['high'] >= sl:
                    pnl_raw = (entry - sl) / entry * 100
                    pnl_weighted = pnl_raw * position / 100
                    results.append({
                        'time': row['datetime'],
                        'signal': signal,
                        'type': signal_type,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'exit': sl,
                        'pnl_raw': pnl_raw,
                        'pnl_weighted': pnl_weighted,
                        'position': position,
                        'breakout_prob': breakout_prob,
                        'volatility': volatility,
                        'reason': reason,
                        'result': 'LOSS'
                    })
                    break
                if df.iloc[j]['low'] <= tp:
                    pnl_raw = (entry - tp) / entry * 100
                    pnl_weighted = pnl_raw * position / 100
                    results.append({
                        'time': row['datetime'],
                        'signal': signal,
                        'type': signal_type,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'exit': tp,
                        'pnl_raw': pnl_raw,
                        'pnl_weighted': pnl_weighted,
                        'position': position,
                        'breakout_prob': breakout_prob,
                        'volatility': volatility,
                        'reason': reason,
                        'result': 'WIN'
                    })
                    break
    
    return pd.DataFrame(results)


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("📊 ETH 5分钟 历史回测系统 v10.1 (优化版)")
    print("=" * 60)
    
    # 加载数据
    df = load_real_data('/workspace/gdp-dashboard/以太坊合约/ETHUSDT_5m_1y_okx.csv')
    
    # 计算指标
    df = calculate_indicators(df)
    
    # 生成信号
    df = generate_signals_optimized(df)
    
    # 运行回测
    results = run_backtest(df)
    
    if results is None or len(results) == 0:
        print("\n❌ 没有完成的交易")
        return
    
    # ===== 统计结果 =====
    print("\n" + "=" * 60)
    print("📈 回测结果 (v10.1优化版)")
    print("=" * 60)
    
    # 总体统计
    total = len(results)
    wins = len(results[results['result'] == 'WIN'])
    losses = len(results[results['result'] == 'LOSS'])
    win_rate = wins / total * 100 if total > 0 else 0
    
    # 加权盈亏
    total_pnl_weighted = results['pnl_weighted'].sum()
    total_pnl_raw = results['pnl_raw'].sum()
    
    avg_win = results[results['result'] == 'WIN']['pnl_raw'].mean() if wins > 0 else 0
    avg_loss = results[results['result'] == 'LOSS']['pnl_raw'].mean() if losses > 0 else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss) if total > 0 else 0
    
    print(f"\n📊 总体统计:")
    print(f"   交易次数: {total}")
    print(f"   胜/负: {wins}/{losses}")
    print(f"   胜率: {win_rate:.1f}%")
    print(f"   总盈亏(加权): {total_pnl_weighted:.2f}%")
    print(f"   总盈亏(原始): {total_pnl_raw:.2f}%")
    print(f"   平均盈利: {avg_win:.2f}%")
    print(f"   平均亏损: {avg_loss:.2f}%")
    print(f"   期望值: {expectancy:.3f}%")
    print(f"   盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   盈亏比: N/A")
    
    # 按信号类型统计
    strong = results[results['type'] == 'STRONG']
    weak = results[results['type'] == 'WEAK']
    
    if len(strong) > 0:
        strong_wins = len(strong[strong['result'] == 'WIN'])
        print(f"\n🟢 强信号统计:")
        print(f"   交易次数: {len(strong)}")
        print(f"   胜率: {strong_wins/len(strong)*100:.1f}%")
        print(f"   总盈亏(加权): {strong['pnl_weighted'].sum():.2f}%")
        print(f"   平均仓位: {strong['position'].mean():.0f}%")
    
    if len(weak) > 0:
        weak_wins = len(weak[weak['result'] == 'WIN'])
        print(f"\n🟡 弱信号统计:")
        print(f"   交易次数: {len(weak)}")
        print(f"   胜率: {weak_wins/len(weak)*100:.1f}%")
        print(f"   总盈亏(加权): {weak['pnl_weighted'].sum():.2f}%")
        print(f"   平均仓位: {weak['position'].mean():.0f}%")
    
    # 按突破概率统计
    print(f"\n📊 按突破概率统计:")
    for prob_range in [(70, 100), (60, 70), (50, 60), (40, 50)]:
        subset = results[(results['breakout_prob'] >= prob_range[0]) & (results['breakout_prob'] < prob_range[1])]
        if len(subset) > 0:
            wins_sub = len(subset[subset['result'] == 'WIN'])
            print(f"   {prob_range[0]}-{prob_range[1]}%: {len(subset)}笔, 胜率{wins_sub/len(subset)*100:.1f}%, 盈亏{subset['pnl_weighted'].sum():.2f}%")
    
    # 按波动率统计
    print(f"\n📊 按波动率统计:")
    for vol_range in [(0.002, 1), (0.0015, 0.002), (0.001, 0.0015), (0, 0.001)]:
        subset = results[(results['volatility'] >= vol_range[0]) & (results['volatility'] < vol_range[1])]
        if len(subset) > 0:
            wins_sub = len(subset[subset['result'] == 'WIN'])
            vol_pct = f"{vol_range[0]*100:.1f}%-{vol_range[1]*100:.0f}%" if vol_range[1] == 1 else f"{vol_range[0]*100:.2f}%-{vol_range[1]*100:.2f}%"
            print(f"   {vol_pct}: {len(subset)}笔, 胜率{wins_sub/len(subset)*100:.1f}%, 盈亏{subset['pnl_weighted'].sum():.2f}%")
    
    # 月度统计
    results['month'] = pd.to_datetime(results['time']).dt.to_period('M')
    monthly = results.groupby('month').agg({
        'pnl_weighted': ['count', 'sum'],
        'result': lambda x: (x == 'WIN').sum() / len(x) * 100
    }).round(2)
    monthly.columns = ['交易次数', '总盈亏%', '胜率%']
    
    print(f"\n📅 月度统计 (最近6个月):")
    print(monthly.tail(6).to_string())
    
    # 保存结果
    results.to_csv('/workspace/gdp-dashboard/backtest_v101_results.csv', index=False)
    print(f"\n💾 结果已保存到: backtest_v101_results.csv")
    
    return results


if __name__ == "__main__":
    main()
