#!/usr/bin/env python3
"""
全面测试脚本
测试所有模块的正确性和精度
"""

import sys
import traceback

def test_section(name):
    """测试装饰器"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"【{name}】")
            print('='*60)
            try:
                func()
                print(f"✓ {name} 测试通过")
                return True
            except Exception as e:
                print(f"✗ {name} 测试失败: {e}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


@test_section("1. 性能计算精度")
def test_performance():
    import numpy as np
    from execution.performance_calculator import PrecisionCalculator
    
    calc = PrecisionCalculator()
    
    # 测试盈亏计算
    pnl, fee = calc.calculate_pnl(50000, 51000, 0.1, 'long', 0.001)
    expected_pnl = (51000 - 50000) * 0.1 - 50000 * 0.1 * 0.001 - 51000 * 0.1 * 0.001
    assert abs(pnl - expected_pnl) < 0.01, f"盈亏计算误差: {pnl} vs {expected_pnl}"
    print(f"  盈亏计算: PnL={pnl:.6f}, Fee={fee:.6f}")
    
    # 测试夏普比率
    returns = [0.001, -0.002, 0.003, -0.001, 0.002]
    sharpe = calc.calculate_sharpe_ratio(returns)
    assert -100 < sharpe < 100, f"夏普比率异常: {sharpe}"
    print(f"  夏普比率: {sharpe:.4f}")
    
    # 测试最大回撤
    equity = [100000, 102000, 99000, 103000, 95000, 98000]
    mdd_result = calc.calculate_max_drawdown(equity)
    mdd = mdd_result[0] if isinstance(mdd_result, tuple) else mdd_result
    assert 0 <= mdd <= 1, f"最大回撤异常: {mdd}"
    print(f"  最大回撤: {mdd*100:.2f}%")


@test_section("2. 技术指标计算")
def test_indicators():
    import numpy as np
    import pandas as pd
    from data.kline_builder import calculate_indicators
    
    np.random.seed(42)
    n = 200
    prices = 50000 * np.cumprod(1 + np.random.normal(0.0001, 0.02, n))
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-01', periods=n, freq='5min'),
        'open': prices,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })
    
    df = calculate_indicators(df)
    last = df.iloc[-1]
    
    # 验证RSI范围
    assert 0 <= last['rsi14'] <= 100, f"RSI范围错误: {last['rsi14']}"
    print(f"  RSI: {last['rsi14']:.2f}")
    
    # 验证布林带
    assert last['bb_upper'] > last['bb_lower'], "布林带顺序错误"
    print(f"  布林带: 上轨={last['bb_upper']:.2f}, 下轨={last['bb_lower']:.2f}")
    
    # 验证ATR
    assert last['atr14'] > 0, "ATR应该大于0"
    print(f"  ATR: {last['atr14']:.2f}")


@test_section("3. Hurst指数和分形维度")
def test_hurst_fractal():
    import numpy as np
    from features.hurst import HurstExponent
    from features.fractal import FractalDimension
    
    # 趋势数据
    trend = 50000 + np.cumsum(np.random.normal(0.5, 1, 200))
    
    h = HurstExponent()
    h_val, h_state = h.calculate(trend)
    assert 0 < h_val < 1, f"Hurst范围错误: {h_val}"
    print(f"  Hurst: {h_val:.3f} ({h_state})")
    
    fd = FractalDimension()
    d_val, d_state = fd.calculate(trend)
    assert 1 <= d_val <= 2, f"分形维度范围错误: {d_val}"
    print(f"  分形维度: {d_val:.3f} ({d_state})")


@test_section("4. 订单簿分析")
def test_orderbook():
    from features.orderbook_features import OrderbookAnalyzer
    
    orderbook = {
        'bids': [[50000 - i*10, 1 + i*0.1] for i in range(20)],
        'asks': [[50000 + i*10, 1 + i*0.05] for i in range(20)]
    }
    
    analyzer = OrderbookAnalyzer()
    result = analyzer.analyze(orderbook)
    
    assert -1 <= result.imbalance <= 1, f"失衡度范围错误: {result.imbalance}"
    print(f"  失衡度: {result.imbalance:.4f}")
    
    assert 0 <= result.quality_score <= 100, f"信号质量范围错误: {result.quality_score}"
    print(f"  信号质量: {result.quality_score:.1f}")


@test_section("5. 流动性热图")
def test_liquidity():
    from features.liquidity_heatmap import generate_liquidity_heatmap
    
    orderbook = {
        'bids': [[3500 - i, 1] for i in range(20)] + [[3400, 50]],
        'asks': [[3500 + i, 1] for i in range(1, 21)] + [[3600, 30]]
    }
    
    result = generate_liquidity_heatmap(orderbook, 3500)
    
    assert 'liquidity_score' in result
    print(f"  流动性得分: {result['liquidity_score']:.1f}")
    
    assert 'support_zones' in result
    print(f"  支撑区数量: {len(result['support_zones'])}")


@test_section("6. 资金费率分析")
def test_funding():
    from features.funding_rate import analyze_funding_rate
    
    test_rates = [0.0001, -0.0005, 0.002, -0.001]
    for rate in test_rates:
        result = analyze_funding_rate(rate)
        print(f"  费率 {rate*100:.4f}%: {result['signal']} ({result['warning_level']})")


@test_section("7. AI概率模型")
def test_probability():
    import numpy as np
    import pandas as pd
    from ai.probability_model import calculate_probabilities
    from data.kline_builder import calculate_indicators
    
    np.random.seed(42)
    n = 200
    prices = 3500 * np.cumprod(1 + np.random.normal(0.0001, 0.015, n))
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-01', periods=n, freq='5min'),
        'open': prices, 'high': prices*1.005, 'low': prices*0.995,
        'close': prices, 'volume': np.random.uniform(100, 1000, n)
    })
    
    df = calculate_indicators(df)
    probs = calculate_probabilities(df)
    
    total = probs['long'] + probs['short'] + probs['hold']
    assert abs(total - 100) < 0.1, f"概率和不等于100: {total}"
    
    print(f"  做多: {probs['long']:.1f}%")
    print(f"  做空: {probs['short']:.1f}%")
    print(f"  观望: {probs['hold']:.1f}%")
    print(f"  信心: {probs['confidence']:.1f}")


@test_section("8. 决策逻辑")
def test_decision():
    from analysis.decision_maker import make_decision
    
    # 测试做多信号
    probs1 = {'long': 70, 'short': 20, 'hold': 10, 'confidence': 50, 'signals': {'reasons': []}}
    decision1 = make_decision(probs1)
    assert decision1.signal.value == 'LONG', f"决策错误: {decision1.signal.value}"
    print(f"  多70%: {decision1.signal.value}")
    
    # 测试观望信号
    probs2 = {'long': 25, 'short': 25, 'hold': 50, 'confidence': 25, 'signals': {'reasons': []}}
    decision2 = make_decision(probs2)
    assert decision2.signal.value == 'HOLD', f"决策错误: {decision2.signal.value}"
    print(f"  观望50%: {decision2.signal.value}")


@test_section("9. 信号解释")
def test_explanation():
    from explain.signal_explainer import explain_signal
    
    explanation = explain_signal(
        'LONG',
        {'confidence': 50},
        {'rsi14': 30, 'ma5': 3500, 'ma20': 3450, 'close': 3550}
    )
    
    assert 'primary_source' in explanation
    assert 'market_structure' in explanation
    print(f"  主要来源: {explanation['primary_source']}")
    print(f"  市场结构: {explanation['market_structure']}")


@test_section("10. 完整系统集成")
def test_full_system():
    from main import run_system
    
    state = run_system('ETH/USDT', use_simulated=True)
    
    assert state is not None, "系统返回None"
    assert state.price > 0, "价格应该大于0"
    assert state.signal is not None, "信号不应为空"
    
    print(f"  价格: ${state.price:,.2f}")
    print(f"  信号: {state.signal.value}")
    print(f"  信心: {state.probabilities['confidence']:.1f}")
    print(f"  Hurst: {state.hurst:.3f}")


def main():
    print("\n" + "="*60)
    print("🧪 全面性能和数据测试")
    print("="*60)
    
    tests = [
        test_performance,
        test_indicators,
        test_hurst_fractal,
        test_orderbook,
        test_liquidity,
        test_funding,
        test_probability,
        test_decision,
        test_explanation,
        test_full_system,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
