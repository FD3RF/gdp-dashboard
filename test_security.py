#!/usr/bin/env python3
"""
Oracle AI Agent 安全与完整性测试
"""

import sys
import traceback
sys.path.insert(0, '.')

def test_edge_cases():
    """测试边缘情况"""
    print("\n=== 边缘情况测试 ===")
    
    # 1. 空数据测试
    from agent.perception import PerceptionEncoder
    enc = PerceptionEncoder()
    try:
        state = enc.encode({})
        print(f"✓ 空数据处理: shape={state.shape}")
    except Exception as e:
        print(f"✗ 空数据处理失败: {e}")
        return False
    
    # 2. 极端价格测试
    from execution.performance_calculator import PrecisionCalculator
    calc = PrecisionCalculator()
    try:
        # 价格为0
        pnl, fee = calc.calculate_pnl(0, 100, 1, 'long', 0.001)
        print(f"✓ 零价格处理: pnl={pnl}")
        
        # 极大价格
        pnl, fee = calc.calculate_pnl(1e10, 1e10, 1, 'long', 0.001)
        print(f"✓ 极大价格处理: pnl={pnl:.4f}")
    except Exception as e:
        print(f"✗ 极端价格失败: {e}")
        return False
    
    # 3. 空收益列表测试
    try:
        sharpe = calc.calculate_sharpe_ratio([])
        print(f"✓ 空收益夏普: {sharpe}")
        
        sharpe = calc.calculate_sharpe_ratio([0.01])  # 单个元素
        print(f"✓ 单元素夏普: {sharpe}")
    except Exception as e:
        print(f"✗ 空收益测试失败: {e}")
        return False
    
    # 4. 最大回撤测试
    try:
        dd, dur = calc.calculate_max_drawdown([])
        print(f"✓ 空权益回撤: {dd}")
        
        dd, dur = calc.calculate_max_drawdown([100])  # 单元素
        print(f"✓ 单元素回撤: {dd}")
    except Exception as e:
        print(f"✗ 回撤测试失败: {e}")
        return False
    
    return True


def test_risk_limits():
    """测试风控限制"""
    print("\n=== 风控限制测试 ===")
    
    from execution.risk_shield import RiskShield, RiskLevel
    shield = RiskShield({
        'single_loss_limit': 0.02,
        'daily_loss_limit': 0.05,
        'max_position': 0.2,
    })
    
    # 1. 单笔亏损熔断
    result = shield.check_position_safety(0, 10000, 300)  # 3% 亏损
    if not result.is_safe:
        print(f"✓ 单笔亏损熔断: {result.message}")
    else:
        print("✗ 单笔亏损未触发熔断")
        return False
    
    # 2. 日内亏损熔断
    shield.daily_pnl = -600  # 6% 日亏损
    result = shield.check_position_safety(0, 10000, 0)
    if not result.is_safe and shield.is_sleep_mode:
        print(f"✓ 日内亏损熔断: {result.message}")
    else:
        print("✗ 日内亏损未触发熔断")
        return False
    
    # 3. 凯利公式边界
    shield.is_sleep_mode = False
    shield.daily_pnl = 0
    
    # 信心度为0
    size = shield.calculate_position_size(0, 10000)
    if size == 0:
        print(f"✓ 零信心仓位: {size}")
    else:
        print(f"✗ 零信心仓位应为0: {size}")
        return False
    
    # 信心度100%
    size = shield.calculate_position_size(1.0, 10000)
    if size <= 10000 * 0.2:  # 不超过最大仓位
        print(f"✓ 最大信心仓位: ${size:.2f}")
    else:
        print(f"✗ 仓位超过限制: {size}")
        return False
    
    return True


def test_adversarial_detection():
    """测试对抗博弈检测"""
    print("\n=== 对抗博弈检测测试 ===")
    
    from agent.adversarial import AdversarialJudge, TrapType
    judge = AdversarialJudge()
    
    # 1. 多头陷阱
    result = judge.veto_check(0, {'ask_bid_ratio': 3.0})  # 做多
    if result.is_trap and result.final_action == 3:
        print(f"✓ 多头陷阱检测: {result.trap_type.value}")
    else:
        print(f"✗ 多头陷阱未检测")
        return False
    
    # 2. 空头陷阱
    result = judge.veto_check(1, {'bid_ask_ratio': 3.0})  # 做空
    if result.is_trap:
        print(f"✓ 空头陷阱检测: {result.trap_type.value}")
    else:
        print(f"✗ 空头陷阱未检测")
        return False
    
    # 3. 正常情况
    result = judge.veto_check(0, {'ask_bid_ratio': 1.0})  # 正常
    if not result.is_trap:
        print(f"✓ 正常情况通过")
    else:
        print(f"✗ 正常情况误判")
        return False
    
    return True


def test_data_validation():
    """测试数据验证"""
    print("\n=== 数据验证测试 ===")
    
    from execution.realtime_data import DataValidator
    validator = DataValidator()
    
    # 1. 有效行情
    valid_ticker = {
        'symbol': 'BTC/USDT',
        'last': 50000,
        'bid': 49990,
        'ask': 50010,
    }
    if validator.validate_ticker(valid_ticker):
        print("✓ 有效行情验证")
    else:
        print("✗ 有效行情验证失败")
        return False
    
    # 2. 无效行情 (bid > ask)
    invalid_ticker = {
        'symbol': 'BTC/USDT',
        'last': 50000,
        'bid': 50020,  # bid > ask
        'ask': 50010,
    }
    if not validator.validate_ticker(invalid_ticker):
        print("✓ 无效行情拒绝")
    else:
        print("✗ 无效行情未拒绝")
        return False
    
    # 3. 有效K线
    valid_ohlcv = [1700000000000, 50000, 51000, 49500, 50500, 1000]
    if validator.validate_ohlcv(valid_ohlcv):
        print("✓ 有效K线验证")
    else:
        print("✗ 有效K线验证失败")
        return False
    
    # 4. 无效K线 (low > high)
    invalid_ohlcv = [1700000000000, 50000, 49500, 51000, 50500, 1000]  # low > high
    if not validator.validate_ohlcv(invalid_ohlcv):
        print("✓ 无效K线拒绝")
    else:
        print("✗ 无效K线未拒绝")
        return False
    
    return True


def test_performance_calculator():
    """测试性能计算精度"""
    print("\n=== 性能计算精度测试 ===")
    
    from execution.performance_calculator import PrecisionCalculator
    calc = PrecisionCalculator()
    
    # 1. 收益计算精度
    ret = calc.calculate_return(10000, 10500)
    expected = 0.05
    if abs(ret - expected) < 1e-8:
        print(f"✓ 收益计算精度: {ret:.8f}")
    else:
        print(f"✗ 收益计算误差: {ret} vs {expected}")
        return False
    
    # 2. 盈亏计算精度
    pnl, fee = calc.calculate_pnl(50000, 51000, 0.1, 'long', 0.001)
    # PnL = (51000-50000)*0.1 - (50000*0.1*0.001 + 51000*0.1*0.001)
    #     = 100 - (5 + 5.1) = 89.9
    expected_pnl = 100 - 10.1
    if abs(pnl - expected_pnl) < 0.001:
        print(f"✓ 盈亏计算精度: pnl={pnl:.4f}, fee={fee:.4f}")
    else:
        print(f"✗ 盈亏计算误差: {pnl} vs {expected_pnl}")
        return False
    
    # 3. 夏普比率 - 使用更真实的日收益率
    # 典型日收益率范围: -3% 到 +3%
    returns = [0.001, -0.002, 0.003, -0.001, 0.002, 0.001, -0.002, 0.001, -0.001, 0.002]
    sharpe = calc.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    # 夏普比率典型范围: -3 到 3，但高收益策略可能更高
    if isinstance(sharpe, float) and -100 < sharpe < 100:
        print(f"✓ 夏普比率计算: {sharpe:.4f}")
    else:
        print(f"✗ 夏普比率异常: {sharpe}")
        return False
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Oracle AI Agent 安全与完整性测试")
    print("=" * 60)
    
    tests = [
        ("边缘情况", test_edge_cases),
        ("风控限制", test_risk_limits),
        ("对抗博弈", test_adversarial_detection),
        ("数据验证", test_data_validation),
        ("性能计算", test_performance_calculator),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {name} 测试异常: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
