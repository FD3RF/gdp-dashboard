#!/usr/bin/env python3
"""
Oracle AI Agent 测试脚本
"""

import sys
sys.path.insert(0, '.')

def test_config():
    """测试配置"""
    from config import AGENT_CONFIG, RISK_CONFIG, SYMBOLS, TIMEFRAME
    assert AGENT_CONFIG['state_dim'] == 256
    assert AGENT_CONFIG['action_dim'] == 4
    assert 'max_position' in RISK_CONFIG
    print('✓ 配置模块测试通过')
    return True

def test_perception():
    """测试感知层"""
    from agent.perception import PerceptionEncoder
    encoder = PerceptionEncoder()
    assert encoder.output_dim == 256
    print('✓ 感知层测试通过')
    return True

def test_brain():
    """测试大脑"""
    from agent.brain import PPOBrain
    import torch
    brain = PPOBrain(state_dim=256, action_dim=4)
    
    # 测试前向传播
    state = torch.randn(256)
    action, probs, confidence = brain.decide(state)
    assert 0 <= action <= 3
    assert len(probs) == 4
    print('✓ 大脑测试通过')
    return True

def test_adversarial():
    """测试对抗博弈"""
    from agent.adversarial import AdversarialJudge
    judge = AdversarialJudge()
    
    # 测试陷阱检测
    market_data = {'ask_bid_ratio': 3.0}
    result = judge.veto_check(0, market_data)  # 做多
    assert result.final_action == 3  # 应该被否决为观望
    print('✓ 对抗博弈测试通过')
    return True

def test_strategy():
    """测试策略矩阵"""
    from execution.strategy_matrix import StrategyMatrix
    matrix = StrategyMatrix()
    assert len(matrix.strategies) > 0
    print('✓ 策略矩阵测试通过')
    return True

def test_risk():
    """测试风控"""
    from execution.risk_shield import RiskShield, RiskLevel
    from config import RISK_CONFIG
    shield = RiskShield(RISK_CONFIG)
    
    # 测试仓位计算
    size = shield.calculate_position_size(0.8, 10000)
    assert size > 0
    assert size <= 10000 * RISK_CONFIG['max_position']
    
    # 测试熔断
    result = shield.check_position_safety(0, 10000, 500)  # 5% 亏损
    assert result.is_safe == False  # 应该被熔断
    print('✓ 风控系统测试通过')
    return True

def test_performance():
    """测试性能计算"""
    from execution.performance_calculator import PrecisionCalculator
    calc = PrecisionCalculator()
    
    # 测试精确计算
    pnl, fee = calc.calculate_pnl(50000, 51000, 0.1, 'long', 0.001)
    expected_pnl = (51000 - 50000) * 0.1 - (50000 * 0.1 * 0.001 + 51000 * 0.1 * 0.001)
    assert abs(pnl - expected_pnl) < 0.01
    
    # 测试夏普比率
    returns = [0.01, -0.005, 0.02, -0.01, 0.015]
    sharpe = calc.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    print('✓ 性能计算测试通过')
    return True

def test_agent():
    """测试智能体"""
    from main import OracleAgent
    agent = OracleAgent({'initial_balance': 10000})
    assert agent.balance == 10000
    print('✓ 智能体测试通过')
    return True

def main():
    """运行所有测试"""
    print("=" * 50)
    print("Oracle AI Agent 测试")
    print("=" * 50)
    
    tests = [
        ('配置模块', test_config),
        ('感知层', test_perception),
        ('DRL大脑', test_brain),
        ('对抗博弈', test_adversarial),
        ('策略矩阵', test_strategy),
        ('风控系统', test_risk),
        ('性能计算', test_performance),
        ('智能体', test_agent),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f'✗ {name} 测试失败: {e}')
            failed += 1
    
    print("=" * 50)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 50)
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
