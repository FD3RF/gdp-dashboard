# main.py
"""
Oracle AI Agent - 主程序入口
===========================

神谕级 AI 交易智能体：
- 第2层：全息感知 (256维向量生成)
- 第3层：DRL大脑 + 自我博弈
- 第4层：自适应策略矩阵
- 第5层：反脆弱风控
- 第7层：神经链接仪表盘
"""

import time
import asyncio
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional

from config import AGENT_CONFIG, RISK_CONFIG, SYMBOLS, TIMEFRAME
from agent.perception import PerceptionEncoder, MarketDataCollector
from agent.brain import PPOBrain, PPOTrainer
from agent.adversarial import AdversarialJudge, AdversarialResult
from execution.strategy_matrix import StrategyMatrix, StrategyType
from execution.risk_shield import RiskShield, RiskLevel


class OracleAgent:
    """
    Oracle AI Agent - 神谕智能体
    
    闭环流程：
    1. 全息感知 → 将市场数据编码为256维向量
    2. 大脑决策 → PPO网络输出动作概率
    3. 对抗博弈 → 自我审查，检测陷阱
    4. 策略融合 → 多策略信号融合
    5. 风控审查 → 仓位计算、熔断检查
    6. 执行反馈 → 记录结果用于训练
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # ===== 第2层：全息感知 =====
        self.perception = PerceptionEncoder(
            input_dim=50,
            output_dim=AGENT_CONFIG['state_dim']
        )
        self.data_collector = MarketDataCollector()
        
        # ===== 第3层：DRL大脑 =====
        self.brain = PPOBrain(
            state_dim=AGENT_CONFIG['state_dim'],
            action_dim=AGENT_CONFIG['action_dim'],
            hidden_dim=AGENT_CONFIG['hidden_dim']
        )
        self.trainer = PPOTrainer(self.brain)
        
        # ===== 第3层：自我博弈 =====
        self.adversary = AdversarialJudge()
        
        # ===== 第4层：策略矩阵 =====
        self.strategy_matrix = StrategyMatrix()
        
        # ===== 第5层：反脆弱风控 =====
        self.shield = RiskShield(RISK_CONFIG)
        
        # 账户状态
        self.balance = self.config.get('initial_balance', 10000.0)
        self.position = 0.0
        self.entry_price = 0.0
        
        # 统计
        self.stats = {
            'total_steps': 0,
            'total_decisions': 0,
            'traps_detected': 0,
            'risk_blocks': 0,
        }
        
        # 状态
        self.running = False
        self.market_data = {}
    
    def get_market_data(self) -> Dict[str, Any]:
        """
        获取市场数据
        
        实际项目中应该对接 OKX/Binance API
        这里生成模拟数据用于演示
        """
        # 模拟价格
        base_price = 50000 if not hasattr(self, '_sim_price') else self._sim_price
        change = np.random.randn() * 100
        self._sim_price = base_price + change
        
        # 生成模拟数据
        data = {
            # 价格
            'price': self._sim_price,
            'price_change_24h': np.random.randn() * 0.05,
            'high_24h': self._sim_price * (1 + np.random.rand() * 0.05),
            'low_24h': self._sim_price * (1 - np.random.rand() * 0.05),
            
            # 订单簿
            'orderbook_imbalance': np.random.rand(),
            'bid_volume': np.random.uniform(100, 1000),
            'ask_volume': np.random.uniform(100, 1000),
            'spread_pct': np.random.rand() * 0.001,
            'ask_bid_ratio': np.random.uniform(0.5, 2.0),
            
            # 动量
            'momentum_5m': np.random.randn() * 0.005,
            'momentum_15m': np.random.randn() * 0.01,
            'momentum_1h': np.random.randn() * 0.02,
            
            # 波动率
            'volatility_5m': np.random.rand() * 0.02,
            'volatility_15m': np.random.rand() * 0.03,
            'volatility_1h': np.random.rand() * 0.05,
            
            # 技术指标
            'rsi_14': 30 + np.random.rand() * 40,
            'macd': np.random.randn() * 100,
            'macd_signal': np.random.randn() * 80,
            'macd_hist': np.random.randn() * 50,
            'ma5': self._sim_price * (1 + np.random.randn() * 0.01),
            'ma20': self._sim_price * (1 + np.random.randn() * 0.02),
            'ma60': self._sim_price * (1 + np.random.randn() * 0.03),
            'bb_upper': self._sim_price * 1.02,
            'bb_lower': self._sim_price * 0.98,
            'atr': self._sim_price * 0.01,
            'atr_ratio': np.random.rand() * 0.03,
            
            # 成交量
            'volume_ratio': 0.8 + np.random.rand() * 0.4,
            'trade_flow': np.random.randn(),
            'large_order_ratio': np.random.rand() * 0.3,
            
            # 链上
            'whale_activity_score': np.random.rand(),
            'exchange_inflow': np.random.rand() * 100,
            'exchange_outflow': np.random.rand() * 100,
            
            # 情绪
            'funding_rate': np.random.randn() * 0.0001,
            'sentiment_score': np.random.rand(),
            'fear_greed_index': 20 + np.random.rand() * 60,
            'long_short_ratio': 0.8 + np.random.rand() * 0.4,
            
            # 清算
            'liquidation_long_nearby': np.random.rand(),
            'liquidation_short_nearby': np.random.rand(),
            
            # 时间
            'timestamp': datetime.now().timestamp() * 1000,
        }
        
        return data
    
    def step(self) -> Dict[str, Any]:
        """
        执行一步决策循环
        
        Returns:
            决策结果
        """
        self.stats['total_steps'] += 1
        
        # ===== 1. 全息感知 =====
        self.market_data = self.get_market_data()
        self.data_collector.update(self.market_data)
        state_vector = self.perception.encode(self.market_data)
        
        # ===== 2. 大脑决策 =====
        raw_action, probs, confidence = self.brain.decide(state_vector)
        
        # ===== 3. 对抗博弈 (自我审查) =====
        adversarial_result = self.adversary.veto_check(raw_action, self.market_data, confidence)
        final_action = adversarial_result.final_action
        
        if adversarial_result.is_trap:
            self.stats['traps_detected'] += 1
        
        # ===== 4. 策略信号融合 =====
        strategy_signals = self.strategy_matrix.generate_signals(self.market_data)
        strategy_action, strategy_confidence, strategy_reason = \
            self.strategy_matrix.fuse_signals(strategy_signals)
        
        # 综合决策
        if strategy_confidence > confidence and strategy_action != 3:
            final_action = strategy_action
            confidence = strategy_confidence
        
        # ===== 5. 风控审查 =====
        risk_result = self.shield.check_position_safety(
            action=final_action,
            account_balance=self.balance,
            current_loss=0,
            position_value=abs(self.position * self.market_data['price'])
        )
        
        if not risk_result.is_safe:
            final_action = 3  # 强制观望
            self.stats['risk_blocks'] += 1
        
        # ===== 6. 计算仓位 =====
        position_size = self.shield.calculate_position_size(confidence, self.balance)
        stop_loss = self.shield.calculate_stop_loss(
            self.market_data['price'],
            'long' if final_action == 0 else 'short',
            self.market_data.get('atr', 0)
        )
        take_profit = self.shield.calculate_take_profit(
            self.market_data['price'],
            'long' if final_action == 0 else 'short',
            stop_loss=stop_loss
        )
        
        # ===== 7. 输出结果 =====
        action_map = {0: "LONG", 1: "SHORT", 2: "CLOSE", 3: "HOLD"}
        
        result = {
            'step': self.stats['total_steps'],
            'price': self.market_data['price'],
            'raw_action': action_map[raw_action],
            'final_action': action_map[final_action],
            'confidence': confidence,
            'action_probs': probs,
            'trap_detected': adversarial_result.is_trap,
            'trap_type': adversarial_result.trap_type.value,
            'risk_level': risk_result.level.value,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'balance': self.balance,
            'strategy_signals': [vars(s) if hasattr(s, '__dict__') else s for s in strategy_signals],
        }
        
        self.stats['total_decisions'] += 1
        
        return result
    
    def print_result(self, result: Dict[str, Any]):
        """打印决策结果"""
        print("-" * 60)
        print(f"📊 Step {result['step']} | Price: ${result['price']:,.2f}")
        print(f"🧠 Brain: {result['raw_action']} → {result['final_action']} "
              f"(Confidence: {result['confidence']*100:.1f}%)")
        print(f"⚔️ Adversarial: {'⚠️ TRAP: ' + result['trap_type'] if result['trap_detected'] else '✅ Passed'}")
        print(f"🛡️ Risk: {result['risk_level'].upper()} | Position: ${result['position_size']:,.2f}")
        print(f"📍 Stop Loss: ${result['stop_loss']:,.2f} | Take Profit: ${result['take_profit']:,.2f}")
        print(f"💰 Balance: ${result['balance']:,.2f}")
    
    def run(self, interval: float = 5.0, max_steps: int = None):
        """
        运行智能体
        
        Args:
            interval: 循环间隔（秒）
            max_steps: 最大步数（None 表示无限）
        """
        self.running = True
        step_count = 0
        
        print("=" * 60)
        print("🧠 Oracle AI Agent Initialized")
        print("=" * 60)
        print(f"State Dim: {AGENT_CONFIG['state_dim']}")
        print(f"Action Dim: {AGENT_CONFIG['action_dim']}")
        print(f"Risk Config: Single Loss {RISK_CONFIG['single_loss_limit']*100}%, "
              f"Daily Loss {RISK_CONFIG['daily_loss_limit']*100}%")
        print("=" * 60)
        print("Starting Neural Loop...")
        print()
        
        try:
            while self.running:
                # 执行决策
                result = self.step()
                self.print_result(result)
                
                step_count += 1
                if max_steps and step_count >= max_steps:
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Agent Stopped by User")
        
        # 打印统计
        self.print_statistics()
    
    def stop(self):
        """停止智能体"""
        self.running = False
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("📊 Agent Statistics")
        print("=" * 60)
        print(f"Total Steps: {self.stats['total_steps']}")
        print(f"Total Decisions: {self.stats['total_decisions']}")
        print(f"Traps Detected: {self.stats['traps_detected']}")
        print(f"Risk Blocks: {self.stats['risk_blocks']}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print("=" * 60)
        
        # 对抗博弈统计
        adv_stats = self.adversary.get_statistics()
        print(f"Adversarial Vetoes: {adv_stats['total_vetoes']}")
        
        # 风控统计
        risk_metrics = self.shield.get_risk_metrics()
        print(f"Win Rate: {risk_metrics['win_rate']*100:.1f}%")
        print(f"Max Drawdown: {risk_metrics['max_drawdown']*100:.2f}%")


def main():
    """主函数"""
    # 创建智能体
    agent = OracleAgent({
        'initial_balance': 10000.0,
    })
    
    # 运行
    agent.run(interval=5.0)


if __name__ == "__main__":
    main()


# 导出
__all__ = ['OracleAgent']
