# main.py
"""
Oracle AI Agent - 主程序入口
===========================

神谕级 AI 交易智能体：
- 第2层：全息感知 (256维向量生成)
- 第3层：DRL大脑 + 自我博弈
- 第4层：自适应策略矩阵
- 第5层：反脆弱风控
- 实时数据获取
- 精确性能计算
"""

import time
import asyncio
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional

from config import AGENT_CONFIG, RISK_CONFIG, SYMBOLS, TIMEFRAME, EXCHANGE_CONFIG
from agent.perception import PerceptionEncoder, MarketDataCollector
from agent.brain import PPOBrain, PPOTrainer
from agent.adversarial import AdversarialJudge, AdversarialResult
from execution.strategy_matrix import StrategyMatrix, StrategyType
from execution.risk_shield import RiskShield, RiskLevel
from execution.performance_calculator import RealtimePerformanceTracker, PrecisionCalculator
from execution.realtime_data import RealtimeDataFeed, DataAggregator


class OracleAgent:
    """
    Oracle AI Agent - 神谕智能体
    
    闭环流程：
    1. 实时数据获取 → 从交易所获取精确数据
    2. 全息感知 → 将市场数据编码为256维向量
    3. 大脑决策 → PPO网络输出动作概率
    4. 对抗博弈 → 自我审查，检测陷阱
    5. 策略融合 → 多策略信号融合
    6. 风控审查 → 仓位计算、熔断检查
    7. 性能追踪 → 精确计算收益和风险指标
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # ===== 实时数据源 =====
        self.data_feed = RealtimeDataFeed(EXCHANGE_CONFIG)
        self.data_aggregator = DataAggregator()
        
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
        
        # ===== 性能追踪 =====
        initial_balance = self.config.get('initial_balance', 10000.0)
        self.performance_tracker = RealtimePerformanceTracker(initial_balance)
        self.calculator = PrecisionCalculator()
        
        # 账户状态
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        
        # 统计
        self.stats = {
            'total_steps': 0,
            'total_decisions': 0,
            'traps_detected': 0,
            'risk_blocks': 0,
            'data_errors': 0,
        }
        
        # 状态
        self.running = False
        self.market_data = {}
        self._connected = False
    
    async def connect(self) -> bool:
        """连接交易所"""
        try:
            self._connected = await self.data_feed.connect()
            if self._connected:
                print(f"✓ 已连接交易所")
            else:
                print(f"⚠ 使用模拟数据")
            return True
        except Exception as e:
            print(f"⚠ 数据源连接失败: {e}, 使用模拟数据")
            return True
    
    async def disconnect(self):
        """断开连接"""
        await self.data_feed.disconnect()
    
    async def get_realtime_data(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时数据
        
        优先从交易所获取，失败则使用模拟数据
        """
        data = {}
        
        # 尝试获取实时数据
        if self._connected:
            try:
                # 行情
                ticker = await self.data_feed.fetch_ticker(symbol)
                if ticker:
                    data.update({
                        'price': ticker.last,
                        'bid': ticker.bid,
                        'ask': ticker.ask,
                        'volume': ticker.volume,
                    })
                
                # K线
                ohlcv = await self.data_feed.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
                if ohlcv:
                    closes = [o.close for o in ohlcv]
                    
                    # 计算技术指标
                    data.update({
                        'open': ohlcv[-1].open,
                        'high': ohlcv[-1].high,
                        'low': ohlcv[-1].low,
                        'close': ohlcv[-1].close,
                        'volume': ohlcv[-1].volume,
                        
                        # 均线
                        'ma5': self.data_aggregator.calculate_sma(closes, 5)[-1] if len(closes) >= 5 else closes[-1],
                        'ma20': self.data_aggregator.calculate_sma(closes, 20)[-1] if len(closes) >= 20 else closes[-1],
                        'ma60': self.data_aggregator.calculate_sma(closes, 60)[-1] if len(closes) >= 60 else closes[-1],
                        
                        # RSI
                        'rsi_14': self.data_aggregator.calculate_rsi(closes, 14)[-1] if len(closes) >= 15 else 50,
                    })
                    
                    # MACD
                    macd_data = self.data_aggregator.calculate_macd(closes)
                    if macd_data['macd']:
                        data.update({
                            'macd': macd_data['macd'][-1],
                            'macd_signal': macd_data['signal'][-1] if macd_data['signal'] else 0,
                            'macd_hist': macd_data['histogram'][-1] if macd_data['histogram'] else 0,
                        })
                
                # 订单簿
                orderbook = await self.data_feed.fetch_orderbook(symbol)
                if orderbook:
                    bids = orderbook.get('bids', [])
                    asks = orderbook.get('asks', [])
                    bid_volume = sum(b[1] for b in bids[:5]) if bids else 0
                    ask_volume = sum(a[1] for a in asks[:5]) if asks else 0
                    
                    data.update({
                        'orderbook_imbalance': bid_volume / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.5,
                        'bid_volume': bid_volume,
                        'ask_volume': ask_volume,
                        'spread_pct': (asks[0][0] - bids[0][0]) / bids[0][0] if bids and asks else 0,
                    })
                
            except Exception as e:
                self.stats['data_errors'] += 1
        
        # 如果实时数据不足，补充模拟数据
        data = self._fill_simulated_data(data)
        
        return data
    
    def _fill_simulated_data(self, data: Dict) -> Dict:
        """填充模拟数据"""
        # 基础价格
        base_price = data.get('price', 50000 + np.random.randn() * 1000)
        
        # 填充缺失字段
        defaults = {
            'price': base_price,
            'price_change_24h': np.random.randn() * 0.05,
            'high_24h': base_price * (1 + np.random.rand() * 0.05),
            'low_24h': base_price * (1 - np.random.rand() * 0.05),
            
            'orderbook_imbalance': np.random.rand(),
            'bid_volume': np.random.uniform(100, 1000),
            'ask_volume': np.random.uniform(100, 1000),
            'spread_pct': np.random.rand() * 0.001,
            'ask_bid_ratio': np.random.uniform(0.5, 2.0),
            
            'momentum_5m': np.random.randn() * 0.005,
            'momentum_15m': np.random.randn() * 0.01,
            'momentum_1h': np.random.randn() * 0.02,
            
            'volatility_5m': np.random.rand() * 0.02,
            'volatility_15m': np.random.rand() * 0.03,
            'volatility_1h': np.random.rand() * 0.05,
            
            'rsi_14': 30 + np.random.rand() * 40,
            'macd': np.random.randn() * 100,
            'macd_signal': np.random.randn() * 80,
            'macd_hist': np.random.randn() * 50,
            'ma5': base_price * (1 + np.random.randn() * 0.01),
            'ma20': base_price * (1 + np.random.randn() * 0.02),
            'ma60': base_price * (1 + np.random.randn() * 0.03),
            'bb_upper': base_price * 1.02,
            'bb_lower': base_price * 0.98,
            'atr': base_price * 0.01,
            'atr_ratio': np.random.rand() * 0.03,
            
            'volume_ratio': 0.8 + np.random.rand() * 0.4,
            'trade_flow': np.random.randn(),
            'large_order_ratio': np.random.rand() * 0.3,
            
            'whale_activity_score': np.random.rand(),
            'exchange_inflow': np.random.rand() * 100,
            'exchange_outflow': np.random.rand() * 100,
            
            'funding_rate': np.random.randn() * 0.0001,
            'sentiment_score': np.random.rand(),
            'fear_greed_index': 20 + np.random.rand() * 60,
            'long_short_ratio': 0.8 + np.random.rand() * 0.4,
            
            'liquidation_long_nearby': np.random.rand(),
            'liquidation_short_nearby': np.random.rand(),
            
            'timestamp': datetime.now().timestamp() * 1000,
        }
        
        for key, value in defaults.items():
            if key not in data or data[key] is None:
                data[key] = value
        
        return data
    
    async def step(self, symbol: str = 'BTC/USDT') -> Dict[str, Any]:
        """
        执行一步决策循环
        
        Args:
            symbol: 交易对
            
        Returns:
            决策结果
        """
        self.stats['total_steps'] += 1
        
        # ===== 1. 获取实时数据 =====
        self.market_data = await self.get_realtime_data(symbol)
        self.data_collector.update(self.market_data)
        
        # ===== 2. 全息感知 =====
        state_vector = self.perception.encode(self.market_data)
        
        # ===== 3. 大脑决策 =====
        raw_action, probs, confidence = self.brain.decide(state_vector)
        
        # ===== 4. 对抗博弈 (自我审查) =====
        adversarial_result = self.adversary.veto_check(raw_action, self.market_data, confidence)
        final_action = adversarial_result.final_action
        
        if adversarial_result.is_trap:
            self.stats['traps_detected'] += 1
        
        # ===== 5. 策略信号融合 =====
        strategy_signals = self.strategy_matrix.generate_signals(self.market_data)
        strategy_action, strategy_confidence, strategy_reason = \
            self.strategy_matrix.fuse_signals(strategy_signals)
        
        # 综合决策
        if strategy_confidence > confidence and strategy_action != 3:
            final_action = strategy_action
            confidence = strategy_confidence
        
        # ===== 6. 风控审查 =====
        risk_result = self.shield.check_position_safety(
            action=final_action,
            account_balance=self.balance,
            current_loss=0,
            position_value=abs(self.position * self.market_data['price'])
        )
        
        if not risk_result.is_safe:
            final_action = 3  # 强制观望
            self.stats['risk_blocks'] += 1
        
        # ===== 7. 计算仓位 =====
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
        
        # ===== 8. 更新性能追踪 =====
        self.performance_tracker.update_equity(self.balance)
        
        # ===== 9. 输出结果 =====
        action_map = {0: "LONG", 1: "SHORT", 2: "CLOSE", 3: "HOLD"}
        
        result = {
            'step': self.stats['total_steps'],
            'symbol': symbol,
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
            'performance': self.performance_tracker.get_metrics(),
        }
        
        self.stats['total_decisions'] += 1
        
        return result
    
    def print_result(self, result: Dict[str, Any]):
        """打印决策结果"""
        print("-" * 70)
        print(f"📊 Step {result['step']} | {result['symbol']} | ${result['price']:,.2f}")
        print(f"🧠 Brain: {result['raw_action']} → {result['final_action']} "
              f"(Confidence: {result['confidence']*100:.1f}%)")
        print(f"⚔️ Adversarial: {'⚠️ TRAP: ' + result['trap_type'] if result['trap_detected'] else '✅ Passed'}")
        print(f"🛡️ Risk: {result['risk_level'].upper()} | Position: ${result['position_size']:,.2f}")
        print(f"📍 Stop Loss: ${result['stop_loss']:,.2f} | Take Profit: ${result['take_profit']:,.2f}")
        print(f"💰 Balance: ${result['balance']:,.2f}")
        
        # 性能指标
        perf = result.get('performance')
        if perf:
            print(f"📈 Total Return: {perf.total_return*100:.2f}% | "
                  f"Win Rate: {perf.win_rate*100:.1f}% | "
                  f"Sharpe: {perf.sharpe_ratio:.2f}")
    
    async def run(self, interval: float = 5.0, max_steps: int = None, symbol: str = 'BTC/USDT'):
        """
        运行智能体
        
        Args:
            interval: 循环间隔（秒）
            max_steps: 最大步数（None 表示无限）
            symbol: 交易对
        """
        self.running = True
        step_count = 0
        
        # 连接数据源
        await self.connect()
        
        print("=" * 70)
        print("🧠 Oracle AI Agent Initialized")
        print("=" * 70)
        print(f"State Dim: {AGENT_CONFIG['state_dim']}")
        print(f"Action Dim: {AGENT_CONFIG['action_dim']}")
        print(f"Risk Config: Single Loss {RISK_CONFIG['single_loss_limit']*100}%, "
              f"Daily Loss {RISK_CONFIG['daily_loss_limit']*100}%")
        print(f"Symbol: {symbol} | Timeframe: {TIMEFRAME}")
        print("=" * 70)
        print("Starting Neural Loop...")
        print()
        
        try:
            while self.running:
                # 执行决策
                result = await self.step(symbol)
                self.print_result(result)
                
                step_count += 1
                if max_steps and step_count >= max_steps:
                    break
                
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️ Agent Stopped by User")
        finally:
            await self.disconnect()
        
        # 打印统计
        self.print_statistics()
    
    def stop(self):
        """停止智能体"""
        self.running = False
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("📊 Agent Statistics")
        print("=" * 70)
        print(f"Total Steps: {self.stats['total_steps']}")
        print(f"Total Decisions: {self.stats['total_decisions']}")
        print(f"Traps Detected: {self.stats['traps_detected']}")
        print(f"Risk Blocks: {self.stats['risk_blocks']}")
        print(f"Data Errors: {self.stats['data_errors']}")
        print(f"Final Balance: ${self.balance:,.2f}")
        print("=" * 70)
        
        # 性能指标
        metrics = self.performance_tracker.get_metrics()
        print(f"Total Return: {metrics.total_return*100:.2f}%")
        print(f"Win Rate: {metrics.win_rate*100:.1f}%")
        print(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print("=" * 70)
        
        # 对抗博弈统计
        adv_stats = self.adversary.get_statistics()
        print(f"Adversarial Vetoes: {adv_stats['total_vetoes']}")


async def main():
    """主函数"""
    # 创建智能体
    agent = OracleAgent({
        'initial_balance': 10000.0,
    })
    
    # 运行
    await agent.run(interval=5.0, symbol='BTC/USDT')


if __name__ == "__main__":
    asyncio.run(main())


# 导出
__all__ = ['OracleAgent']
