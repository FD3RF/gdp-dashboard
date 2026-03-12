# agent/adversarial.py
"""
第3层：自我博弈机制
==================

功能：模拟庄家思维，过滤掉会被猎杀的"愚蠢决策"
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TrapType(Enum):
    """陷阱类型"""
    NONE = 'none'
    FAKE_BREAKOUT = 'fake_breakout'      # 假突破
    BEAR_TRAP = 'bear_trap'              # 空头陷阱
    BULL_TRAP = 'bull_trap'              # 多头陷阱
    STOP_HUNT = 'stop_hunt'              # 扫损
    WHALE_WASH = 'whale_wash'            # 鲸鱼洗盘
    LIQUIDATION_CASCADE = 'liquidation_cascade'  # 清算瀑布


@dataclass
class AdversarialResult:
    """对抗博弈结果"""
    original_action: int
    final_action: int
    is_trap: bool
    trap_type: TrapType
    confidence: float
    reason: str


class AdversarialJudge:
    """
    对抗博弈判断器
    
    模拟庄家思维，检测市场陷阱：
    1. 订单簿陷阱检测
    2. 假突破检测
    3. 扫损检测
    4. 清算瀑布检测
    5. 鲸鱼行为分析
    """
    
    # 动作定义
    ACTION_LONG = 0    # 开多
    ACTION_SHORT = 1   # 开空
    ACTION_CLOSE = 2   # 平仓
    ACTION_HOLD = 3    # 观望
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 陷阱检测阈值
        self.thresholds = {
            'orderbook_trap_ratio': 2.0,      # 订单簿买卖比阈值
            'volume_spike_ratio': 3.0,         # 成交量异常阈值
            'liquidation_threshold': 0.8,      # 清算风险阈值
            'whale_activity_threshold': 0.7,   # 鲸鱼活动阈值
            'stop_hunt_range': 0.01,           # 扫损范围 (1%)
        }
        
        # 历史记录
        self.trap_history: List[Dict] = []
        self.veto_count = 0
    
    def simulate_trap(self, proposed_action: int, market_data: Dict[str, Any]) -> Tuple[bool, TrapType]:
        """
        模拟庄家陷阱：检测当前决策是否可能落入陷阱
        
        Args:
            proposed_action: 提议的动作
            market_data: 市场数据
            
        Returns:
            (是否陷阱, 陷阱类型)
        """
        # 1. 订单簿陷阱检测
        if proposed_action == self.ACTION_LONG:
            liquidity_ratio = market_data.get('ask_bid_ratio', 1.0)
            if liquidity_ratio > self.thresholds['orderbook_trap_ratio']:
                return True, TrapType.BULL_TRAP
        
        elif proposed_action == self.ACTION_SHORT:
            liquidity_ratio = market_data.get('bid_ask_ratio', 1.0)
            if liquidity_ratio > self.thresholds['orderbook_trap_ratio']:
                return True, TrapType.BEAR_TRAP
        
        # 2. 假突破检测
        if self._detect_fake_breakout(proposed_action, market_data):
            return True, TrapType.FAKE_BREAKOUT
        
        # 3. 扫损检测
        if self._detect_stop_hunt(proposed_action, market_data):
            return True, TrapType.STOP_HUNT
        
        # 4. 清算瀑布检测
        if self._detect_liquidation_cascade(proposed_action, market_data):
            return True, TrapType.LIQUIDATION_CASCADE
        
        # 5. 鲸鱼洗盘检测
        if self._detect_whale_wash(proposed_action, market_data):
            return True, TrapType.WHALE_WASH
        
        return False, TrapType.NONE
    
    def _detect_fake_breakout(self, action: int, data: Dict) -> bool:
        """检测假突破"""
        # 成交量突然放大但价格未延续
        volume_ratio = data.get('volume_ratio', 1.0)
        price_momentum = data.get('momentum_5m', 0)
        
        if volume_ratio > self.thresholds['volume_spike_ratio']:
            # 成交量放大但动量不足
            if action == self.ACTION_LONG and price_momentum < 0.001:
                return True
            if action == self.ACTION_SHORT and price_momentum > -0.001:
                return True
        
        return False
    
    def _detect_stop_hunt(self, action: int, data: Dict) -> bool:
        """检测扫损"""
        price = data.get('price', 0)
        low_24h = data.get('low_24h', price)
        high_24h = data.get('high_24h', price)
        
        if high_24h <= 0 or low_24h <= 0:
            return False
        
        # 价格接近近期高点/低点
        if action == self.ACTION_LONG:
            # 检查是否在扫低点
            price_range = high_24h - low_24h
            if price_range > 0:
                distance_to_low = (price - low_24h) / price_range
                if distance_to_low < self.thresholds['stop_hunt_range']:
                    return True
        
        elif action == self.ACTION_SHORT:
            # 检查是否在扫高点
            price_range = high_24h - low_24h
            if price_range > 0:
                distance_to_high = (high_24h - price) / price_range
                if distance_to_high < self.thresholds['stop_hunt_range']:
                    return True
        
        return False
    
    def _detect_liquidation_cascade(self, action: int, data: Dict) -> bool:
        """检测清算瀑布"""
        # 大量多头/空头接近清算价
        liquidation_long = data.get('liquidation_long_nearby', 0)
        liquidation_short = data.get('liquidation_short_nearby', 0)
        
        if action == self.ACTION_LONG:
            # 如果大量多头即将清算，可能被做空者利用
            if liquidation_long > self.thresholds['liquidation_threshold']:
                return True
        
        elif action == self.ACTION_SHORT:
            if liquidation_short > self.thresholds['liquidation_threshold']:
                return True
        
        return False
    
    def _detect_whale_wash(self, action: int, data: Dict) -> bool:
        """检测鲸鱼洗盘"""
        whale_score = data.get('whale_activity_score', 0)
        
        if whale_score > self.thresholds['whale_activity_threshold']:
            # 鲸鱼活跃时，跟随大单可能被反向操作
            large_order_direction = data.get('large_order_direction', 0)
            if action == self.ACTION_LONG and large_order_direction < 0:
                return True
            if action == self.ACTION_SHORT and large_order_direction > 0:
                return True
        
        return False
    
    def veto_check(
        self, 
        action: int, 
        market_data: Dict[str, Any],
        confidence: float = 0.5
    ) -> AdversarialResult:
        """
        对抗博弈检查
        
        Args:
            action: 提议的动作
            market_data: 市场数据
            confidence: 决策信心
            
        Returns:
            对抗博弈结果
        """
        is_trap, trap_type = self.simulate_trap(action, market_data)
        
        if is_trap:
            # 记录否决
            self.veto_count += 1
            self.trap_history.append({
                'original_action': action,
                'trap_type': trap_type.value,
                'market_data': market_data,
            })
            
            # 根据陷阱类型决定最终动作
            if trap_type in [TrapType.BULL_TRAP, TrapType.BEAR_TRAP]:
                final_action = self.ACTION_HOLD
                reason = f"检测到{trap_type.value}陷阱，建议观望"
            elif trap_type == TrapType.FAKE_BREAKOUT:
                final_action = self.ACTION_HOLD
                reason = "检测到假突破，建议观望"
            elif trap_type == TrapType.STOP_HUNT:
                # 扫损后可能反弹，反向操作
                final_action = self.ACTION_SHORT if action == self.ACTION_LONG else self.ACTION_LONG
                reason = "检测到扫损，建议反向操作"
            elif trap_type == TrapType.LIQUIDATION_CASCADE:
                final_action = self.ACTION_HOLD
                reason = "清算瀑布风险，建议观望"
            elif trap_type == TrapType.WHALE_WASH:
                final_action = self.ACTION_HOLD
                reason = "鲸鱼洗盘中，建议观望"
            else:
                final_action = self.ACTION_HOLD
                reason = "未知陷阱，建议观望"
        else:
            final_action = action
            reason = "通过对抗博弈检查"
        
        return AdversarialResult(
            original_action=action,
            final_action=final_action,
            is_trap=is_trap,
            trap_type=trap_type,
            confidence=confidence,
            reason=reason
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        trap_types = {}
        for record in self.trap_history:
            t = record['trap_type']
            trap_types[t] = trap_types.get(t, 0) + 1
        
        return {
            'total_vetoes': self.veto_count,
            'trap_history_count': len(self.trap_history),
            'trap_types': trap_types,
        }
    
    def reset_statistics(self):
        """重置统计"""
        self.trap_history = []
        self.veto_count = 0


class MarketMakerSimulator:
    """做市商模拟器"""
    
    def __init__(self):
        self.position = 0.0
        self.avg_price = 0.0
        self.pnl = 0.0
    
    def simulate_market_impact(
        self, 
        order_size: float, 
        orderbook: Dict
    ) -> float:
        """
        模拟订单对市场的影响
        
        Args:
            order_size: 订单大小
            orderbook: 订单簿数据
            
        Returns:
            预计滑点
        """
        # 简化模型：根据订单簿深度估算滑点
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if order_size > 0:  # 买入
            depth = sum(level[1] for level in asks[:5])
        else:  # 卖出
            depth = sum(level[1] for level in bids[:5])
        
        if depth <= 0:
            return 0
        
        # 滑点与订单大小/深度成正比
        slippage = abs(order_size) / depth * 0.001
        return min(slippage, 0.05)  # 最大5%
    
    def predict_price_movement(
        self, 
        market_data: Dict, 
        horizon: int = 5
    ) -> Tuple[float, float]:
        """
        预测价格变动
        
        Args:
            market_data: 市场数据
            horizon: 预测周期（分钟）
            
        Returns:
            (预期变动方向, 预测置信度)
        """
        # 简化模型：基于订单簿不平衡和动量
        ob_imbalance = market_data.get('orderbook_imbalance', 0.5)
        momentum = market_data.get('momentum_5m', 0)
        
        # 预测方向
        signal = (ob_imbalance - 0.5) * 2 + momentum * 10
        
        if signal > 0.1:
            direction = 1  # 上涨
            confidence = min(abs(signal), 1.0)
        elif signal < -0.1:
            direction = -1  # 下跌
            confidence = min(abs(signal), 1.0)
        else:
            direction = 0  # 震荡
            confidence = 0.5
        
        return direction, confidence


# 导出
__all__ = [
    'AdversarialJudge',
    'AdversarialResult',
    'TrapType',
    'MarketMakerSimulator'
]
