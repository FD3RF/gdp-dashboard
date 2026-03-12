# features/orderbook_features.py
"""
模块 15: 订单簿失衡分析
=======================
计算买卖盘力量对比，识别"虚假托单"
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """订单类型"""
    SUPPORT = "支撑墙"
    RESISTANCE = "阻力墙"
    FAKE_BID = "虚假托单"
    FAKE_ASK = "虚假压单"
    ICEBERG = "冰山单"
    NORMAL = "正常"


@dataclass
class OrderbookLevel:
    """订单簿层级"""
    price: float
    volume: float
    cumulative_volume: float = 0.0


@dataclass
class OrderbookAnalysis:
    """订单簿分析结果"""
    bid_volume: float
    ask_volume: float
    imbalance: float  # -1 到 1
    spread: float
    spread_percent: float
    mid_price: float
    detected_walls: List[Dict[str, Any]]
    fake_orders: List[Dict[str, Any]]
    support_levels: List[float]
    resistance_levels: List[float]
    quality_score: float  # 信号质量评分 0-100


class OrderbookAnalyzer:
    """
    订单簿分析器
    
    功能：
    - 计算买卖盘失衡度
    - 识别支撑/阻力墙
    - 检测虚假托单
    - 发现冰山单
    """
    
    def __init__(self, wall_threshold: float = 3.0, fake_ratio: float = 0.1):
        """
        Args:
            wall_threshold: 墙判定阈值 (相对平均量的倍数)
            fake_ratio: 虚假订单判定比例
        """
        self.wall_threshold = wall_threshold
        self.fake_ratio = fake_ratio
    
    def analyze(self, orderbook: Dict[str, List]) -> OrderbookAnalysis:
        """
        分析订单簿
        
        Args:
            orderbook: {'bids': [[price, volume]], 'asks': [[price, volume]]}
            
        Returns:
            OrderbookAnalysis
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return self._empty_analysis()
        
        # 计算基础数据
        bid_volumes = [float(b[1]) for b in bids[:20]]
        ask_volumes = [float(a[1]) for a in asks[:20]]
        bid_prices = [float(b[0]) for b in bids[:20]]
        ask_prices = [float(a[0]) for a in asks[:20]]
        
        total_bid = sum(bid_volumes)
        total_ask = sum(ask_volumes)
        
        # 计算失衡度
        imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)
        
        # 计算价差
        best_bid = bid_prices[0] if bid_prices else 0
        best_ask = ask_prices[0] if ask_prices else 0
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_percent = spread / (mid_price + 1e-8) * 100
        
        # 检测墙和虚假订单
        walls = self._detect_walls(bids, asks, bid_volumes, ask_volumes)
        fake_orders = self._detect_fake_orders(bids, asks, bid_volumes, ask_volumes)
        
        # 支撑阻力位
        support_levels = self._find_support_levels(bids, bid_volumes)
        resistance_levels = self._find_resistance_levels(asks, ask_volumes)
        
        # 信号质量评分
        quality_score = self._calculate_quality_score(
            imbalance, spread_percent, walls, fake_orders
        )
        
        return OrderbookAnalysis(
            bid_volume=total_bid,
            ask_volume=total_ask,
            imbalance=imbalance,
            spread=spread,
            spread_percent=spread_percent,
            mid_price=mid_price,
            detected_walls=walls,
            fake_orders=fake_orders,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            quality_score=quality_score
        )
    
    def _detect_walls(self, bids, asks, bid_volumes, ask_volumes) -> List[Dict]:
        """检测大额订单墙"""
        walls = []
        avg_bid = np.mean(bid_volumes) if bid_volumes else 1
        avg_ask = np.mean(ask_volumes) if ask_volumes else 1
        
        # 检测买盘墙
        for i, (bid, vol) in enumerate(zip(bids, bid_volumes)):
            if vol > avg_bid * self.wall_threshold:
                walls.append({
                    'type': OrderType.SUPPORT.value,
                    'price': float(bid[0]),
                    'volume': vol,
                    'strength': vol / avg_bid
                })
        
        # 检测卖盘墙
        for i, (ask, vol) in enumerate(zip(asks, ask_volumes)):
            if vol > avg_ask * self.wall_threshold:
                walls.append({
                    'type': OrderType.RESISTANCE.value,
                    'price': float(ask[0]),
                    'volume': vol,
                    'strength': vol / avg_ask
                })
        
        return walls
    
    def _detect_fake_orders(self, bids, asks, bid_volumes, ask_volumes) -> List[Dict]:
        """检测虚假订单"""
        fake_orders = []
        
        # 检测突然消失的订单模式 (需要历史数据，这里简化)
        # 简化：检测异常薄的订单层级
        
        for i, (bid, vol) in enumerate(zip(bids, bid_volumes)):
            if i > 0 and vol < bid_volumes[i-1] * self.fake_ratio:
                # 突然减少可能是假托单被撤销
                fake_orders.append({
                    'type': OrderType.FAKE_BID.value,
                    'price': float(bid[0]),
                    'hint': '订单深度异常减少'
                })
        
        for i, (ask, vol) in enumerate(zip(asks, ask_volumes)):
            if i > 0 and vol < ask_volumes[i-1] * self.fake_ratio:
                fake_orders.append({
                    'type': OrderType.FAKE_ASK.value,
                    'price': float(ask[0]),
                    'hint': '订单深度异常减少'
                })
        
        return fake_orders
    
    def _find_support_levels(self, bids, volumes) -> List[float]:
        """找支撑位"""
        levels = []
        avg_vol = np.mean(volumes) if volumes else 0
        
        for bid, vol in zip(bids[:10], volumes[:10]):
            if vol > avg_vol * 1.5:
                levels.append(float(bid[0]))
        
        return levels[:3]
    
    def _find_resistance_levels(self, asks, volumes) -> List[float]:
        """找阻力位"""
        levels = []
        avg_vol = np.mean(volumes) if volumes else 0
        
        for ask, vol in zip(asks[:10], volumes[:10]):
            if vol > avg_vol * 1.5:
                levels.append(float(ask[0]))
        
        return levels[:3]
    
    def _calculate_quality_score(self, imbalance, spread_pct, walls, fake_orders) -> float:
        """计算信号质量评分"""
        score = 50.0
        
        # 失衡度贡献
        score += abs(imbalance) * 30
        
        # 价差惩罚
        if spread_pct > 0.1:
            score -= spread_pct * 10
        
        # 墙的加分
        if walls:
            score += min(len(walls) * 5, 15)
        
        # 虚假订单惩罚
        if fake_orders:
            score -= min(len(fake_orders) * 5, 20)
        
        return max(0, min(100, score))
    
    def _empty_analysis(self) -> OrderbookAnalysis:
        """返回空分析结果"""
        return OrderbookAnalysis(
            bid_volume=0, ask_volume=0, imbalance=0,
            spread=0, spread_percent=0, mid_price=0,
            detected_walls=[], fake_orders=[],
            support_levels=[], resistance_levels=[],
            quality_score=0
        )


def analyze_orderbook_imbalance(orderbook: Dict) -> Dict[str, Any]:
    """
    快速计算订单簿失衡度
    
    Args:
        orderbook: 订单簿数据
        
    Returns:
        分析结果字典
    """
    analyzer = OrderbookAnalyzer()
    result = analyzer.analyze(orderbook)
    
    return {
        'imbalance': result.imbalance,
        'bid_volume': result.bid_volume,
        'ask_volume': result.ask_volume,
        'spread': result.spread,
        'quality_score': result.quality_score,
        'walls': result.detected_walls,
        'fake_orders': result.fake_orders,
    }
