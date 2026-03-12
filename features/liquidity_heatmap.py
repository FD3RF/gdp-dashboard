# features/liquidity_heatmap.py
"""
模块 41: 流动性热图分析
=======================
可视化订单簿"冰山单"，识别真实支撑与诱多/诱空墙
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class WallType(Enum):
    """墙类型"""
    STRONG_SUPPORT = "强支撑"
    WEAK_SUPPORT = "弱支撑"
    STRONG_RESISTANCE = "强阻力"
    WEAK_RESISTANCE = "弱阻力"
    FAKE_BID = "诱多陷阱"
    FAKE_ASK = "诱空陷阱"
    ICEBERG = "冰山单"


@dataclass
class LiquidityLevel:
    """流动性层级"""
    price: float
    volume: float
    type: WallType
    confidence: float  # 置信度 0-1
    description: str


@dataclass
class LiquidityHeatmap:
    """流动性热图结果"""
    levels: List[LiquidityLevel]
    support_zones: List[Tuple[float, float]]  # (价格, 强度)
    resistance_zones: List[Tuple[float, float]]
    trap_warnings: List[Dict[str, Any]]
    overall_imbalance: float
    liquidity_score: float  # 流动性健康度


class LiquidityHeatmapAnalyzer:
    """
    流动性热图分析器
    
    功能：
    - 识别真实支撑/阻力
    - 检测诱多/诱空陷阱
    - 发现冰山单
    - 生成流动性热图
    """
    
    def __init__(self, 
                 strong_wall_ratio: float = 5.0,
                 weak_wall_ratio: float = 2.0,
                 fake_detection_threshold: float = 0.3):
        """
        Args:
            strong_wall_ratio: 强墙判定比例
            weak_wall_ratio: 弱墙判定比例
            fake_detection_threshold: 假订单检测阈值
        """
        self.strong_wall_ratio = strong_wall_ratio
        self.weak_wall_ratio = weak_wall_ratio
        self.fake_detection_threshold = fake_detection_threshold
    
    def analyze(self, orderbook: Dict[str, List], price: float) -> LiquidityHeatmap:
        """
        分析订单簿流动性
        
        Args:
            orderbook: {'bids': [[price, volume]], 'asks': [[price, volume]]}
            price: 当前价格
            
        Returns:
            LiquidityHeatmap
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return self._empty_heatmap()
        
        # 提取数据
        bid_prices = np.array([float(b[0]) for b in bids[:50]])
        bid_volumes = np.array([float(b[1]) for b in bids[:50]])
        ask_prices = np.array([float(a[0]) for a in asks[:50]])
        ask_volumes = np.array([float(a[1]) for a in asks[:50]])
        
        # 计算统计量
        avg_bid_vol = np.mean(bid_volumes)
        avg_ask_vol = np.mean(ask_volumes)
        std_bid_vol = np.std(bid_volumes) + 1e-8
        std_ask_vol = np.std(ask_volumes) + 1e-8
        
        # 识别流动性层级
        levels = []
        support_zones = []
        resistance_zones = []
        trap_warnings = []
        
        # 分析买盘
        for i, (p, v) in enumerate(zip(bid_prices, bid_volumes)):
            level = self._analyze_level(p, v, avg_bid_vol, std_bid_vol, price, True, i)
            if level:
                levels.append(level)
                if level.type in [WallType.STRONG_SUPPORT, WallType.WEAK_SUPPORT]:
                    support_zones.append((p, v / avg_bid_vol))
                elif level.type == WallType.FAKE_BID:
                    trap_warnings.append({
                        'type': '诱多陷阱',
                        'price': p,
                        'description': f'${p:.2f} 检测到虚假托单'
                    })
        
        # 分析卖盘
        for i, (p, v) in enumerate(zip(ask_prices, ask_volumes)):
            level = self._analyze_level(p, v, avg_ask_vol, std_ask_vol, price, False, i)
            if level:
                levels.append(level)
                if level.type in [WallType.STRONG_RESISTANCE, WallType.WEAK_RESISTANCE]:
                    resistance_zones.append((p, v / avg_ask_vol))
                elif level.type == WallType.FAKE_ASK:
                    trap_warnings.append({
                        'type': '诱空陷阱',
                        'price': p,
                        'description': f'${p:.2f} 检测到虚假压单'
                    })
        
        # 计算整体失衡度
        total_bid = np.sum(bid_volumes)
        total_ask = np.sum(ask_volumes)
        overall_imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)
        
        # 流动性健康度
        liquidity_score = self._calculate_liquidity_score(
            len(support_zones), len(resistance_zones), 
            len(trap_warnings), overall_imbalance
        )
        
        return LiquidityHeatmap(
            levels=levels,
            support_zones=support_zones[:5],
            resistance_zones=resistance_zones[:5],
            trap_warnings=trap_warnings,
            overall_imbalance=overall_imbalance,
            liquidity_score=liquidity_score
        )
    
    def _analyze_level(self, price: float, volume: float, 
                       avg_vol: float, std_vol: float,
                       current_price: float, is_bid: bool,
                       depth_index: int) -> Optional[LiquidityLevel]:
        """分析单个层级"""
        ratio = volume / (avg_vol + 1e-8)
        
        # 判断墙类型
        wall_type = None
        confidence = 0.5
        description = ""
        
        if ratio > self.strong_wall_ratio:
            if is_bid:
                # 检查是否可能是陷阱
                if depth_index < 3 and ratio > self.strong_wall_ratio * 2:
                    wall_type = WallType.FAKE_BID
                    confidence = 0.7
                    description = f"可疑大额托单，可能是诱多陷阱"
                else:
                    wall_type = WallType.STRONG_SUPPORT
                    confidence = 0.9
                    description = f"强支撑墙 {ratio:.1f}倍平均量"
            else:
                if depth_index < 3 and ratio > self.strong_wall_ratio * 2:
                    wall_type = WallType.FAKE_ASK
                    confidence = 0.7
                    description = f"可疑大额压单，可能是诱空陷阱"
                else:
                    wall_type = WallType.STRONG_RESISTANCE
                    confidence = 0.9
                    description = f"强阻力墙 {ratio:.1f}倍平均量"
        
        elif ratio > self.weak_wall_ratio:
            if is_bid:
                wall_type = WallType.WEAK_SUPPORT
                confidence = 0.7
                description = f"弱支撑 {ratio:.1f}倍平均量"
            else:
                wall_type = WallType.WEAK_RESISTANCE
                confidence = 0.7
                description = f"弱阻力 {ratio:.1f}倍平均量"
        
        # 检测冰山单模式 (连续小量 + 偶尔大量)
        if wall_type is None and volume < avg_vol * 0.5:
            # 可能是冰山单的一部分
            pass
        
        if wall_type is None:
            return None
        
        return LiquidityLevel(
            price=price,
            volume=volume,
            type=wall_type,
            confidence=confidence,
            description=description
        )
    
    def _calculate_liquidity_score(self, supports: int, resistances: int,
                                    traps: int, imbalance: float) -> float:
        """计算流动性健康度"""
        score = 50.0
        
        # 有支撑阻力加分
        score += min(supports + resistances, 5) * 5
        
        # 陷阱惩罚
        score -= traps * 10
        
        # 过度失衡惩罚
        if abs(imbalance) > 0.5:
            score -= abs(imbalance) * 20
        
        return max(0, min(100, score))
    
    def _empty_heatmap(self) -> LiquidityHeatmap:
        """返回空热图"""
        return LiquidityHeatmap(
            levels=[],
            support_zones=[],
            resistance_zones=[],
            trap_warnings=[],
            overall_imbalance=0,
            liquidity_score=50
        )


def generate_liquidity_heatmap(orderbook: Dict, price: float) -> Dict[str, Any]:
    """
    生成流动性热图
    
    Args:
        orderbook: 订单簿数据
        price: 当前价格
        
    Returns:
        热图数据字典
    """
    analyzer = LiquidityHeatmapAnalyzer()
    result = analyzer.analyze(orderbook, price)
    
    return {
        'support_zones': result.support_zones,
        'resistance_zones': result.resistance_zones,
        'trap_warnings': result.trap_warnings,
        'imbalance': result.overall_imbalance,
        'liquidity_score': result.liquidity_score,
        'levels': [
            {
                'price': l.price,
                'volume': l.volume,
                'type': l.type.value,
                'confidence': l.confidence,
                'description': l.description
            }
            for l in result.levels[:10]
        ]
    }
