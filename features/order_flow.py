"""
订单流分析模块 (Order Flow Analysis)
===================================
核心功能：分析真实成交流，识别机构行为
- CVD (Cumulative Volume Delta)
- 主动买卖量分析
- 订单流失衡检测
- 大单追踪
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """单笔成交"""
    timestamp: datetime
    price: float
    volume: float
    side: str  # "buy" or "sell"
    is_aggressive: bool = False  # 是否主动成交
    
    # 计算字段
    notional: float = 0.0  # 成交金额
    price_impact: float = 0.0  # 价格冲击


@dataclass
class OrderFlowSnapshot:
    """订单流快照"""
    timestamp: datetime
    
    # 成交量统计
    buy_volume: float
    sell_volume: float
    total_volume: float
    
    # 主动成交
    aggressive_buy: float
    aggressive_sell: float
    
    # Delta
    delta: float  # buy - sell
    delta_pct: float  # delta / total
    
    # CVD
    cvd: float  # 累计Delta
    
    # 大单
    large_buy_count: int
    large_sell_count: int
    large_buy_volume: float
    large_sell_volume: float
    
    # 订单流失衡
    imbalance_score: float  # -1 到 1
    imbalance_direction: str  # "bullish" / "bearish" / "neutral"
    
    # 价格影响
    vwap_buy: float
    vwap_sell: float
    price_change: float


@dataclass
class CVDAnalysis:
    """CVD分析结果"""
    cvd_value: float
    cvd_trend: str  # "rising" / "falling" / "flat"
    cvd_divergence: bool  # 与价格背离
    cvd_signal: str  # "bullish" / "bearish" / "neutral"
    
    # 区间统计
    cvd_high: float
    cvd_low: float
    cvd_range: float
    
    # 背离详情
    divergence_type: Optional[str] = None  # "bullish_div" / "bearish_div"
    divergence_strength: float = 0.0


class OrderFlowAnalyzer:
    """
    订单流分析器
    
    核心功能：
    1. 实时追踪买卖成交
    2. 计算CVD (Cumulative Volume Delta)
    3. 识别主动买卖行为
    4. 检测大单和机构行为
    5. 分析订单流失衡
    """
    
    # 大单阈值 (ETH)
    LARGE_ORDER_THRESHOLD = 10.0  # 10 ETH
    
    # 巨型单阈值
    WHALE_ORDER_THRESHOLD = 50.0  # 50 ETH
    
    def __init__(
        self,
        large_order_threshold: float = None,
        history_length: int = 1000
    ):
        self.large_order_threshold = large_order_threshold or self.LARGE_ORDER_THRESHOLD
        self.history_length = history_length
        
        # 成交历史
        self.trades: deque = deque(maxlen=history_length)
        
        # CVD历史
        self.cvd_history: deque = deque(maxlen=history_length)
        self.cvd = 0.0
        
        # 区间统计
        self.interval_trades: List[Trade] = []
        self.interval_start: Optional[datetime] = None
        
        # 统计
        self.stats = {
            "total_trades": 0,
            "total_buy_volume": 0,
            "total_sell_volume": 0,
            "large_orders": 0,
            "whale_orders": 0,
        }
    
    def add_trade(
        self,
        price: float,
        volume: float,
        side: str,
        timestamp: datetime = None,
        is_aggressive: bool = None,
        prev_price: float = None
    ) -> Trade:
        """
        添加成交记录
        
        Args:
            price: 成交价格
            volume: 成交量
            side: 买卖方向 ("buy" / "sell")
            timestamp: 时间戳
            is_aggressive: 是否主动成交
            prev_price: 前一价格（用于判断主动成交）
        
        Returns:
            Trade对象
        """
        timestamp = timestamp or datetime.now()
        
        # 判断是否主动成交
        if is_aggressive is None and prev_price is not None:
            # 主动买入：成交价 >= 卖一价
            # 主动卖出：成交价 <= 买一价
            if side == "buy" and price >= prev_price:
                is_aggressive = True
            elif side == "sell" and price <= prev_price:
                is_aggressive = True
            else:
                is_aggressive = False
        else:
            is_aggressive = is_aggressive or False
        
        trade = Trade(
            timestamp=timestamp,
            price=price,
            volume=volume,
            side=side,
            is_aggressive=is_aggressive,
            notional=price * volume,
        )
        
        self.trades.append(trade)
        self.interval_trades.append(trade)
        
        # 更新CVD
        if side == "buy":
            self.cvd += volume
        else:
            self.cvd -= volume
        
        self.cvd_history.append({
            "timestamp": timestamp,
            "cvd": self.cvd,
            "price": price,
        })
        
        # 更新统计
        self.stats["total_trades"] += 1
        if side == "buy":
            self.stats["total_buy_volume"] += volume
        else:
            self.stats["total_sell_volume"] += volume
        
        if volume >= self.large_order_threshold:
            self.stats["large_orders"] += 1
        if volume >= self.WHALE_ORDER_THRESHOLD:
            self.stats["whale_orders"] += 1
        
        return trade
    
    def process_trades_batch(
        self,
        trades_data: List[Dict],
        prev_price: float = None
    ) -> List[Trade]:
        """
        批量处理成交数据
        
        Args:
            trades_data: 成交数据列表 [{"price", "volume", "side", "timestamp"}, ...]
            prev_price: 前一价格
        
        Returns:
            Trade对象列表
        """
        trades = []
        last_price = prev_price
        
        for data in trades_data:
            trade = self.add_trade(
                price=data.get("price", 0),
                volume=data.get("volume", 0),
                side=data.get("side", "buy"),
                timestamp=data.get("timestamp"),
                prev_price=last_price,
            )
            trades.append(trade)
            last_price = trade.price
        
        return trades
    
    def get_snapshot(self, period_seconds: int = 300) -> OrderFlowSnapshot:
        """
        获取订单流快照
        
        Args:
            period_seconds: 统计周期（秒）
        
        Returns:
            OrderFlowSnapshot
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=period_seconds)
        
        # 过滤周期内成交
        period_trades = [t for t in self.trades if t.timestamp >= cutoff]
        
        if not period_trades:
            return self._empty_snapshot(now)
        
        # 基础统计
        buy_trades = [t for t in period_trades if t.side == "buy"]
        sell_trades = [t for t in period_trades if t.side == "sell"]
        
        buy_volume = sum(t.volume for t in buy_trades)
        sell_volume = sum(t.volume for t in sell_trades)
        total_volume = buy_volume + sell_volume
        
        # 主动成交
        aggressive_buy = sum(t.volume for t in buy_trades if t.is_aggressive)
        aggressive_sell = sum(t.volume for t in sell_trades if t.is_aggressive)
        
        # Delta
        delta = buy_volume - sell_volume
        delta_pct = delta / total_volume if total_volume > 0 else 0
        
        # 大单统计
        large_buy = [t for t in buy_trades if t.volume >= self.large_order_threshold]
        large_sell = [t for t in sell_trades if t.volume >= self.large_order_threshold]
        
        large_buy_volume = sum(t.volume for t in large_buy)
        large_sell_volume = sum(t.volume for t in large_sell)
        
        # 订单流失衡
        imbalance = self._calculate_imbalance(buy_volume, sell_volume, aggressive_buy, aggressive_sell)
        
        # VWAP
        vwap_buy = sum(t.notional for t in buy_trades) / buy_volume if buy_volume > 0 else 0
        vwap_sell = sum(t.notional for t in sell_trades) / sell_volume if sell_volume > 0 else 0
        
        # 价格变化
        if len(period_trades) >= 2:
            price_change = period_trades[-1].price - period_trades[0].price
        else:
            price_change = 0
        
        return OrderFlowSnapshot(
            timestamp=now,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            total_volume=total_volume,
            aggressive_buy=aggressive_buy,
            aggressive_sell=aggressive_sell,
            delta=delta,
            delta_pct=delta_pct,
            cvd=self.cvd,
            large_buy_count=len(large_buy),
            large_sell_count=len(large_sell),
            large_buy_volume=large_buy_volume,
            large_sell_volume=large_sell_volume,
            imbalance_score=imbalance["score"],
            imbalance_direction=imbalance["direction"],
            vwap_buy=vwap_buy,
            vwap_sell=vwap_sell,
            price_change=price_change,
        )
    
    def _calculate_imbalance(
        self,
        buy_vol: float,
        sell_vol: float,
        aggressive_buy: float,
        aggressive_sell: float
    ) -> Dict[str, Any]:
        """计算订单流失衡"""
        total = buy_vol + sell_vol
        if total == 0:
            return {"score": 0, "direction": "neutral"}
        
        # 基础失衡
        basic_imb = (buy_vol - sell_vol) / total
        
        # 主动成交失衡（权重更高）
        aggressive_total = aggressive_buy + aggressive_sell
        if aggressive_total > 0:
            aggressive_imb = (aggressive_buy - aggressive_sell) / aggressive_total
        else:
            aggressive_imb = 0
        
        # 综合评分
        score = basic_imb * 0.4 + aggressive_imb * 0.6
        
        if score > 0.2:
            direction = "bullish"
        elif score < -0.2:
            direction = "bearish"
        else:
            direction = "neutral"
        
        return {"score": score, "direction": direction}
    
    def _empty_snapshot(self, timestamp: datetime) -> OrderFlowSnapshot:
        """空快照"""
        return OrderFlowSnapshot(
            timestamp=timestamp,
            buy_volume=0, sell_volume=0, total_volume=0,
            aggressive_buy=0, aggressive_sell=0,
            delta=0, delta_pct=0, cvd=self.cvd,
            large_buy_count=0, large_sell_count=0,
            large_buy_volume=0, large_sell_volume=0,
            imbalance_score=0, imbalance_direction="neutral",
            vwap_buy=0, vwap_sell=0, price_change=0,
        )
    
    def analyze_cvd(self, lookback: int = 100) -> CVDAnalysis:
        """
        分析CVD
        
        Args:
            lookback: 回看条数
        
        Returns:
            CVDAnalysis
        """
        if len(self.cvd_history) < 10:
            return CVDAnalysis(
                cvd_value=self.cvd,
                cvd_trend="flat",
                cvd_divergence=False,
                cvd_signal="neutral",
                cvd_high=self.cvd, cvd_low=self.cvd, cvd_range=0,
            )
        
        history = list(self.cvd_history)[-lookback:]
        
        # CVD趋势
        recent = [h["cvd"] for h in history[-20:]]
        older = [h["cvd"] for h in history[-40:-20]] if len(history) >= 40 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.02:
            trend = "rising"
        elif recent_avg < older_avg * 0.98:
            trend = "falling"
        else:
            trend = "flat"
        
        # 区间统计
        cvd_values = [h["cvd"] for h in history]
        cvd_high = max(cvd_values)
        cvd_low = min(cvd_values)
        cvd_range = cvd_high - cvd_low
        
        # 背离检测
        prices = [h["price"] for h in history]
        price_trend = prices[-1] - prices[0]
        cvd_trend_val = cvd_values[-1] - cvd_values[0]
        
        divergence = False
        divergence_type = None
        divergence_strength = 0
        
        # 看跌背离：价格上涨但CVD下降
        if price_trend > 0 and cvd_trend_val < 0:
            divergence = True
            divergence_type = "bearish_div"
            divergence_strength = abs(cvd_trend_val / (cvd_range + 1))
        
        # 看涨背离：价格下跌但CVD上升
        elif price_trend < 0 and cvd_trend_val > 0:
            divergence = True
            divergence_type = "bullish_div"
            divergence_strength = abs(cvd_trend_val / (cvd_range + 1))
        
        # 信号
        if divergence:
            if divergence_type == "bullish_div":
                signal = "bullish"
            else:
                signal = "bearish"
        elif trend == "rising" and self.cvd > 0:
            signal = "bullish"
        elif trend == "falling" and self.cvd < 0:
            signal = "bearish"
        else:
            signal = "neutral"
        
        return CVDAnalysis(
            cvd_value=self.cvd,
            cvd_trend=trend,
            cvd_divergence=divergence,
            cvd_signal=signal,
            cvd_high=cvd_high,
            cvd_low=cvd_low,
            cvd_range=cvd_range,
            divergence_type=divergence_type,
            divergence_strength=divergence_strength,
        )
    
    def detect_absorption(self, window: int = 50) -> Dict[str, Any]:
        """
        检测吸收行为
        
        吸收 = 大量成交但价格不移动，说明有机构在吸收
        """
        if len(self.trades) < window:
            return {"detected": False}
        
        recent_trades = list(self.trades)[-window:]
        
        # 计算价格变化
        price_start = recent_trades[0].price
        price_end = recent_trades[-1].price
        price_change_pct = abs(price_end - price_start) / price_start * 100
        
        # 计算成交量
        total_volume = sum(t.volume for t in recent_trades)
        avg_volume = total_volume / window
        
        # 吸收条件：高成交量 + 低价格变化
        is_absorption = (
            price_change_pct < 0.1 and  # 价格变化小于0.1%
            avg_volume > self.large_order_threshold * 0.5  # 成交量较高
        )
        
        # 判断方向
        buy_vol = sum(t.volume for t in recent_trades if t.side == "buy")
        sell_vol = sum(t.volume for t in recent_trades if t.side == "sell")
        
        if buy_vol > sell_vol * 1.5:
            direction = "buy_absorption"  # 买入吸收，潜在上涨
        elif sell_vol > buy_vol * 1.5:
            direction = "sell_absorption"  # 卖出吸收，潜在下跌
        else:
            direction = "neutral"
        
        return {
            "detected": is_absorption,
            "direction": direction,
            "volume": total_volume,
            "price_change_pct": price_change_pct,
            "strength": avg_volume / self.large_order_threshold if is_absorption else 0,
        }
    
    def get_large_orders(self, min_volume: float = None) -> List[Trade]:
        """获取大单列表"""
        threshold = min_volume or self.large_order_threshold
        return [t for t in self.trades if t.volume >= threshold]
    
    def get_order_flow_summary(self) -> Dict[str, Any]:
        """获取订单流摘要"""
        snapshot = self.get_snapshot(300)  # 5分钟
        cvd_analysis = self.analyze_cvd()
        absorption = self.detect_absorption()
        
        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "cvd": {
                "value": round(snapshot.cvd, 4),
                "trend": cvd_analysis.cvd_trend,
                "signal": cvd_analysis.cvd_signal,
                "divergence": cvd_analysis.cvd_divergence,
                "divergence_type": cvd_analysis.divergence_type,
            },
            "delta": {
                "value": round(snapshot.delta, 4),
                "pct": round(snapshot.delta_pct * 100, 2),
            },
            "volume": {
                "buy": round(snapshot.buy_volume, 4),
                "sell": round(snapshot.sell_volume, 4),
                "total": round(snapshot.total_volume, 4),
            },
            "aggressive": {
                "buy": round(snapshot.aggressive_buy, 4),
                "sell": round(snapshot.aggressive_sell, 4),
            },
            "large_orders": {
                "buy_count": snapshot.large_buy_count,
                "sell_count": snapshot.large_sell_count,
                "buy_volume": round(snapshot.large_buy_volume, 4),
                "sell_volume": round(snapshot.large_sell_volume, 4),
            },
            "imbalance": {
                "score": round(snapshot.imbalance_score, 3),
                "direction": snapshot.imbalance_direction,
            },
            "absorption": absorption,
            "stats": self.stats,
        }


# 全局实例
_order_flow_analyzer: Optional[OrderFlowAnalyzer] = None


def get_order_flow_analyzer() -> OrderFlowAnalyzer:
    """获取全局订单流分析器"""
    global _order_flow_analyzer
    if _order_flow_analyzer is None:
        _order_flow_analyzer = OrderFlowAnalyzer()
    return _order_flow_analyzer


def analyze_order_flow(trades_data: List[Dict] = None) -> Dict[str, Any]:
    """
    分析订单流（便捷函数）
    
    Args:
        trades_data: 成交数据（可选，用于添加新数据）
    
    Returns:
        订单流摘要
    """
    analyzer = get_order_flow_analyzer()
    
    if trades_data:
        analyzer.process_trades_batch(trades_data)
    
    return analyzer.get_order_flow_summary()
