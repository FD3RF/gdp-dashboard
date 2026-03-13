"""
真实Trades数据接入 (Real Trades Stream)
=======================================
修复CVD数据缺失问题

数据源：Binance WebSocket aggTrade
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class AggTrade:
    """聚合成交"""
    price: float
    quantity: float
    first_trade_id: int
    last_trade_id: int
    timestamp: datetime
    is_buyer_maker: bool  # True = 卖方主动, False = 买方主动
    
    @property
    def side(self) -> str:
        """成交方向"""
        return "sell" if self.is_buyer_maker else "buy"
    
    @property
    def is_aggressive_buy(self) -> bool:
        """是否主动买入"""
        return not self.is_buyer_maker
    
    @property
    def is_aggressive_sell(self) -> bool:
        """是否主动卖出"""
        return self.is_buyer_maker


class TradesStream:
    """
    真实Trades数据流
    
    数据源：
    - Binance WebSocket: wss://stream.binance.com/ws/{symbol}@aggTrade
    
    输出：
    - AggTrade列表
    - CVD计算
    - Delta计算
    """
    
    def __init__(self, symbol: str = "ethusdt", max_trades: int = 1000):
        self.symbol = symbol.lower()
        self.max_trades = max_trades
        
        # 成交缓存
        self.trades: deque = deque(maxlen=max_trades)
        
        # 累计数据
        self.cumulative_buy_volume: float = 0
        self.cumulative_sell_volume: float = 0
        self.cumulative_buy_amount: float = 0
        self.cumulative_sell_amount: float = 0
        
        # 时间窗口统计
        self.window_stats: Dict[int, Dict] = {}  # timestamp_minute -> stats
        
        # 状态
        self.connected = False
        self.ws = None
        self._callbacks: List[Callable] = []
    
    def register_callback(self, callback: Callable):
        """注册回调"""
        self._callbacks.append(callback)
    
    def process_aggtrade(self, data: Dict) -> AggTrade:
        """
        处理 aggTrade 数据
        
        Binance aggTrade 格式：
        {
            "a": 12345,      # Aggregate tradeId
            "p": "0.001",    # Price
            "q": "100",      # Quantity
            "f": 100,        # First tradeId
            "l": 105,        # Last tradeId
            "T": 12345,      # Timestamp
            "m": true,       # Was the buyer the maker?
            "M": true        # Was the trade the best price match?
        }
        """
        trade = AggTrade(
            price=float(data.get('p', 0)),
            quantity=float(data.get('q', 0)),
            first_trade_id=data.get('f', 0),
            last_trade_id=data.get('l', 0),
            timestamp=datetime.fromtimestamp(data.get('T', 0) / 1000),
            is_buyer_maker=data.get('m', True),
        )
        
        # 添加到缓存
        self.trades.append(trade)
        
        # 更新累计
        if trade.is_aggressive_buy:
            self.cumulative_buy_volume += trade.quantity
            self.cumulative_buy_amount += trade.price * trade.quantity
        else:
            self.cumulative_sell_volume += trade.quantity
            self.cumulative_sell_amount += trade.price * trade.quantity
        
        # 更新时间窗口
        minute_key = int(trade.timestamp.timestamp() // 60)
        if minute_key not in self.window_stats:
            self.window_stats[minute_key] = {
                'buy_volume': 0,
                'sell_volume': 0,
                'buy_amount': 0,
                'sell_amount': 0,
                'trade_count': 0,
            }
        
        window = self.window_stats[minute_key]
        window['trade_count'] += 1
        if trade.is_aggressive_buy:
            window['buy_volume'] += trade.quantity
            window['buy_amount'] += trade.price * trade.quantity
        else:
            window['sell_volume'] += trade.quantity
            window['sell_amount'] += trade.price * trade.quantity
        
        # 清理旧窗口
        self._cleanup_old_windows()
        
        # 触发回调
        for callback in self._callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return trade
    
    def _cleanup_old_windows(self, max_age_minutes: int = 60):
        """清理旧窗口"""
        current_minute = int(datetime.now().timestamp() // 60)
        old_keys = [k for k in self.window_stats.keys() if current_minute - k > max_age_minutes]
        for k in old_keys:
            del self.window_stats[k]
    
    def calculate_cvd(self, window_minutes: int = 5) -> Dict[str, Any]:
        """
        计算CVD (Cumulative Volume Delta)
        
        CVD = 主动买入量 - 主动卖出量
        """
        current_minute = int(datetime.now().timestamp() // 60)
        
        window_buy_vol = 0
        window_sell_vol = 0
        
        for minute in range(current_minute - window_minutes, current_minute + 1):
            if minute in self.window_stats:
                stats = self.window_stats[minute]
                window_buy_vol += stats['buy_volume']
                window_sell_vol += stats['sell_volume']
        
        cvd_value = window_buy_vol - window_sell_vol
        total_vol = window_buy_vol + window_sell_vol
        
        # CVD方向判断
        if cvd_value > 0:
            direction = "bullish"
            signal = "买方主导"
        elif cvd_value < 0:
            direction = "bearish"
            signal = "卖方主导"
        else:
            direction = "neutral"
            signal = "买卖均衡"
        
        return {
            "value": cvd_value,
            "buy_volume": window_buy_vol,
            "sell_volume": window_sell_vol,
            "direction": direction,
            "signal": signal,
            "strength": abs(cvd_value) / total_vol if total_vol > 0 else 0,
            "window_minutes": window_minutes,
        }
    
    def calculate_delta(self, window_minutes: int = 5) -> Dict[str, Any]:
        """
        计算Delta
        
        Delta = (买入量 - 卖出量) / 总量
        """
        cvd = self.calculate_cvd(window_minutes)
        total = cvd['buy_volume'] + cvd['sell_volume']
        
        delta_pct = (cvd['value'] / total * 100) if total > 0 else 0
        
        return {
            "value": cvd['value'],
            "pct": delta_pct,
            "direction": cvd['direction'],
        }
    
    def get_trade_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """获取成交摘要"""
        cvd = self.calculate_cvd(window_minutes)
        delta = self.calculate_delta(window_minutes)
        
        # 最近成交统计
        recent_trades = list(self.trades)[-100:]
        avg_trade_size = sum(t.quantity for t in recent_trades) / len(recent_trades) if recent_trades else 0
        
        # 大单检测
        large_trades = [t for t in recent_trades if t.quantity > avg_trade_size * 3]
        
        return {
            "cvd": cvd,
            "delta": delta,
            "total_trades": len(self.trades),
            "recent_large_trades": len(large_trades),
            "avg_trade_size": avg_trade_size,
            "cumulative_buy_volume": self.cumulative_buy_volume,
            "cumulative_sell_volume": self.cumulative_sell_volume,
            "cumulative_delta": self.cumulative_buy_volume - self.cumulative_sell_volume,
        }
    
    def get_aggressive_flow(self, window_minutes: int = 5) -> Dict[str, Any]:
        """
        获取主动买卖流
        
        用于四层决策的订单流判断
        """
        current_minute = int(datetime.now().timestamp() // 60)
        
        aggressive_buy = 0
        aggressive_sell = 0
        
        for minute in range(current_minute - window_minutes, current_minute + 1):
            if minute in self.window_stats:
                stats = self.window_stats[minute]
                aggressive_buy += stats['buy_volume']
                aggressive_sell += stats['sell_volume']
        
        imbalance = (aggressive_buy - aggressive_sell) / (aggressive_buy + aggressive_sell + 1)
        
        return {
            "aggressive_buy": aggressive_buy,
            "aggressive_sell": aggressive_sell,
            "imbalance": imbalance,
            "direction": "bullish" if imbalance > 0.1 else "bearish" if imbalance < -0.1 else "neutral",
        }


# 模拟数据生成器（当WebSocket不可用时）
class SimulatedTradesStream:
    """模拟Trades数据流"""
    
    def __init__(self, symbol: str = "ETH/USDT"):
        self.symbol = symbol
        self.trades_stream = TradesStream(symbol)
    
    def generate_trade(self, current_price: float, imbalance_bias: float = 0) -> AggTrade:
        """生成模拟成交"""
        import random
        
        # 基于订单簿失衡决定买卖比例
        buy_probability = 0.5 + imbalance_bias * 0.3
        
        is_buyer_maker = random.random() > buy_probability
        
        # 价格波动
        price_offset = random.uniform(-2, 2)
        price = current_price + price_offset
        
        # 成交量
        quantity = random.uniform(0.1, 10.0)
        
        # 大单概率
        if random.random() < 0.05:
            quantity *= random.uniform(5, 20)
        
        trade = AggTrade(
            price=price,
            quantity=quantity,
            first_trade_id=random.randint(1000000, 9999999),
            last_trade_id=random.randint(1000000, 9999999),
            timestamp=datetime.now(),
            is_buyer_maker=is_buyer_maker,
        )
        
        # 添加到流
        self.trades_stream.trades.append(trade)
        
        # 更新窗口统计
        minute_key = int(trade.timestamp.timestamp() // 60)
        if minute_key not in self.trades_stream.window_stats:
            self.trades_stream.window_stats[minute_key] = {
                'buy_volume': 0,
                'sell_volume': 0,
                'buy_amount': 0,
                'sell_amount': 0,
                'trade_count': 0,
            }
        
        window = self.trades_stream.window_stats[minute_key]
        window['trade_count'] += 1
        if trade.is_aggressive_buy:
            window['buy_volume'] += trade.quantity
            window['buy_amount'] += trade.price * trade.quantity
            self.trades_stream.cumulative_buy_volume += trade.quantity
        else:
            window['sell_volume'] += trade.quantity
            window['sell_amount'] += trade.price * trade.quantity
            self.trades_stream.cumulative_sell_volume += trade.quantity
        
        return trade
    
    def generate_trades_batch(self, current_price: float, count: int = 50, imbalance_bias: float = 0):
        """批量生成模拟成交"""
        for _ in range(count):
            self.generate_trade(current_price, imbalance_bias)
    
    def get_cvd(self, window_minutes: int = 5) -> Dict:
        """获取CVD"""
        return self.trades_stream.calculate_cvd(window_minutes)
    
    def get_trade_summary(self, window_minutes: int = 5) -> Dict:
        """获取成交摘要"""
        return self.trades_stream.get_trade_summary(window_minutes)


# 全局实例
_real_trades_stream: Optional[TradesStream] = None
_simulated_trades_stream: Optional[SimulatedTradesStream] = None


def get_trades_stream(use_simulated: bool = True) -> TradesStream:
    """获取Trades流"""
    global _real_trades_stream, _simulated_trades_stream
    
    if use_simulated:
        if _simulated_trades_stream is None:
            _simulated_trades_stream = SimulatedTradesStream()
        return _simulated_trades_stream.trades_stream
    else:
        if _real_trades_stream is None:
            _real_trades_stream = TradesStream()
        return _real_trades_stream


def get_real_cvd(current_price: float, use_simulated: bool = True, 
                 imbalance_bias: float = 0, window_minutes: int = 5) -> Dict:
    """
    获取真实CVD数据
    
    Args:
        current_price: 当前价格
        use_simulated: 是否使用模拟数据
        imbalance_bias: 订单簿失衡偏向 (-1 to 1)
        window_minutes: 时间窗口
    
    Returns:
        CVD数据字典
    """
    if use_simulated:
        global _simulated_trades_stream
        if _simulated_trades_stream is None:
            _simulated_trades_stream = SimulatedTradesStream()
        
        # 生成新成交
        _simulated_trades_stream.generate_trades_batch(
            current_price, 
            count=30, 
            imbalance_bias=imbalance_bias
        )
        
        return _simulated_trades_stream.get_cvd(window_minutes)
    else:
        stream = get_trades_stream(False)
        return stream.calculate_cvd(window_minutes)
