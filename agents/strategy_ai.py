"""
策略 AI 代理
============

智能策略生成模块：
- 技术指标分析
- AI 信号生成
- 多策略组合
- 趋势/反弹/套利策略
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from decimal import Decimal

from core.base import BaseModule
from core.constants import OrderSide
from agents.trade_agent import TradeSignal


class StrategyType(Enum):
    """策略类型"""
    TREND = 'trend'           # 趋势策略
    REVERSAL = 'reversal'     # 反弹策略
    SCALPING = 'scalping'     # 短线套利
    ARBITRAGE = 'arbitrage'   # 套利策略
    MOMENTUM = 'momentum'     # 动量策略
    MEAN_REVERSION = 'mean_reversion'  # 均值回归


class SignalStrength(Enum):
    """信号强度"""
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9


@dataclass
class TechnicalIndicators:
    """技术指标"""
    # 均线
    ma5: float = 0
    ma10: float = 0
    ma20: float = 0
    ma60: float = 0
    
    # MACD
    macd: float = 0
    macd_signal: float = 0
    macd_hist: float = 0
    
    # RSI
    rsi: float = 50
    
    # 布林带
    boll_upper: float = 0
    boll_middle: float = 0
    boll_lower: float = 0
    
    # 其他
    atr: float = 0          # 平均真实波幅
    obv: float = 0          # 能量潮
    volume_ratio: float = 1  # 量比
    
    def to_dict(self) -> Dict:
        return {
            'ma5': self.ma5,
            'ma10': self.ma10,
            'ma20': self.ma20,
            'ma60': self.ma60,
            'macd': self.macd,
            'macd_signal': self.macd_signal,
            'macd_hist': self.macd_hist,
            'rsi': self.rsi,
            'boll_upper': self.boll_upper,
            'boll_middle': self.boll_middle,
            'boll_lower': self.boll_lower,
            'atr': self.atr,
            'obv': self.obv,
            'volume_ratio': self.volume_ratio,
        }


@dataclass
class MarketContext:
    """市场环境"""
    symbol: str
    price: float
    volume_24h: float = 0
    price_change_24h: float = 0
    price_high_24h: float = 0
    price_low_24h: float = 0
    trend: str = 'neutral'  # up, down, neutral
    volatility: str = 'normal'  # low, normal, high
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume_24h': self.volume_24h,
            'price_change_24h': self.price_change_24h,
            'price_high_24h': self.price_high_24h,
            'price_low_24h': self.price_low_24h,
            'trend': self.trend,
            'volatility': self.volatility,
            'timestamp': self.timestamp.isoformat(),
        }


class StrategyAI(BaseModule):
    """
    策略 AI 代理
    
    功能：
    1. 技术指标计算 - MA, RSI, MACD, 布林带等
    2. 信号生成 - 基于 AI 分析生成交易信号
    3. 多策略组合 - 趋势、反弹、套利策略
    4. 风险评估 - 评估交易风险
    5. 仓位建议 - 智能仓位管理
    """
    
    # 策略权重配置
    STRATEGY_WEIGHTS = {
        StrategyType.TREND: 0.35,
        StrategyType.REVERSAL: 0.25,
        StrategyType.MOMENTUM: 0.20,
        StrategyType.MEAN_REVERSION: 0.20,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('strategy_ai', config)
        
        # 活跃策略
        self._active_strategies: List[StrategyType] = [
            StrategyType.TREND,
            StrategyType.REVERSAL,
            StrategyType.MOMENTUM,
        ]
        
        # 历史数据缓存
        self._kline_cache: Dict[str, pd.DataFrame] = {}
        self._indicator_cache: Dict[str, TechnicalIndicators] = {}
        self._context_cache: Dict[str, MarketContext] = {}
        
        # 信号历史
        self._signal_history: List[TradeSignal] = []
        
        # AI 模型（可扩展）
        self._models: Dict[str, Any] = {}
        
        # 统计
        self._stats = {
            'signals_generated': 0,
            'signals_by_type': {},
            'avg_strength': 0,
        }
    
    async def initialize(self) -> bool:
        """初始化"""
        self.logger.info("Initializing Strategy AI...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """启动"""
        self._running = True
        self._start_time = datetime.now()
        self.logger.info("Strategy AI started")
        return True
    
    async def stop(self) -> bool:
        """停止"""
        self._running = False
        self.logger.info("Strategy AI stopped")
        return True
    
    # ==================== 技术指标计算 ====================
    
    def calculate_indicators(self, klines: pd.DataFrame) -> TechnicalIndicators:
        """
        计算技术指标
        
        Args:
            klines: K线数据 DataFrame (open, high, low, close, volume)
            
        Returns:
            技术指标对象
        """
        indicators = TechnicalIndicators()
        
        if klines.empty or len(klines) < 60:
            return indicators
        
        close = klines['close']
        high = klines['high']
        low = klines['low']
        volume = klines['volume']
        
        # 均线
        indicators.ma5 = self._calculate_ma(close, 5)
        indicators.ma10 = self._calculate_ma(close, 10)
        indicators.ma20 = self._calculate_ma(close, 20)
        indicators.ma60 = self._calculate_ma(close, 60)
        
        # MACD
        macd, signal, hist = self._calculate_macd(close)
        indicators.macd = macd
        indicators.macd_signal = signal
        indicators.macd_hist = hist
        
        # RSI
        indicators.rsi = self._calculate_rsi(close, 14)
        
        # 布林带
        upper, middle, lower = self._calculate_bollinger(close, 20)
        indicators.boll_upper = upper
        indicators.boll_middle = middle
        indicators.boll_lower = lower
        
        # ATR
        indicators.atr = self._calculate_atr(high, low, close, 14)
        
        # OBV
        indicators.obv = self._calculate_obv(close, volume)
        
        # 量比
        if len(volume) >= 20:
            avg_volume = volume.iloc[-20:].mean()
            current_volume = volume.iloc[-1]
            indicators.volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        return indicators
    
    def _calculate_ma(self, series: pd.Series, period: int) -> float:
        """计算移动平均"""
        if len(series) < period:
            return series.iloc[-1] if len(series) > 0 else 0
        return float(series.iloc[-period:].mean())
    
    def _calculate_macd(
        self, 
        series: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """计算 MACD"""
        if len(series) < slow + signal:
            return 0, 0, 0
        
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        """计算 RSI"""
        if len(series) < period + 1:
            return 50
        
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    
    def _calculate_bollinger(
        self, 
        series: pd.Series, 
        period: int = 20, 
        std_dev: int = 2
    ) -> Tuple[float, float, float]:
        """计算布林带"""
        if len(series) < period:
            return 0, 0, 0
        
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return float(upper.iloc[-1]), float(middle.iloc[-1]), float(lower.iloc[-1])
    
    def _calculate_atr(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> float:
        """计算 ATR"""
        if len(close) < period + 1:
            return 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> float:
        """计算 OBV"""
        if len(close) < 2:
            return 0
        
        direction = np.where(close > close.shift(), 1, np.where(close < close.shift(), -1, 0))
        obv = (volume * direction).cumsum()
        
        return float(obv.iloc[-1])
    
    # ==================== 策略分析 ====================
    
    async def analyze(self, symbol: str, klines: pd.DataFrame) -> List[TradeSignal]:
        """
        分析市场并生成信号
        
        Args:
            symbol: 交易对
            klines: K线数据
            
        Returns:
            信号列表
        """
        signals = []
        
        # 计算指标
        indicators = self.calculate_indicators(klines)
        self._indicator_cache[symbol] = indicators
        
        # 更新市场环境
        context = self._build_context(symbol, klines, indicators)
        self._context_cache[symbol] = context
        
        # 运行各策略
        for strategy_type in self._active_strategies:
            strategy_signal = await self._run_strategy(
                strategy_type, symbol, klines, indicators, context
            )
            if strategy_signal:
                signals.append(strategy_signal)
        
        # 综合信号
        final_signal = self._combine_signals(signals, symbol)
        
        if final_signal:
            self._signal_history.append(final_signal)
            self._stats['signals_generated'] += 1
            
            # 更新统计
            signal_type = final_signal.signal_type
            self._stats['signals_by_type'][signal_type] = \
                self._stats['signals_by_type'].get(signal_type, 0) + 1
        
        return [final_signal] if final_signal else []
    
    def _build_context(
        self, 
        symbol: str, 
        klines: pd.DataFrame, 
        indicators: TechnicalIndicators
    ) -> MarketContext:
        """构建市场环境"""
        context = MarketContext(symbol=symbol, price=0)
        
        if klines.empty:
            return context
        
        close = klines['close']
        
        context.price = float(close.iloc[-1])
        
        if len(klines) >= 24:
            context.price_change_24h = (close.iloc[-1] - close.iloc[-24]) / close.iloc[-24] * 100
            context.price_high_24h = float(klines['high'].iloc[-24:].max())
            context.price_low_24h = float(klines['low'].iloc[-24:].min())
            context.volume_24h = float(klines['volume'].iloc[-24:].sum())
        
        # 判断趋势
        if indicators.ma5 > indicators.ma20 > indicators.ma60:
            context.trend = 'up'
        elif indicators.ma5 < indicators.ma20 < indicators.ma60:
            context.trend = 'down'
        else:
            context.trend = 'neutral'
        
        # 判断波动性
        if indicators.atr > 0 and context.price > 0:
            atr_pct = indicators.atr / context.price * 100
            if atr_pct > 3:
                context.volatility = 'high'
            elif atr_pct < 1:
                context.volatility = 'low'
        
        return context
    
    async def _run_strategy(
        self,
        strategy_type: StrategyType,
        symbol: str,
        klines: pd.DataFrame,
        indicators: TechnicalIndicators,
        context: MarketContext
    ) -> Optional[TradeSignal]:
        """运行单个策略"""
        if strategy_type == StrategyType.TREND:
            return self._trend_strategy(symbol, indicators, context)
        elif strategy_type == StrategyType.REVERSAL:
            return self._reversal_strategy(symbol, indicators, context)
        elif strategy_type == StrategyType.MOMENTUM:
            return self._momentum_strategy(symbol, indicators, context)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return self._mean_reversion_strategy(symbol, indicators, context)
        
        return None
    
    def _trend_strategy(
        self, 
        symbol: str, 
        indicators: TechnicalIndicators, 
        context: MarketContext
    ) -> Optional[TradeSignal]:
        """趋势策略：MA 多空排列 + MACD"""
        signal = None
        
        # 多头排列
        if (indicators.ma5 > indicators.ma20 > indicators.ma60 and
            indicators.macd_hist > 0):
            
            # MACD 金叉确认
            if indicators.macd > indicators.macd_signal:
                signal = TradeSignal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    signal_type='entry',
                    strength=SignalStrength.STRONG.value,
                    price=context.price,
                    stop_loss=context.price * 0.95,  # 5% 止损
                    take_profit=context.price * 1.10,  # 10% 止盈
                    reason='趋势多头：MA排列+MACD金叉',
                )
        
        # 空头排列
        elif (indicators.ma5 < indicators.ma20 < indicators.ma60 and
              indicators.macd_hist < 0):
            
            # MACD 死叉确认
            if indicators.macd < indicators.macd_signal:
                signal = TradeSignal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    signal_type='entry',
                    strength=SignalStrength.STRONG.value,
                    price=context.price,
                    stop_loss=context.price * 1.05,
                    take_profit=context.price * 0.90,
                    reason='趋势空头：MA排列+MACD死叉',
                )
        
        return signal
    
    def _reversal_strategy(
        self, 
        symbol: str, 
        indicators: TechnicalIndicators, 
        context: MarketContext
    ) -> Optional[TradeSignal]:
        """反弹策略：RSI 超买超卖"""
        signal = None
        
        # 超卖反弹
        if indicators.rsi < 30:
            # RSI 超卖，且价格接近布林带下轨
            price_vs_boll = (context.price - indicators.boll_lower) / indicators.boll_lower if indicators.boll_lower > 0 else 0
            
            if price_vs_boll < 0.02:  # 价格接近下轨
                signal = TradeSignal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    signal_type='entry',
                    strength=SignalStrength.MODERATE.value,
                    price=context.price,
                    stop_loss=context.price * 0.97,
                    take_profit=context.price * 1.05,
                    reason=f'超卖反弹：RSI={indicators.rsi:.1f}',
                )
        
        # 超买回落
        elif indicators.rsi > 70:
            price_vs_boll = (context.price - indicators.boll_upper) / indicators.boll_upper if indicators.boll_upper > 0 else 0
            
            if price_vs_boll > -0.02:  # 价格接近上轨
                signal = TradeSignal(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    signal_type='entry',
                    strength=SignalStrength.MODERATE.value,
                    price=context.price,
                    stop_loss=context.price * 1.03,
                    take_profit=context.price * 0.95,
                    reason=f'超买回落：RSI={indicators.rsi:.1f}',
                )
        
        return signal
    
    def _momentum_strategy(
        self, 
        symbol: str, 
        indicators: TechnicalIndicators, 
        context: MarketContext
    ) -> Optional[TradeSignal]:
        """动量策略：成交量突破"""
        signal = None
        
        # 放量上涨
        if (indicators.volume_ratio > 1.5 and  # 放量
            context.price_change_24h > 2 and   # 上涨
            indicators.macd_hist > 0):         # MACD 正向
            
            signal = TradeSignal(
                symbol=symbol,
                side=OrderSide.BUY,
                signal_type='entry',
                strength=min(0.6 + indicators.volume_ratio * 0.1, 0.9),
                price=context.price,
                stop_loss=context.price * 0.97,
                take_profit=context.price * 1.08,
                reason=f'动量突破：量比={indicators.volume_ratio:.2f}',
            )
        
        # 放量下跌
        elif (indicators.volume_ratio > 1.5 and
              context.price_change_24h < -2 and
              indicators.macd_hist < 0):
            
            signal = TradeSignal(
                symbol=symbol,
                side=OrderSide.SELL,
                signal_type='entry',
                strength=min(0.6 + indicators.volume_ratio * 0.1, 0.9),
                price=context.price,
                stop_loss=context.price * 1.03,
                take_profit=context.price * 0.92,
                reason=f'动量突破：量比={indicators.volume_ratio:.2f}',
            )
        
        return signal
    
    def _mean_reversion_strategy(
        self, 
        symbol: str, 
        indicators: TechnicalIndicators, 
        context: MarketContext
    ) -> Optional[TradeSignal]:
        """均值回归策略：价格偏离均线"""
        signal = None
        
        if indicators.ma20 == 0:
            return None
        
        deviation = (context.price - indicators.ma20) / indicators.ma20
        
        # 价格大幅低于均线，预期回归
        if deviation < -0.05:  # 低于均线 5%
            signal = TradeSignal(
                symbol=symbol,
                side=OrderSide.BUY,
                signal_type='entry',
                strength=SignalStrength.WEAK.value,
                price=context.price,
                stop_loss=context.price * 0.95,
                take_profit=indicators.ma20,
                reason=f'均值回归：偏离={deviation*100:.1f}%',
            )
        
        # 价格大幅高于均线，预期回调
        elif deviation > 0.05:  # 高于均线 5%
            signal = TradeSignal(
                symbol=symbol,
                side=OrderSide.SELL,
                signal_type='entry',
                strength=SignalStrength.WEAK.value,
                price=context.price,
                stop_loss=context.price * 1.05,
                take_profit=indicators.ma20,
                reason=f'均值回归：偏离={deviation*100:.1f}%',
            )
        
        return signal
    
    def _combine_signals(self, signals: List[TradeSignal], symbol: str) -> Optional[TradeSignal]:
        """综合多个信号"""
        if not signals:
            return None
        
        # 按权重综合
        total_strength = 0
        buy_strength = 0
        sell_strength = 0
        
        for signal in signals:
            strategy_weight = 1.0  # 可根据策略类型加权
            
            if signal.side == OrderSide.BUY:
                buy_strength += signal.strength * strategy_weight
            else:
                sell_strength += signal.strength * strategy_weight
            
            total_strength += signal.strength * strategy_weight
        
        # 判断最终方向
        if buy_strength > sell_strength and buy_strength > 0.4:
            final_side = OrderSide.BUY
            final_strength = buy_strength / len(signals)
            reasons = [s.reason for s in signals if s.side == OrderSide.BUY]
        elif sell_strength > buy_strength and sell_strength > 0.4:
            final_side = OrderSide.SELL
            final_strength = sell_strength / len(signals)
            reasons = [s.reason for s in signals if s.side == OrderSide.SELL]
        else:
            return None  # 信号不明确
        
        # 计算止盈止损
        prices = [s.price for s in signals if s.price]
        avg_price = sum(prices) / len(prices) if prices else 0
        
        return TradeSignal(
            symbol=symbol,
            side=final_side,
            signal_type='entry',
            strength=min(final_strength, 1.0),
            price=avg_price,
            stop_loss=avg_price * 0.95 if final_side == OrderSide.BUY else avg_price * 1.05,
            take_profit=avg_price * 1.10 if final_side == OrderSide.BUY else avg_price * 0.90,
            reason=' | '.join(reasons[:3]),  # 最多显示3个原因
        )
    
    # ==================== 仓位建议 ====================
    
    def suggest_position_size(
        self,
        symbol: str,
        total_equity: float,
        risk_per_trade: float = 0.02
    ) -> Dict[str, Any]:
        """
        建议仓位大小
        
        Args:
            symbol: 交易对
            total_equity: 总权益
            risk_per_trade: 单笔风险比例
            
        Returns:
            仓位建议
        """
        indicators = self._indicator_cache.get(symbol)
        context = self._context_cache.get(symbol)
        
        if not indicators or not context:
            return {
                'suggested_size_pct': 0.05,
                'reason': '数据不足',
            }
        
        # 基于 ATR 调整仓位
        atr_pct = indicators.atr / context.price if context.price > 0 else 0.02
        
        # 波动越大，仓位越小
        volatility_factor = 1.0
        if atr_pct > 0.03:
            volatility_factor = 0.5
        elif atr_pct > 0.02:
            volatility_factor = 0.75
        
        # 基于 RSI 调整
        rsi_factor = 1.0
        if 40 <= indicators.rsi <= 60:
            rsi_factor = 1.2  # 中性区域可加大
        elif indicators.rsi < 25 or indicators.rsi > 75:
            rsi_factor = 0.7  # 极端区域减小
        
        # 最终建议
        base_size = risk_per_trade * total_equity
        adjusted_size = base_size * volatility_factor * rsi_factor
        
        suggested_pct = adjusted_size / total_equity
        suggested_pct = min(suggested_pct, 0.30)  # 最大 30%
        
        return {
            'suggested_size_pct': suggested_pct,
            'suggested_size_value': adjusted_size,
            'volatility_factor': volatility_factor,
            'rsi_factor': rsi_factor,
            'atr_pct': atr_pct,
            'reason': f'基于波动{volatility_factor:.1f}x和RSI{rsi_factor:.1f}x调整',
        }
    
    # ==================== 状态查询 ====================
    
    def get_indicators(self, symbol: str) -> Optional[TechnicalIndicators]:
        """获取指标"""
        return self._indicator_cache.get(symbol)
    
    def get_context(self, symbol: str) -> Optional[MarketContext]:
        """获取市场环境"""
        return self._context_cache.get(symbol)
    
    def get_signal_history(self, limit: int = 50) -> List[TradeSignal]:
        """获取信号历史"""
        return self._signal_history[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'running': self._running,
            'active_strategies': [s.value for s in self._active_strategies],
            'cached_symbols': list(self._indicator_cache.keys()),
            'stats': self._stats,
        }
    
    def add_strategy(self, strategy_type: StrategyType):
        """添加策略"""
        if strategy_type not in self._active_strategies:
            self._active_strategies.append(strategy_type)
    
    def remove_strategy(self, strategy_type: StrategyType):
        """移除策略"""
        if strategy_type in self._active_strategies:
            self._active_strategies.remove(strategy_type)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'healthy': self._running,
            'active_strategies': len(self._active_strategies),
            'signals_today': self._stats['signals_generated'],
        }


# 导出
__all__ = [
    'StrategyAI',
    'StrategyType',
    'TechnicalIndicators',
    'MarketContext',
    'SignalStrength'
]
