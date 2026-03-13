"""
高级仓位管理 (Advanced Position Management)
============================================
凯利公式 + 动态止损止盈

核心改进：
1. 凯利公式计算最优仓位
2. 动态止损（基于ATR + 流动性热图）
3. 动态止盈（清算墙 + 阻力位）
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class PositionSizing:
    """仓位计算结果"""
    position_size: float  # 仓位比例 0-1
    kelly_fraction: float  # 凯利比例
    risk_amount: float  # 风险金额
    stop_loss_price: float  # 止损价
    take_profit_price: float  # 止盈价
    risk_reward_ratio: float  # 风险收益比
    
    # 动态参数
    stop_type: str  # "atr" / "liquidity" / "liquidation"
    tp_type: str  # "resistance" / "liquidation" / "fixed"
    
    # 元信息
    confidence: float
    expected_value: float  # 期望值


class KellyPositionManager:
    """
    凯利公式仓位管理器
    
    凯利公式: f* = (p * b - q) / b
    - p = 胜率
    - q = 1 - p (败率)
    - b = 盈亏比
    
    实际使用时用 Kelly / 2 或 Kelly / 3（半凯利）降低风险
    """
    
    def __init__(self, kelly_divisor: float = 3.0):
        """
        Args:
            kelly_divisor: 凯利除数，降低仓位风险
        """
        self.kelly_divisor = kelly_divisor
    
    def calculate_kelly(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
    ) -> float:
        """
        计算凯利比例
        
        Args:
            win_rate: 胜率 (0-1)
            avg_win_pct: 平均盈利百分比
            avg_loss_pct: 平均亏损百分比
        
        Returns:
            凯利比例 (0-1)
        """
        if avg_loss_pct == 0:
            return 0
        
        # 盈亏比 b = avg_win / avg_loss
        b = avg_win_pct / avg_loss_pct
        
        # 凯利公式
        p = win_rate
        q = 1 - p
        
        kelly = (p * b - q) / b
        
        # 限制范围
        kelly = max(0, min(kelly, 0.25))  # 最大25%
        
        # 应用除数（半凯利）
        kelly_adjusted = kelly / self.kelly_divisor
        
        return kelly_adjusted
    
    def calculate_position(
        self,
        account_balance: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        signal_confidence: float,
        risk_per_trade: float = 0.02,  # 单笔最大风险2%
    ) -> PositionSizing:
        """
        计算仓位
        
        Args:
            account_balance: 账户余额
            win_rate: 胜率
            avg_win_pct: 平均盈利%
            avg_loss_pct: 平均亏损%
            signal_confidence: 信号置信度 (0-1)
            risk_per_trade: 单笔最大风险比例
        
        Returns:
            PositionSizing
        """
        # 凯利比例
        kelly = self.calculate_kelly(win_rate, avg_win_pct, avg_loss_pct)
        
        # 置信度调整
        kelly_adjusted = kelly * signal_confidence
        
        # 最大风险限制
        max_position_by_risk = risk_per_trade / (avg_loss_pct / 100) if avg_loss_pct > 0 else 0.02
        
        # 最终仓位取较小值
        position_size = min(kelly_adjusted, max_position_by_risk, 0.30)  # 最大30%
        
        # 风险金额
        risk_amount = account_balance * position_size * (avg_loss_pct / 100)
        
        # 期望值
        expected_value = win_rate * avg_win_pct - (1 - win_rate) * avg_loss_pct
        
        return PositionSizing(
            position_size=position_size,
            kelly_fraction=kelly,
            risk_amount=risk_amount,
            stop_loss_price=0,  # 后续计算
            take_profit_price=0,
            risk_reward_ratio=avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0,
            stop_type="calculated",
            tp_type="calculated",
            confidence=signal_confidence,
            expected_value=expected_value,
        )


class DynamicStopLoss:
    """
    动态止损计算器
    
    策略：
    1. ATR止损：入场价 - N * ATR
    2. 流动性止损：最近强支撑下方
    3. 清算止损：清算墙外
    """
    
    # 止损类型优先级
    STOP_PRIORITY = ["liquidation", "liquidity", "atr"]
    
    def calculate_stop_loss(
        self,
        direction: str,  # "long" / "short"
        entry_price: float,
        atr: float,
        support_zones: List[Tuple[float, float]],
        resistance_zones: List[Tuple[float, float]],
        liquidation_zones: List[Dict],
        atr_multiplier: float = 2.0,
    ) -> Tuple[float, str]:
        """
        计算动态止损
        
        Returns:
            (止损价, 止损类型)
        """
        if direction == "long":
            return self._calculate_long_stop(
                entry_price, atr, support_zones, liquidation_zones, atr_multiplier
            )
        else:
            return self._calculate_short_stop(
                entry_price, atr, resistance_zones, liquidation_zones, atr_multiplier
            )
    
    def _calculate_long_stop(
        self,
        entry_price: float,
        atr: float,
        support_zones: List[Tuple[float, float]],
        liquidation_zones: List[Dict],
        atr_multiplier: float,
    ) -> Tuple[float, str]:
        """做多止损"""
        stops = []
        
        # 1. ATR止损
        atr_stop = entry_price - atr * atr_multiplier
        stops.append(("atr", atr_stop))
        
        # 2. 流动性止损（最近支撑下方）
        for price, strength in support_zones[:2]:
            if price < entry_price:
                # 支撑下方 0.5%
                liq_stop = price * 0.995
                stops.append(("liquidity", liq_stop))
                break
        
        # 3. 清算止损（清算墙外）
        for liq in liquidation_zones[:3]:
            liq_price = liq.get('price', 0)
            liq_dir = liq.get('direction', '')
            if liq_price < entry_price and liq_dir == 'short':
                # 空头清算墙下方（可能被扫后反弹）
                liq_stop = liq_price * 0.998
                stops.append(("liquidation", liq_stop))
                break
        
        # 选择最优（最高）止损价
        if not stops:
            return atr_stop, "atr"
        
        best_stop = max(stops, key=lambda x: x[1])
        return best_stop[1], best_stop[0]
    
    def _calculate_short_stop(
        self,
        entry_price: float,
        atr: float,
        resistance_zones: List[Tuple[float, float]],
        liquidation_zones: List[Dict],
        atr_multiplier: float,
    ) -> Tuple[float, str]:
        """做空止损"""
        stops = []
        
        # 1. ATR止损
        atr_stop = entry_price + atr * atr_multiplier
        stops.append(("atr", atr_stop))
        
        # 2. 流动性止损（最近阻力上方）
        for price, strength in resistance_zones[:2]:
            if price > entry_price:
                liq_stop = price * 1.005
                stops.append(("liquidity", liq_stop))
                break
        
        # 3. 清算止损
        for liq in liquidation_zones[:3]:
            liq_price = liq.get('price', 0)
            liq_dir = liq.get('direction', '')
            if liq_price > entry_price and liq_dir == 'long':
                liq_stop = liq_price * 1.002
                stops.append(("liquidation", liq_stop))
                break
        
        if not stops:
            return atr_stop, "atr"
        
        best_stop = min(stops, key=lambda x: x[1])
        return best_stop[1], best_stop[0]


class DynamicTakeProfit:
    """
    动态止盈计算器
    
    策略：
    1. 固定盈亏比：止损 * RR
    2. 阻力位止盈
    3. 清算墙止盈
    """
    
    MIN_RR = 1.5  # 最小盈亏比
    
    def calculate_take_profit(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        resistance_zones: List[Tuple[float, float]],
        support_zones: List[Tuple[float, float]],
        liquidation_zones: List[Dict],
        min_rr: float = 1.5,
    ) -> Tuple[float, str]:
        """
        计算动态止盈
        
        Returns:
            (止盈价, 止盈类型)
        """
        # 基础风险
        risk = abs(entry_price - stop_loss)
        
        if direction == "long":
            return self._calculate_long_tp(
                entry_price, risk, resistance_zones, liquidation_zones, min_rr
            )
        else:
            return self._calculate_short_tp(
                entry_price, risk, support_zones, liquidation_zones, min_rr
            )
    
    def _calculate_long_tp(
        self,
        entry_price: float,
        risk: float,
        resistance_zones: List[Tuple[float, float]],
        liquidation_zones: List[Dict],
        min_rr: float,
    ) -> Tuple[float, str]:
        """做多止盈"""
        tps = []
        
        # 1. 固定盈亏比
        fixed_tp = entry_price + risk * min_rr
        tps.append(("fixed_rr", fixed_tp))
        
        # 2. 阻力位止盈
        for price, strength in resistance_zones[:2]:
            if price > entry_price:
                rr_at_resistance = (price - entry_price) / risk if risk > 0 else 0
                if rr_at_resistance >= min_rr:
                    tps.append(("resistance", price))
                break
        
        # 3. 清算墙止盈（多头清算墙 = 空头被扫）
        for liq in liquidation_zones[:3]:
            liq_price = liq.get('price', 0)
            liq_dir = liq.get('direction', '')
            if liq_price > entry_price and liq_dir == 'long':
                # 多头清算墙 = 价格可能继续涨
                tps.append(("liquidation", liq_price))
                break
        
        # 选择最近的合理止盈
        valid_tps = [(t, p) for t, p in tps if p > entry_price]
        if not valid_tps:
            return fixed_tp, "fixed_rr"
        
        best_tp = min(valid_tps, key=lambda x: x[1])
        return best_tp[1], best_tp[0]
    
    def _calculate_short_tp(
        self,
        entry_price: float,
        risk: float,
        support_zones: List[Tuple[float, float]],
        liquidation_zones: List[Dict],
        min_rr: float,
    ) -> Tuple[float, str]:
        """做空止盈"""
        tps = []
        
        # 1. 固定盈亏比
        fixed_tp = entry_price - risk * min_rr
        tps.append(("fixed_rr", fixed_tp))
        
        # 2. 支撑位止盈
        for price, strength in support_zones[:2]:
            if price < entry_price:
                rr_at_support = (entry_price - price) / risk if risk > 0 else 0
                if rr_at_support >= min_rr:
                    tps.append(("support", price))
                break
        
        # 3. 清算墙止盈
        for liq in liquidation_zones[:3]:
            liq_price = liq.get('price', 0)
            liq_dir = liq.get('direction', '')
            if liq_price < entry_price and liq_dir == 'short':
                tps.append(("liquidation", liq_price))
                break
        
        valid_tps = [(t, p) for t, p in tps if p < entry_price]
        if not valid_tps:
            return fixed_tp, "fixed_rr"
        
        best_tp = max(valid_tps, key=lambda x: x[1])
        return best_tp[1], best_tp[0]


def calculate_optimal_position(
    account_balance: float,
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    signal_confidence: float,
    entry_price: float,
    direction: str,
    atr: float,
    support_zones: List[Tuple[float, float]],
    resistance_zones: List[Tuple[float, float]],
    liquidation_zones: List[Dict],
) -> PositionSizing:
    """
    计算最优仓位和止损止盈
    
    整合：凯利公式 + 动态止损 + 动态止盈
    """
    # 凯利仓位管理器
    kelly = KellyPositionManager(kelly_divisor=3.0)
    sizing = kelly.calculate_position(
        account_balance=account_balance,
        win_rate=win_rate,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        signal_confidence=signal_confidence,
    )
    
    # 动态止损
    stop_calc = DynamicStopLoss()
    stop_loss, stop_type = stop_calc.calculate_stop_loss(
        direction=direction,
        entry_price=entry_price,
        atr=atr,
        support_zones=support_zones,
        resistance_zones=resistance_zones,
        liquidation_zones=liquidation_zones,
    )
    
    # 动态止盈
    tp_calc = DynamicTakeProfit()
    take_profit, tp_type = tp_calc.calculate_take_profit(
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        resistance_zones=resistance_zones,
        support_zones=support_zones,
        liquidation_zones=liquidation_zones,
    )
    
    # 计算实际盈亏比
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    actual_rr = reward / risk if risk > 0 else 0
    
    # 更新结果
    sizing.stop_loss_price = stop_loss
    sizing.take_profit_price = take_profit
    sizing.risk_reward_ratio = actual_rr
    sizing.stop_type = stop_type
    sizing.tp_type = tp_type
    
    return sizing
