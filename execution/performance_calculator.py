# execution/performance_calculator.py
"""
性能计算模块 - 精准稳定
=======================

功能：
- 精确收益计算
- 风险指标计算
- 实时性能监控
- USDT统一计价
"""

import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 收益指标
    total_return: float = 0.0          # 总收益率
    annualized_return: float = 0.0     # 年化收益率
    cagr: float = 0.0                  # 复合年增长率
    
    # 风险指标
    volatility: float = 0.0            # 年化波动率
    max_drawdown: float = 0.0          # 最大回撤
    max_drawdown_duration: int = 0     # 最大回撤持续天数
    var_95: float = 0.0                # 95% VaR
    var_99: float = 0.0                # 99% VaR
    
    # 风险调整收益
    sharpe_ratio: float = 0.0          # 夏普比率
    sortino_ratio: float = 0.0         # 索提诺比率
    calmar_ratio: float = 0.0          # 卡玛比率
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # 资金
    total_equity: float = 0.0
    available_balance: float = 0.0
    margin_used: float = 0.0


class PrecisionCalculator:
    """
    精确计算器
    
    使用 Decimal 确保金融计算精度
    """
    
    # 精度设置
    PRICE_DECIMALS = 8      # 价格精度
    QUANTITY_DECIMALS = 8   # 数量精度
    PERCENTAGE_DECIMALS = 6 # 百分比精度
    
    @staticmethod
    def to_decimal(value: float, decimals: int = 8) -> Decimal:
        """转换为 Decimal"""
        return Decimal(str(value)).quantize(
            Decimal(10) ** -decimals,
            rounding=ROUND_HALF_UP
        )
    
    @staticmethod
    def calculate_return(
        start_value: float, 
        end_value: float,
        decimals: int = 8
    ) -> float:
        """
        计算收益率
        
        使用 Decimal 确保精度
        """
        if start_value == 0:
            return 0.0
        
        start = Decimal(str(start_value))
        end = Decimal(str(end_value))
        
        return_pct = (end - start) / start
        return float(return_pct.quantize(
            Decimal(10) ** -decimals,
            rounding=ROUND_HALF_UP
        ))
    
    @staticmethod
    def calculate_pnl(
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str,  # 'long' or 'short'
        fee_rate: float = 0.001
    ) -> Tuple[float, float]:
        """
        计算盈亏
        
        Args:
            entry_price: 入场价
            exit_price: 出场价
            quantity: 数量
            side: 方向
            fee_rate: 手续费率
            
        Returns:
            (盈亏金额, 手续费)
        """
        entry = Decimal(str(entry_price))
        exit_p = Decimal(str(exit_price))
        qty = Decimal(str(quantity))
        fee = Decimal(str(fee_rate))
        
        # 计算价差盈亏
        if side == 'long':
            price_pnl = (exit_p - entry) * qty
        else:
            price_pnl = (entry - exit_p) * qty
        
        # 计算手续费
        entry_fee = entry * qty * fee
        exit_fee = exit_p * qty * fee
        total_fee = entry_fee + exit_fee
        
        # 净盈亏
        net_pnl = price_pnl - total_fee
        
        return float(net_pnl), float(total_fee)
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 365
    ) -> float:
        """
        计算夏普比率
        
        公式: (年化收益 - 无风险利率) / 年化波动率
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_arr = np.array(returns)
        
        # 年化收益率（几何平均）
        mean_return = np.mean(returns_arr)
        annualized_return = (1 + mean_return) ** periods_per_year - 1
        
        # 年化波动率
        std_return = np.std(returns_arr, ddof=1)
        annualized_vol = std_return * np.sqrt(periods_per_year)
        
        if annualized_vol == 0:
            return 0.0
        
        # 夏普比率
        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        
        return round(sharpe, 6)
    
    @staticmethod
    def calculate_max_drawdown(
        equity_curve: List[float]
    ) -> Tuple[float, int]:
        """
        计算最大回撤
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            (最大回撤比例, 持续天数)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0
        
        equity = np.array(equity_curve)
        peak = equity[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_start = 0
        
        for i, value in enumerate(equity):
            if value > peak:
                peak = value
                current_dd_start = i
            else:
                dd = (peak - value) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_duration = i - current_dd_start
        
        return round(max_dd, 6), max_dd_duration


class RealtimePerformanceTracker:
    """
    实时性能追踪器
    
    功能：
    - 实时更新性能指标
    - 追踪交易历史
    - 计算滚动指标
    """
    
    def __init__(self, initial_equity: float = 10000.0):
        self.initial_equity = initial_equity
        self.calculator = PrecisionCalculator()
        
        # 权益历史
        self.equity_history: List[Dict] = [{
            'timestamp': datetime.now().isoformat(),
            'equity': initial_equity,
            'pnl': 0.0,
        }]
        
        # 交易历史
        self.trade_history: List[Dict] = []
        
        # 持仓
        self.positions: Dict[str, Dict] = {}
        
        # 当前权益
        self.current_equity = initial_equity
        
        # 统计
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
        }
    
    def update_equity(self, equity: float):
        """更新权益"""
        self.current_equity = equity
        last_equity = self.equity_history[-1]['equity'] if self.equity_history else equity
        
        self.equity_history.append({
            'timestamp': datetime.now().isoformat(),
            'equity': equity,
            'pnl': equity - last_equity,
        })
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        fee: float = 0.0,
        pnl: float = 0.0
    ):
        """记录交易"""
        is_win = pnl > 0
        
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'fee': fee,
            'pnl': pnl,
            'is_win': is_win,
        })
        
        # 更新统计
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += pnl
        self.stats['total_fees'] += fee
        
        if is_win:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
    
    def get_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        metrics = PerformanceMetrics()
        
        # 权益曲线
        equity_values = [h['equity'] for h in self.equity_history]
        
        if len(equity_values) > 1:
            # 总收益
            metrics.total_return = self.calculator.calculate_return(
                equity_values[0], equity_values[-1]
            )
            
            # 年化收益
            days = len(equity_values) / 288  # 假设5分钟间隔
            if days > 0:
                metrics.annualized_return = (1 + metrics.total_return) ** (365 / days) - 1
            
            # 最大回撤
            metrics.max_drawdown, metrics.max_drawdown_duration = \
                self.calculator.calculate_max_drawdown(equity_values)
            
            # 波动率
            returns = [h['pnl'] / self.initial_equity for h in self.equity_history[1:] if h['pnl'] != 0]
            if returns:
                metrics.volatility = np.std(returns) * np.sqrt(365 * 288)
                metrics.sharpe_ratio = self.calculator.calculate_sharpe_ratio(returns)
        
        # 交易统计
        metrics.total_trades = self.stats['total_trades']
        metrics.winning_trades = self.stats['winning_trades']
        metrics.losing_trades = self.stats['losing_trades']
        metrics.win_rate = self.stats['winning_trades'] / self.stats['total_trades'] if self.stats['total_trades'] > 0 else 0
        
        # 盈亏统计
        wins = [t['pnl'] for t in self.trade_history if t['is_win']]
        losses = [t['pnl'] for t in self.trade_history if not t['is_win']]
        
        metrics.avg_win = np.mean(wins) if wins else 0
        metrics.avg_loss = np.mean(losses) if losses else 0
        metrics.largest_win = max(wins) if wins else 0
        metrics.largest_loss = min(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 资金
        metrics.total_equity = self.current_equity
        
        return metrics
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """获取权益曲线 DataFrame"""
        return pd.DataFrame(self.equity_history)
    
    def get_trade_dataframe(self) -> pd.DataFrame:
        """获取交易记录 DataFrame"""
        return pd.DataFrame(self.trade_history)


# 导出
__all__ = [
    'PerformanceMetrics',
    'PrecisionCalculator',
    'RealtimePerformanceTracker',
]
