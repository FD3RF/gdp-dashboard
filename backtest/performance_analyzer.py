"""
Performance Analyzer for backtest results.

IMPORTANT: 性能指标计算说明
============================
- 所有价格以 USDT 计价
- 年化收益使用几何平均: (1+R)^(365/days) - 1
- Sharpe 比率: (年化收益 - 无风险利率) / 年化波动率
- Sortino 比率: 只考虑下行风险
- 加密货币全年 365 天交易
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from core.base import BaseModule

# 导入精准计算器
from .precise_performance import (
    PrecisePerformanceCalculator,
    PerformanceMetrics,
    PriceConverter
)


class PerformanceAnalyzer(BaseModule):
    """
    Analyzes trading performance and calculates metrics.
    
    使用机构级计算方法确保精准稳定。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('performance_analyzer', config)
        
        self._risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self._trading_days = self.config.get('trading_days', 365)
        
        # 使用精准计算器
        self._precise_calculator = PrecisePerformanceCalculator(
            risk_free_rate=self._risk_free_rate
        )
        
        # 价格转换器（确保 USDT 计价）
        self._price_converter = PriceConverter()
    
    async def initialize(self) -> bool:
        """Initialize the analyzer."""
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the analyzer."""
        self._running = True
        return True
    
    async def stop(self) -> bool:
        """Stop the analyzer."""
        self._running = False
        return True
    
    def analyze_precise(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Dict] = None,
        benchmark_returns: pd.Series = None
    ) -> PerformanceMetrics:
        """
        使用精准计算器分析性能
        
        推荐：使用此方法获得机构级精准指标
        
        Args:
            equity_curve: 权益曲线 DataFrame
            trades: 交易列表
            benchmark_returns: 基准收益率
            
        Returns:
            PerformanceMetrics 对象
        """
        return self._precise_calculator.calculate_metrics(
            equity_curve, trades, benchmark_returns
        )
    
    def analyze(
        self,
        trades: List[Dict],
        equity_curve: List[Dict],
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        Analyze backtest performance.
        
        Args:
            trades: List of trades
            equity_curve: List of equity points
            initial_capital: Initial capital
        
        Returns:
            Performance metrics
        """
        if not trades or not equity_curve:
            return {'error': 'No data to analyze'}
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculate metrics
        metrics = {
            'initial_capital': initial_capital,
            'final_capital': equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial_capital,
            'total_return': self._calculate_total_return(equity_df, initial_capital),
            'annualized_return': self._calculate_annualized_return(equity_df),
            'sharpe_ratio': self._calculate_sharpe_ratio(equity_df),
            'sortino_ratio': self._calculate_sortino_ratio(equity_df),
            'max_drawdown': self._calculate_max_drawdown(equity_df),
            'max_drawdown_duration': self._calculate_max_drawdown_duration(equity_df),
            'volatility': self._calculate_volatility(equity_df),
            'win_rate': self._calculate_win_rate(trades_df),
            'profit_factor': self._calculate_profit_factor(trades_df),
            'avg_trade_return': self._calculate_avg_trade_return(trades_df),
            'total_trades': len(trades),
            'winning_trades': len(trades_df[trades_df.get('pnl', 0) > 0]) if 'pnl' in trades_df.columns else 0,
            'losing_trades': len(trades_df[trades_df.get('pnl', 0) < 0]) if 'pnl' in trades_df.columns else 0,
        }
        
        # Additional metrics
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(metrics)
        metrics['avg_win'] = self._calculate_avg_win(trades_df)
        metrics['avg_loss'] = self._calculate_avg_loss(trades_df)
        metrics['largest_win'] = self._calculate_largest_win(trades_df)
        metrics['largest_loss'] = self._calculate_largest_loss(trades_df)
        metrics['avg_trade_duration'] = self._calculate_avg_trade_duration(trades_df)
        
        # Trade statistics
        metrics['total_fees'] = trades_df['fee'].sum() if 'fee' in trades_df.columns else 0
        
        return metrics
    
    def _calculate_total_return(self, equity_df: pd.DataFrame, initial: float) -> float:
        """Calculate total return."""
        if len(equity_df) == 0:
            return 0
        final = equity_df['equity'].iloc[-1]
        return (final - initial) / initial
    
    def _calculate_annualized_return(self, equity_df: pd.DataFrame) -> float:
        """Calculate annualized return."""
        if len(equity_df) < 2:
            return 0
        
        start = equity_df['timestamp'].iloc[0]
        end = equity_df['timestamp'].iloc[-1]
        
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)
        
        days = (end - start).days
        if days == 0:
            return 0
        
        total_return = self._calculate_total_return(equity_df, equity_df['equity'].iloc[0])
        return (1 + total_return) ** (365 / days) - 1
    
    def _calculate_sharpe_ratio(self, equity_df: pd.DataFrame) -> float:
        """
        Calculate Sharpe ratio.
        
        正确公式：Sharpe = (年化收益 - 无风险利率) / 年化波动率
        使用几何平均年化收益
        """
        if len(equity_df) < 2:
            return 0
        
        returns = equity_df['equity'].pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        # 年化收益（几何平均）
        annualized_return = (1 + returns.mean()) ** self._trading_days - 1
        
        # 年化波动率
        annualized_vol = returns.std() * np.sqrt(self._trading_days)
        
        # 夏普比率
        excess_return = annualized_return - self._risk_free_rate
        
        return excess_return / annualized_vol
    
    def _calculate_sortino_ratio(self, equity_df: pd.DataFrame) -> float:
        """
        Calculate Sortino ratio.
        
        只考虑下行风险
        """
        if len(equity_df) < 2:
            return 0
        
        returns = equity_df['equity'].pct_change().dropna()
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        # 年化收益
        annualized_return = (1 + returns.mean()) ** self._trading_days - 1
        
        # 下行标准差
        downside_std = negative_returns.std() * np.sqrt(self._trading_days)
        
        if downside_std == 0:
            return 0
        
        excess_return = annualized_return - self._risk_free_rate
        
        return excess_return / downside_std
        
        excess_return = returns.mean() * self._trading_days - self._risk_free_rate
        return excess_return / downside_std
    
    def _calculate_max_drawdown(self, equity_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if len(equity_df) == 0:
            return 0
        
        equity = equity_df['equity']
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        
        return float(drawdown.min())
    
    def _calculate_max_drawdown_duration(self, equity_df: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in days."""
        if len(equity_df) < 2:
            return 0
        
        equity = equity_df['equity']
        rolling_max = equity.cummax()
        
        # Find drawdown periods
        in_drawdown = equity < rolling_max
        
        if not in_drawdown.any():
            return 0
        
        # Calculate duration
        max_duration = 0
        current_duration = 0
        
        for val in in_drawdown:
            if val:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_volatility(self, equity_df: pd.DataFrame) -> float:
        """Calculate annualized volatility."""
        if len(equity_df) < 2:
            return 0
        
        returns = equity_df['equity'].pct_change().dropna()
        return returns.std() * np.sqrt(self._trading_days)
    
    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate win rate."""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0
        
        wins = (trades_df['pnl'] > 0).sum()
        total = len(trades_df[trades_df['pnl'] != 0])
        
        return wins / total if total > 0 else 0
    
    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """Calculate profit factor."""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_avg_trade_return(self, trades_df: pd.DataFrame) -> float:
        """Calculate average trade return."""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0
        
        return trades_df['pnl'].mean()
    
    def _calculate_calmar_ratio(self, metrics: Dict) -> float:
        """Calculate Calmar ratio."""
        if metrics['max_drawdown'] == 0:
            return 0
        
        return metrics['annualized_return'] / abs(metrics['max_drawdown'])
    
    def _calculate_avg_win(self, trades_df: pd.DataFrame) -> float:
        """Calculate average winning trade."""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0
        
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        return wins.mean() if len(wins) > 0 else 0
    
    def _calculate_avg_loss(self, trades_df: pd.DataFrame) -> float:
        """Calculate average losing trade."""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0
        
        losses = trades_df[trades_df['pnl'] < 0]['pnl']
        return losses.mean() if len(losses) > 0 else 0
    
    def _calculate_largest_win(self, trades_df: pd.DataFrame) -> float:
        """Calculate largest winning trade."""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0
        
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        return wins.max() if len(wins) > 0 else 0
    
    def _calculate_largest_loss(self, trades_df: pd.DataFrame) -> float:
        """Calculate largest losing trade."""
        if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
            return 0
        
        losses = trades_df[trades_df['pnl'] < 0]['pnl']
        return losses.min() if len(losses) > 0 else 0
    
    def _calculate_avg_trade_duration(self, trades_df: pd.DataFrame) -> float:
        """Calculate average trade duration in hours."""
        # This would need proper entry/exit matching
        return 0
