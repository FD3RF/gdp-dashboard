"""
精准性能计算模块
==================

提供机构级的性能指标计算，确保数据精准稳定。
支持多币种统一转换为 USDT 计价。
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    # 收益指标
    total_return: float = 0.0  # 总收益率
    annualized_return: float = 0.0  # 年化收益率
    cagr: float = 0.0  # 复合年增长率
    
    # 风险指标
    volatility: float = 0.0  # 年化波动率
    max_drawdown: float = 0.0  # 最大回撤
    max_drawdown_duration: int = 0  # 最大回撤持续天数
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    cvar: float = 0.0  # 条件风险价值
    
    # 风险调整收益
    sharpe_ratio: float = 0.0  # 夏普比率
    sortino_ratio: float = 0.0  # 索提诺比率
    calmar_ratio: float = 0.0  # 卡玛比率
    information_ratio: float = 0.0  # 信息比率
    
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
    avg_trade_duration: float = 0.0
    
    # 费用
    total_fees: float = 0.0
    total_slippage: float = 0.0


class PrecisePerformanceCalculator:
    """
    精准性能计算器
    
    特点：
    - 使用 Decimal 确保精度
    - 正确的年化计算
    - 支持 USDT 统一计价
    - 机构级指标计算
    """
    
    # 常量
    TRADING_DAYS_PER_YEAR = 365  # 加密货币全年交易
    TRADING_HOURS_PER_DAY = 24
    RISK_FREE_RATE = 0.02  # 2% 无风险利率
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger("PerformanceCalculator")
    
    def calculate_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Dict] = None,
        benchmark_returns: pd.Series = None
    ) -> PerformanceMetrics:
        """
        计算完整性能指标
        
        Args:
            equity_curve: 权益曲线 DataFrame，需包含 'timestamp' 和 'equity' 列
            trades: 交易列表（可选）
            benchmark_returns: 基准收益率（可选）
            
        Returns:
            PerformanceMetrics 对象
        """
        metrics = PerformanceMetrics()
        
        if equity_curve.empty or len(equity_curve) < 2:
            self.logger.warning("权益曲线数据不足")
            return metrics
        
        # 确保数据格式正确
        equity_curve = self._prepare_equity_curve(equity_curve)
        
        # 计算收益率序列
        returns = self._calculate_returns(equity_curve)
        
        if returns.empty:
            return metrics
        
        # 1. 收益指标
        metrics.total_return = self._calculate_total_return(equity_curve)
        metrics.annualized_return = self._calculate_annualized_return(equity_curve, returns)
        metrics.cagr = self._calculate_cagr(equity_curve)
        
        # 2. 风险指标
        metrics.volatility = self._calculate_volatility(returns)
        metrics.max_drawdown, metrics.max_drawdown_duration = self._calculate_drawdown(equity_curve)
        metrics.var_95 = self._calculate_var(returns, 0.95)
        metrics.var_99 = self._calculate_var(returns, 0.99)
        metrics.cvar = self._calculate_cvar(returns, 0.95)
        
        # 3. 风险调整收益
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
        metrics.calmar_ratio = self._calculate_calmar_ratio(
            metrics.annualized_return, 
            metrics.max_drawdown
        )
        
        if benchmark_returns is not None:
            metrics.information_ratio = self._calculate_information_ratio(
                returns, benchmark_returns
            )
        
        # 4. 交易统计
        if trades:
            trade_metrics = self._calculate_trade_statistics(trades)
            metrics.total_trades = trade_metrics['total_trades']
            metrics.winning_trades = trade_metrics['winning_trades']
            metrics.losing_trades = trade_metrics['losing_trades']
            metrics.win_rate = trade_metrics['win_rate']
            metrics.profit_factor = trade_metrics['profit_factor']
            metrics.avg_win = trade_metrics['avg_win']
            metrics.avg_loss = trade_metrics['avg_loss']
            metrics.largest_win = trade_metrics['largest_win']
            metrics.largest_loss = trade_metrics['largest_loss']
            metrics.total_fees = trade_metrics['total_fees']
        
        return metrics
    
    def _prepare_equity_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备权益曲线数据"""
        df = df.copy()
        
        # 确保时间戳格式
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 确保权益列存在
        if 'equity' not in df.columns:
            if 'value' in df.columns:
                df['equity'] = df['value']
            else:
                raise ValueError("权益曲线必须包含 'equity' 或 'value' 列")
        
        return df
    
    def _calculate_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """计算收益率序列"""
        returns = equity_curve['equity'].pct_change().dropna()
        # 移除异常值
        returns = returns[returns.abs() < 1]  # 单日收益不超过100%
        return returns
    
    def _calculate_total_return(self, equity_curve: pd.DataFrame) -> float:
        """计算总收益率"""
        initial = equity_curve['equity'].iloc[0]
        final = equity_curve['equity'].iloc[-1]
        
        if initial == 0:
            return 0.0
        
        return float(Decimal(str(final)) / Decimal(str(initial)) - 1)
    
    def _calculate_annualized_return(
        self, 
        equity_curve: pd.DataFrame,
        returns: pd.Series
    ) -> float:
        """
        计算年化收益率（正确方法）
        
        使用几何平均而非算术平均
        """
        if len(equity_curve) < 2:
            return 0.0
        
        # 计算时间跨度
        start = equity_curve.index[0]
        end = equity_curve.index[-1]
        days = (end - start).days
        
        if days == 0:
            return 0.0
        
        # 计算总收益
        total_return = self._calculate_total_return(equity_curve)
        
        # 年化： (1 + R)^(365/days) - 1
        try:
            annualized = (1 + total_return) ** (365 / days) - 1
            return float(annualized)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_cagr(self, equity_curve: pd.DataFrame) -> float:
        """计算复合年增长率 (CAGR)"""
        if len(equity_curve) < 2:
            return 0.0
        
        start_value = equity_curve['equity'].iloc[0]
        end_value = equity_curve['equity'].iloc[-1]
        
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        years = (end_date - start_date).days / 365
        
        if years == 0 or start_value == 0:
            return 0.0
        
        try:
            cagr = (end_value / start_value) ** (1 / years) - 1
            return float(cagr)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """计算年化波动率"""
        if returns.empty:
            return 0.0
        
        # 日波动率 * sqrt(365)
        return float(returns.std() * np.sqrt(self.TRADING_DAYS_PER_YEAR))
    
    def _calculate_drawdown(
        self, 
        equity_curve: pd.DataFrame
    ) -> Tuple[float, int]:
        """计算最大回撤和持续时间"""
        equity = equity_curve['equity']
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        
        max_dd = float(drawdown.min())
        
        # 计算最大回撤持续时间
        in_drawdown = drawdown < 0
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_dd, max_duration
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """计算风险价值 (VaR)"""
        if returns.empty:
            return 0.0
        
        # 历史模拟法
        return float(np.percentile(returns, (1 - confidence) * 100))
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """计算条件风险价值 (CVaR/ES)"""
        if returns.empty:
            return 0.0
        
        var = self._calculate_var(returns, confidence)
        # 取低于 VaR 的收益的均值
        return float(returns[returns <= var].mean())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        计算夏普比率（正确方法）
        
        Sharpe = (年化收益 - 无风险利率) / 年化波动率
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        # 年化收益率（几何）
        avg_return = returns.mean()
        annualized_return = (1 + avg_return) ** 365 - 1
        
        # 年化波动率
        annualized_vol = returns.std() * np.sqrt(365)
        
        if annualized_vol == 0:
            return 0.0
        
        # 夏普比率
        daily_rf = self.risk_free_rate / 365
        excess_return = annualized_return - self.risk_free_rate
        
        return float(excess_return / annualized_vol)
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        计算索提诺比率（正确方法）
        
        只考虑下行风险
        """
        if returns.empty:
            return 0.0
        
        # 年化收益率
        avg_return = returns.mean()
        annualized_return = (1 + avg_return) ** 365 - 1
        
        # 下行标准差
        negative_returns = returns[returns < 0]
        
        if negative_returns.empty:
            return float('inf')
        
        downside_std = negative_returns.std() * np.sqrt(365)
        
        if downside_std == 0:
            return 0.0
        
        excess_return = annualized_return - self.risk_free_rate
        
        return float(excess_return / downside_std)
    
    def _calculate_calmar_ratio(
        self, 
        annualized_return: float, 
        max_drawdown: float
    ) -> float:
        """计算卡玛比率"""
        if max_drawdown == 0:
            return 0.0
        
        return float(annualized_return / abs(max_drawdown))
    
    def _calculate_information_ratio(
        self, 
        returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """计算信息比率"""
        if returns.empty or benchmark_returns.empty:
            return 0.0
        
        # 对齐索引
        aligned_returns, aligned_benchmark = returns.align(
            benchmark_returns, join='inner'
        )
        
        if aligned_returns.empty:
            return 0.0
        
        # 超额收益
        excess_returns = aligned_returns - aligned_benchmark
        
        if excess_returns.std() == 0:
            return 0.0
        
        # 信息比率 = 超额收益均值 / 超额收益标准差 * sqrt(252)
        return float(
            excess_returns.mean() / excess_returns.std() * np.sqrt(365)
        )
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """计算交易统计"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'total_fees': 0.0
            }
        
        df = pd.DataFrame(trades)
        
        pnl_col = 'pnl' if 'pnl' in df.columns else 'net_pnl'
        
        if pnl_col not in df.columns:
            return {
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'total_fees': df['fee'].sum() if 'fee' in df.columns else 0.0
            }
        
        wins = df[df[pnl_col] > 0]
        losses = df[df[pnl_col] < 0]
        
        gross_profit = wins[pnl_col].sum()
        gross_loss = abs(losses[pnl_col].sum())
        
        return {
            'total_trades': len(df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0.0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_win': wins[pnl_col].mean() if len(wins) > 0 else 0.0,
            'avg_loss': losses[pnl_col].mean() if len(losses) > 0 else 0.0,
            'largest_win': wins[pnl_col].max() if len(wins) > 0 else 0.0,
            'largest_loss': losses[pnl_col].min() if len(losses) > 0 else 0.0,
            'total_fees': df['fee'].sum() if 'fee' in df.columns else 0.0
        }


class PriceConverter:
    """
    价格转换器
    
    确保所有币价统一转换为 USDT 计价
    """
    
    # 稳定币列表（直接计价）
    STABLECOINS = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD'}
    
    def __init__(self):
        self._exchange_rates: Dict[str, float] = {}
    
    def update_rates(self, rates: Dict[str, float]):
        """更新汇率"""
        self._exchange_rates.update(rates)
    
    def to_usdt(
        self, 
        value: float, 
        currency: str,
        price_data: Dict[str, float] = None
    ) -> float:
        """
        将任意币种转换为 USDT 计价
        
        Args:
            value: 数值
            currency: 币种代码
            price_data: 当前价格数据 {币种: USDT价格}
            
        Returns:
            USDT 计价的数值
        """
        currency = currency.upper()
        
        # 如果已经是稳定币，直接返回
        if currency in self.STABLECOINS:
            return value
        
        # 使用价格数据转换
        if price_data and currency in price_data:
            return value * price_data[currency]
        
        # 使用缓存的汇率
        if currency in self._exchange_rates:
            return value * self._exchange_rates[currency]
        
        # 无法转换
        self.logger.warning(f"无法转换 {currency} 到 USDT，缺少汇率数据")
        return value
    
    def normalize_symbol(self, symbol: str) -> Tuple[str, str]:
        """
        标准化交易对
        
        Args:
            symbol: 交易对，如 'BTC/USDT'
            
        Returns:
            (基础货币, 计价货币)
        """
        parts = symbol.replace('-', '/').split('/')
        
        if len(parts) != 2:
            raise ValueError(f"无效的交易对格式: {symbol}")
        
        return parts[0].upper(), parts[1].upper()
    
    def is_usdt_pair(self, symbol: str) -> bool:
        """检查是否为 USDT 交易对"""
        _, quote = self.normalize_symbol(symbol)
        return quote in self.STABLECOINS


# 导出
__all__ = [
    'PrecisePerformanceCalculator',
    'PerformanceMetrics',
    'PriceConverter'
]
