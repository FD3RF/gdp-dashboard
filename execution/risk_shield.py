# execution/risk_shield.py
"""
第5层：反脆弱风控
================

功能：凯利公式仓位计算，熔断机制，风控检查
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class RiskLevel(Enum):
    """风险等级"""
    SAFE = 'safe'
    CAUTION = 'caution'
    DANGER = 'danger'
    CRITICAL = 'critical'


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    is_safe: bool
    level: RiskLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    side: str  # long, short
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    margin: float
    leverage: float


class RiskShield:
    """
    反脆弱风控系统
    
    功能：
    1. 单笔亏损熔断
    2. 日内亏损熔断
    3. 凯利公式仓位计算
    4. 动态风险调整
    5. 黑天鹅保护
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 默认配置
        default_config = {
            'single_loss_limit': 0.02,   # 单笔最大亏损 2%
            'daily_loss_limit': 0.05,    # 日内最大亏损 5%
            'max_position': 0.2,         # 最大仓位 20%
            'max_leverage': 5,           # 最大杠杆
            'api_timeout': 500,          # API 超时
            'black_swan_threshold': 0.1, # 黑天鹅阈值 10%
        }
        
        # 合并配置
        self.config = {**default_config, **(config or {})}
        
        # 日内统计
        self.daily_pnl = 0.0
        self.daily_start = datetime.now().replace(hour=0, minute=0, second=0)
        self.daily_trades = 0
        
        # 历史记录
        self.trade_history: List[Dict] = []
        self.max_history = 1000
        
        # 风险状态
        self.risk_level = RiskLevel.SAFE
        self.is_sleep_mode = False
        
        # 账户状态
        self.account_balance = 0.0
        self.positions: Dict[str, PositionInfo] = {}
        
        # 统计
        self.stats = {
            'total_checks': 0,
            'blocks': 0,
            'sleep_activations': 0,
        }
    
    def update_account(self, balance: float, positions: Dict[str, PositionInfo] = None):
        """更新账户状态"""
        self.account_balance = balance
        if positions:
            self.positions = positions
        
        # 检查是否需要重置日内统计
        now = datetime.now()
        if now.date() > self.daily_start.date():
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_start = now.replace(hour=0, minute=0, second=0)
            self.is_sleep_mode = False
            self.risk_level = RiskLevel.SAFE
    
    def check_position_safety(
        self, 
        action: int, 
        account_balance: float,
        current_loss: float = 0,
        position_value: float = 0
    ) -> RiskCheckResult:
        """
        检查仓位安全性
        
        Args:
            action: 动作 (0=多, 1=空, 2=平, 3=观望)
            account_balance: 账户余额
            current_loss: 当前亏损
            position_value: 持仓价值
            
        Returns:
            风控检查结果
        """
        self.stats['total_checks'] += 1
        details = {}
        
        # 1. 单笔亏损熔断
        if current_loss > account_balance * self.config['single_loss_limit']:
            self.stats['blocks'] += 1
            self.risk_level = RiskLevel.DANGER
            return RiskCheckResult(
                is_safe=False,
                level=RiskLevel.DANGER,
                message="🚫 单笔亏损超限",
                details={'current_loss': current_loss, 'limit': account_balance * self.config['single_loss_limit']}
            )
        
        # 2. 日内亏损熔断
        if self.daily_pnl < -account_balance * self.config['daily_loss_limit']:
            self.stats['blocks'] += 1
            self.is_sleep_mode = True
            self.risk_level = RiskLevel.CRITICAL
            return RiskCheckResult(
                is_safe=False,
                level=RiskLevel.CRITICAL,
                message="🚫 日内亏损超限 (休眠模式)",
                details={'daily_pnl': self.daily_pnl, 'limit': -account_balance * self.config['daily_loss_limit']}
            )
        
        # 3. 仓位超限检查
        if action in [0, 1]:  # 开仓
            total_position = sum(p.size * p.current_price for p in self.positions.values())
            total_position_pct = total_position / account_balance if account_balance > 0 else 0
            
            if total_position_pct > self.config['max_position']:
                self.stats['blocks'] += 1
                self.risk_level = RiskLevel.DANGER
                return RiskCheckResult(
                    is_safe=False,
                    level=RiskLevel.DANGER,
                    message="🚫 总仓位超限",
                    details={'position_pct': total_position_pct, 'limit': self.config['max_position']}
                )
        
        # 4. 黑天鹅检测
        if self._detect_black_swan():
            self.risk_level = RiskLevel.CRITICAL
            return RiskCheckResult(
                is_safe=False,
                level=RiskLevel.CRITICAL,
                message="🚨 黑天鹅事件检测",
                details={'recommendation': '立即平仓或减少仓位'}
            )
        
        # 更新风险等级
        if self.daily_pnl < -account_balance * self.config['daily_loss_limit'] * 0.5:
            self.risk_level = RiskLevel.CAUTION
        elif self.daily_pnl < -account_balance * self.config['daily_loss_limit'] * 0.3:
            self.risk_level = RiskLevel.CAUTION
        else:
            self.risk_level = RiskLevel.SAFE
        
        return RiskCheckResult(
            is_safe=True,
            level=self.risk_level,
            message="✅ 安全",
            details={'daily_pnl': self.daily_pnl}
        )
    
    def calculate_position_size(
        self, 
        confidence: float, 
        account_balance: float,
        win_rate: float = 0.5,
        win_loss_ratio: float = 1.5
    ) -> float:
        """
        凯利公式计算仓位大小
        
        f* = p - (1-p)/b
        其中：
        - p = 胜率
        - b = 盈亏比
        
        Args:
            confidence: 信心度
            account_balance: 账户余额
            win_rate: 胜率
            win_loss_ratio: 盈亏比
            
        Returns:
            仓位大小
        """
        # 凯利公式
        kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
        
        # 限制为正数
        kelly_fraction = max(kelly_fraction, 0)
        
        # 半凯利策略（更保守）
        safe_fraction = kelly_fraction * 0.5
        
        # 根据信心度调整
        position_pct = safe_fraction * confidence
        
        # 限制最大仓位
        position_pct = min(position_pct, self.config['max_position'])
        
        return account_balance * position_pct
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        side: str,
        atr: float = 0,
        volatility: float = 0.02
    ) -> float:
        """
        计算止损价
        
        Args:
            entry_price: 入场价
            side: 方向 (long/short)
            atr: ATR值
            volatility: 波动率
            
        Returns:
            止损价
        """
        # 基于 ATR 止损
        if atr > 0:
            stop_distance = atr * 2  # 2倍ATR
        else:
            stop_distance = entry_price * volatility  # 基于波动率
        
        if side == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        side: str,
        risk_reward_ratio: float = 2.0,
        stop_loss: float = 0
    ) -> float:
        """
        计算止盈价
        
        Args:
            entry_price: 入场价
            side: 方向
            risk_reward_ratio: 盈亏比
            stop_loss: 止损价
            
        Returns:
            止盈价
        """
        if stop_loss > 0:
            risk = abs(entry_price - stop_loss)
            profit_target = risk * risk_reward_ratio
        else:
            profit_target = entry_price * 0.02 * risk_reward_ratio  # 默认 2% 风险
        
        if side == 'long':
            return entry_price + profit_target
        else:
            return entry_price - profit_target
    
    def _detect_black_swan(self) -> bool:
        """检测黑天鹅事件"""
        # 检查最近的交易记录
        if len(self.trade_history) < 10:
            return False
        
        recent = self.trade_history[-10:]
        
        # 检查连续大亏损
        consecutive_losses = 0
        for trade in reversed(recent):
            if trade.get('pnl', 0) < 0:
                consecutive_losses += 1
            else:
                break
        
        # 连续 5 笔亏损
        if consecutive_losses >= 5:
            return True
        
        # 检查单日亏损
        if self.daily_pnl < -self.account_balance * self.config['black_swan_threshold']:
            return True
        
        return False
    
    def record_trade(self, trade: Dict[str, Any]):
        """记录交易"""
        self.trade_history.append({
            **trade,
            'timestamp': datetime.now().isoformat()
        })
        
        # 更新日内盈亏
        pnl = trade.get('pnl', 0)
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        # 限制历史记录
        if len(self.trade_history) > self.max_history:
            self.trade_history = self.trade_history[-self.max_history // 2:]
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        # 计算最大回撤
        max_drawdown = 0
        if len(self.trade_history) > 1:
            peak = 0
            running_pnl = 0
            for trade in self.trade_history:
                running_pnl += trade.get('pnl', 0)
                if running_pnl > peak:
                    peak = running_pnl
                drawdown = peak - running_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # 计算胜率
        wins = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        total = len(self.trade_history)
        win_rate = wins / total if total > 0 else 0
        
        return {
            'risk_level': self.risk_level.value,
            'is_sleep_mode': self.is_sleep_mode,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total,
            'account_balance': self.account_balance,
        }
    
    def force_sleep_mode(self):
        """强制进入休眠模式"""
        self.is_sleep_mode = True
        self.risk_level = RiskLevel.CRITICAL
        self.stats['sleep_activations'] += 1
    
    def wake_up(self):
        """唤醒"""
        self.is_sleep_mode = False
        self.risk_level = RiskLevel.SAFE


# 导出
__all__ = [
    'RiskShield',
    'RiskLevel',
    'RiskCheckResult',
    'PositionInfo',
]
