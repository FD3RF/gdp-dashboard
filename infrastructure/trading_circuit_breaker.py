"""
交易熔断器 (Trading Circuit Breaker)
====================================
自动风控熔断，保护资金安全

触发条件：
1. 连续亏损 N 笔
2. 回撤超过阈值
3. 单日亏损超限
4. 黑天鹅事件检测
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CircuitBreakerStatus(Enum):
    """熔断器状态"""
    ACTIVE = "active"           # 正常运行
    WARNING = "warning"         # 警告（接近触发）
    TRIGGERED = "triggered"     # 已触发（暂停交易）
    COOLDOWN = "cooldown"       # 冷却期


class TriggerReason(Enum):
    """触发原因"""
    CONSECUTIVE_LOSSES = "consecutive_losses"   # 连续亏损
    DRAWDOWN_EXCEEDED = "drawdown_exceeded"     # 回撤超限
    DAILY_LOSS_LIMIT = "daily_loss_limit"       # 单日亏损超限
    BLACK_SWAN = "black_swan"                   # 黑天鹅事件
    MANUAL = "manual"                           # 手动触发


@dataclass
class TradingStats:
    """交易统计"""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    current_streak: int = 0  # 正数=连胜，负数=连亏
    
    total_pnl: float = 0
    max_profit: float = 0
    max_loss: float = 0
    
    peak_balance: float = 10000  # 峰值余额
    current_balance: float = 10000
    
    daily_pnl: float = 0
    daily_trades: int = 0
    
    last_trade_time: Optional[datetime] = None
    last_trade_result: Optional[str] = None  # "win" / "loss"


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    max_consecutive_losses: int = 3        # 最大连续亏损次数
    max_drawdown_pct: float = 0.05         # 最大回撤百分比 (5%)
    max_daily_loss_pct: float = 0.03       # 最大单日亏损 (3%)
    max_daily_trades: int = 10             # 单日最大交易次数
    
    cooldown_hours: int = 24               # 冷却期（小时）
    warning_threshold: float = 0.7         # 警告阈值（70%时发出警告）


@dataclass
class CircuitBreakerState:
    """熔断器状态"""
    status: CircuitBreakerStatus = CircuitBreakerStatus.ACTIVE
    trigger_reason: Optional[TriggerReason] = None
    trigger_time: Optional[datetime] = None
    trigger_value: float = 0
    cooldown_end: Optional[datetime] = None
    warning_messages: List[str] = field(default_factory=list)


class TradingCircuitBreaker:
    """
    交易熔断器
    
    自动风控系统，在危险情况下暂停交易
    
    熔断规则：
    ┌──────────────────────────────────────────────────────────────┐
    │ 触发条件                    │ 状态      │ 冷却期   │ 操作    │
    ├──────────────────────────────────────────────────────────────┤
    │ 连续亏损 3 笔              │ TRIGGERED │ 24小时   │ 暂停交易│
    │ 回撤 > 5%                  │ TRIGGERED │ 48小时   │ 暂停交易│
    │ 单日亏损 > 3%              │ TRIGGERED │ 24小时   │ 暂停交易│
    │ 黑天鹅事件                 │ TRIGGERED │ 72小时   │ 暂停交易│
    │ 接近阈值 (70%)             │ WARNING   │ -        │ 发出警告│
    └──────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.stats = TradingStats()
        self.state = CircuitBreakerState()
        self._trade_history: List[Dict] = []
    
    def record_trade_result(
        self,
        pnl: float,
        is_win: bool,
        balance: float = None,
    ) -> Dict[str, Any]:
        """
        记录交易结果
        
        Args:
            pnl: 盈亏金额
            is_win: 是否盈利
            balance: 当前余额
        
        Returns:
            状态更新结果
        """
        now = datetime.now()
        
        # 更新统计
        self.stats.total_trades += 1
        self.stats.total_pnl += pnl
        self.stats.last_trade_time = now
        self.stats.last_trade_result = "win" if is_win else "loss"
        
        if balance is not None:
            self.stats.current_balance = balance
            if balance > self.stats.peak_balance:
                self.stats.peak_balance = balance
        
        if is_win:
            self.stats.wins += 1
            self.stats.consecutive_wins += 1
            self.stats.consecutive_losses = 0
            self.stats.current_streak = max(1, self.stats.current_streak + 1 if self.stats.current_streak > 0 else 1)
            
            if pnl > self.stats.max_profit:
                self.stats.max_profit = pnl
        else:
            self.stats.losses += 1
            self.stats.consecutive_losses += 1
            self.stats.consecutive_wins = 0
            self.stats.current_streak = min(-1, self.stats.current_streak - 1 if self.stats.current_streak < 0 else -1)
            
            if pnl < self.stats.max_loss:
                self.stats.max_loss = pnl
        
        # 检查是否需要重置日内统计
        if self.stats.last_trade_time and (now - self.stats.last_trade_time).days > 0:
            self.stats.daily_pnl = 0
            self.stats.daily_trades = 0
        
        self.stats.daily_pnl += pnl
        self.stats.daily_trades += 1
        
        # 记录历史
        self._trade_history.append({
            "time": now.isoformat(),
            "pnl": pnl,
            "is_win": is_win,
            "balance": balance,
        })
        if len(self._trade_history) > 100:
            self._trade_history = self._trade_history[-100:]
        
        # 检查熔断条件
        return self._check_conditions()
    
    def _check_conditions(self) -> Dict[str, Any]:
        """检查是否触发熔断"""
        warnings = []
        triggered = False
        trigger_reason = None
        
        # 检查连续亏损
        if self.stats.consecutive_losses >= self.config.max_consecutive_losses:
            triggered = True
            trigger_reason = TriggerReason.CONSECUTIVE_LOSSES
            logger.warning(f"🚨 熔断触发: 连续亏损 {self.stats.consecutive_losses} 笔")
        
        elif self.stats.consecutive_losses >= self.config.max_consecutive_losses * self.config.warning_threshold:
            warnings.append(f"连续亏损 {self.stats.consecutive_losses} 笔，接近熔断阈值 {self.config.max_consecutive_losses}")
        
        # 检查回撤
        if self.stats.peak_balance > 0:
            drawdown = (self.stats.peak_balance - self.stats.current_balance) / self.stats.peak_balance
            
            if drawdown >= self.config.max_drawdown_pct:
                triggered = True
                trigger_reason = TriggerReason.DRAWDOWN_EXCEEDED
                logger.warning(f"🚨 熔断触发: 回撤 {drawdown*100:.1f}% 超过阈值 {self.config.max_drawdown_pct*100}%")
            
            elif drawdown >= self.config.max_drawdown_pct * self.config.warning_threshold:
                warnings.append(f"回撤 {drawdown*100:.1f}%，接近阈值 {self.config.max_drawdown_pct*100}%")
        
        # 检查单日亏损
        if self.stats.peak_balance > 0:
            daily_loss_pct = abs(self.stats.daily_pnl) / self.stats.peak_balance if self.stats.daily_pnl < 0 else 0
            
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                triggered = True
                trigger_reason = TriggerReason.DAILY_LOSS_LIMIT
                logger.warning(f"🚨 熔断触发: 单日亏损 {daily_loss_pct*100:.1f}%")
            
            elif daily_loss_pct >= self.config.max_daily_loss_pct * self.config.warning_threshold:
                warnings.append(f"单日亏损 {daily_loss_pct*100:.1f}%，接近阈值 {self.config.max_daily_loss_pct*100}%")
        
        # 更新状态
        if triggered:
            cooldown_hours = self.config.cooldown_hours
            if trigger_reason == TriggerReason.DRAWDOWN_EXCEEDED:
                cooldown_hours *= 2  # 回撤熔断冷却期加倍
            
            self.state.status = CircuitBreakerStatus.TRIGGERED
            self.state.trigger_reason = trigger_reason
            self.state.trigger_time = datetime.now()
            self.state.trigger_value = self.stats.consecutive_losses
            self.state.cooldown_end = datetime.now() + timedelta(hours=cooldown_hours)
            self.state.warning_messages = []
        
        elif warnings:
            self.state.status = CircuitBreakerStatus.WARNING
            self.state.warning_messages = warnings
        else:
            self.state.status = CircuitBreakerStatus.ACTIVE
            self.state.warning_messages = []
        
        return {
            "triggered": triggered,
            "trigger_reason": trigger_reason.value if trigger_reason else None,
            "status": self.state.status.value,
            "warnings": warnings,
        }
    
    def trigger_black_swan(self, description: str = "") -> Dict:
        """手动触发黑天鹅熔断"""
        self.state.status = CircuitBreakerStatus.TRIGGERED
        self.state.trigger_reason = TriggerReason.BLACK_SWAN
        self.state.trigger_time = datetime.now()
        self.state.cooldown_end = datetime.now() + timedelta(hours=self.config.cooldown_hours * 3)
        
        logger.critical(f"🚨 黑天鹅熔断触发: {description}")
        
        return {
            "triggered": True,
            "trigger_reason": "black_swan",
            "description": description,
            "cooldown_end": self.state.cooldown_end.isoformat(),
        }
    
    def manual_pause(self, reason: str = "") -> Dict:
        """手动暂停交易"""
        self.state.status = CircuitBreakerStatus.TRIGGERED
        self.state.trigger_reason = TriggerReason.MANUAL
        self.state.trigger_time = datetime.now()
        self.state.cooldown_end = None  # 手动暂停需手动恢复
        
        return {
            "triggered": True,
            "trigger_reason": "manual",
            "reason": reason,
        }
    
    def manual_resume(self) -> Dict:
        """手动恢复交易"""
        self.state.status = CircuitBreakerStatus.ACTIVE
        self.state.trigger_reason = None
        self.state.trigger_time = None
        self.state.cooldown_end = None
        
        # 重置统计
        self.stats.consecutive_losses = 0
        self.stats.consecutive_wins = 0
        self.stats.current_streak = 0
        
        return {"resumed": True, "status": "active"}
    
    def is_trading_allowed(self) -> bool:
        """检查是否允许交易"""
        if self.state.status == CircuitBreakerStatus.TRIGGERED:
            # 检查冷却期是否结束
            if self.state.cooldown_end and datetime.now() >= self.state.cooldown_end:
                self.state.status = CircuitBreakerStatus.COOLDOWN
                return False
            return False
        
        if self.state.status == CircuitBreakerStatus.COOLDOWN:
            # 冷却期后自动恢复
            self.state.status = CircuitBreakerStatus.ACTIVE
            self.state.trigger_reason = None
            return True
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "status": self.state.status.value,
            "trading_allowed": self.is_trading_allowed(),
            "trigger_reason": self.state.trigger_reason.value if self.state.trigger_reason else None,
            "trigger_time": self.state.trigger_time.isoformat() if self.state.trigger_time else None,
            "cooldown_end": self.state.cooldown_end.isoformat() if self.state.cooldown_end else None,
            "warnings": self.state.warning_messages,
            "stats": {
                "total_trades": self.stats.total_trades,
                "wins": self.stats.wins,
                "losses": self.stats.losses,
                "win_rate": self.stats.wins / self.stats.total_trades if self.stats.total_trades > 0 else 0,
                "consecutive_losses": self.stats.consecutive_losses,
                "current_streak": self.stats.current_streak,
                "total_pnl": round(self.stats.total_pnl, 2),
                "peak_balance": round(self.stats.peak_balance, 2),
                "current_balance": round(self.stats.current_balance, 2),
                "drawdown_pct": round((self.stats.peak_balance - self.stats.current_balance) / self.stats.peak_balance * 100, 2) if self.stats.peak_balance > 0 else 0,
                "daily_pnl": round(self.stats.daily_pnl, 2),
            },
            "config": {
                "max_consecutive_losses": self.config.max_consecutive_losses,
                "max_drawdown_pct": self.config.max_drawdown_pct * 100,
                "max_daily_loss_pct": self.config.max_daily_loss_pct * 100,
                "cooldown_hours": self.config.cooldown_hours,
            }
        }


# 全局实例
_circuit_breaker: Optional[TradingCircuitBreaker] = None


def get_circuit_breaker(config: CircuitBreakerConfig = None) -> TradingCircuitBreaker:
    """获取熔断器"""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = TradingCircuitBreaker(config)
    return _circuit_breaker


def is_trading_allowed() -> bool:
    """检查是否允许交易（便捷函数）"""
    return get_circuit_breaker().is_trading_allowed()


def record_trade(pnl: float, is_win: bool, balance: float = None) -> Dict:
    """记录交易结果（便捷函数）"""
    return get_circuit_breaker().record_trade_result(pnl, is_win, balance)
