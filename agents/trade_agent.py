"""
交易执行代理
============

自动交易执行模块：
- 自动下单
- 仓位管理
- 止盈止损
- 订单追踪
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

from core.base import BaseModule
from core.constants import OrderSide, OrderType
from execution.okx_adapter import OKXAdapter


class TradeMode(Enum):
    """交易模式"""
    FULL_AUTO = 'full_auto'      # 全自动：AI信号直接下单
    SEMI_AUTO = 'semi_auto'      # 半自动：AI信号需确认
    SIMULATION = 'simulation'    # 模拟：不真实下单
    MANUAL = 'manual'            # 手动：仅手动操作


class OrderStatus(Enum):
    """订单状态"""
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partially_filled'
    CANCELLED = 'cancelled'
    FAILED = 'failed'


@dataclass
class TradeSignal:
    """交易信号"""
    symbol: str
    side: OrderSide
    signal_type: str  # entry, exit, stop_loss, take_profit, add_position
    strength: float  # 0-1 信号强度
    price: Optional[float] = None
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ''
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side.value if isinstance(self.side, OrderSide) else self.side,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'price': self.price,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    side: str  # long, short
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    margin: float
    leverage: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'margin': self.margin,
            'leverage': self.leverage,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'opened_at': self.opened_at.isoformat(),
        }


class TradeAgent(BaseModule):
    """
    交易执行代理
    
    功能：
    1. 信号处理 - 接收并处理交易信号
    2. 订单管理 - 创建、取消、查询订单
    3. 仓位管理 - 开仓、平仓、调整仓位
    4. 止盈止损 - 自动设置和触发
    5. 订单追踪 - 实时追踪订单状态
    """
    
    # 风控参数
    RISK_CONFIG = {
        'max_single_position_pct': 0.30,  # 单仓最大 30%
        'max_total_position_pct': 0.50,   # 总仓最大 50%
        'max_loss_pct': 0.15,             # 最大亏损 15%
        'default_stop_loss_pct': 0.05,    # 默认止损 5%
        'default_take_profit_pct': 0.10,  # 默认止盈 10%
        'max_orders_per_hour': 100,       # 每小时最大订单数
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('trade_agent', config)
        
        # 交易模式
        self._mode = TradeMode(
            self.config.get('mode', 'simulation')
        )
        
        # 交易所适配器
        self._exchange: Optional[OKXAdapter] = None
        
        # 持仓
        self._positions: Dict[str, Position] = {}
        
        # 订单
        self._pending_orders: Dict[str, Dict] = {}
        self._order_history: List[Dict] = []
        
        # 资金
        self._total_equity: float = 0
        self._available_balance: float = 0
        self._margin_used: float = 0
        
        # 统计
        self._stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'orders_today': 0,
            'orders_this_hour': 0,
            'last_hour_reset': datetime.now(),
        }
        
        # 信号回调
        self._on_signal_callbacks: List[Callable] = []
        self._on_order_callbacks: List[Callable] = []
        self._on_position_callbacks: List[Callable] = []
        
        # 待确认信号（半自动模式）
        self._pending_signals: List[TradeSignal] = []
    
    async def initialize(self) -> bool:
        """初始化"""
        self.logger.info(f"Initializing trade agent, mode: {self._mode.value}")
        
        # 初始化交易所连接
        if self._mode != TradeMode.SIMULATION:
            exchange_config = self.config.get('exchange', {})
            self._exchange = OKXAdapter(exchange_config)
            
            if not await self._exchange.initialize():
                self.logger.error("Failed to initialize exchange")
                return False
        
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """启动"""
        self._running = True
        self._start_time = datetime.now()
        
        if self._exchange:
            await self._exchange.start()
        
        self.logger.info(f"Trade agent started, mode: {self._mode.value}")
        return True
    
    async def stop(self) -> bool:
        """停止"""
        self._running = False
        
        # 取消所有未完成订单
        if self._exchange:
            try:
                await self._exchange.cancel_all_orders()
            except Exception as e:
                self.logger.error(f"Error cancelling orders: {e}")
            
            await self._exchange.stop()
        
        self.logger.info("Trade agent stopped")
        return True
    
    # ==================== 信号处理 ====================
    
    async def process_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """
        处理交易信号
        
        Args:
            signal: 交易信号
            
        Returns:
            处理结果
        """
        self.logger.info(
            f"Processing signal: {signal.symbol} {signal.side.value} "
            f"{signal.signal_type} (strength: {signal.strength:.2f})"
        )
        
        result = {
            'signal': signal.to_dict(),
            'action': 'none',
            'order': None,
            'error': None,
        }
        
        # 根据模式处理
        if self._mode == TradeMode.FULL_AUTO:
            # 全自动：直接执行
            result = await self._execute_signal(signal)
            
        elif self._mode == TradeMode.SEMI_AUTO:
            # 半自动：加入待确认队列
            self._pending_signals.append(signal)
            result['action'] = 'pending_confirmation'
            self._notify_signal(signal)
            
        elif self._mode == TradeMode.SIMULATION:
            # 模拟：记录但不执行
            result['action'] = 'simulated'
            self._log_trade(signal, simulated=True)
            
        elif self._mode == TradeMode.MANUAL:
            # 手动：仅记录
            result['action'] = 'logged'
        
        return result
    
    async def confirm_signal(self, signal_id: str) -> Dict[str, Any]:
        """确认信号（半自动模式）"""
        # 查找信号
        for i, signal in enumerate(self._pending_signals):
            if id(signal) == int(signal_id):
                self._pending_signals.pop(i)
                return await self._execute_signal(signal)
        
        return {'error': 'Signal not found'}
    
    async def reject_signal(self, signal_id: str) -> Dict[str, Any]:
        """拒绝信号"""
        for i, signal in enumerate(self._pending_signals):
            if id(signal) == int(signal_id):
                self._pending_signals.pop(i)
                return {'action': 'rejected'}
        
        return {'error': 'Signal not found'}
    
    async def _execute_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        """执行信号"""
        result = {
            'signal': signal.to_dict(),
            'action': 'none',
            'order': None,
            'error': None,
        }
        
        try:
            # 检查风控
            risk_check = await self._check_risk(signal)
            if not risk_check['allowed']:
                result['error'] = risk_check['reason']
                result['action'] = 'rejected_by_risk'
                return result
            
            # 执行操作
            if signal.signal_type == 'entry':
                order = await self._open_position(signal)
                result['action'] = 'opened'
                result['order'] = order
                
            elif signal.signal_type == 'exit':
                order = await self._close_position(signal)
                result['action'] = 'closed'
                result['order'] = order
                
            elif signal.signal_type == 'stop_loss':
                order = await self._close_position(signal)
                result['action'] = 'stop_loss_triggered'
                result['order'] = order
                
            elif signal.signal_type == 'take_profit':
                order = await self._close_position(signal)
                result['action'] = 'take_profit_triggered'
                result['order'] = order
                
            elif signal.signal_type == 'add_position':
                order = await self._open_position(signal)
                result['action'] = 'added'
                result['order'] = order
            
            # 记录交易
            self._log_trade(signal, order=result.get('order'))
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            result['error'] = str(e)
            result['action'] = 'failed'
        
        return result
    
    # ==================== 仓位操作 ====================
    
    async def _open_position(self, signal: TradeSignal) -> Optional[Dict]:
        """开仓"""
        if not self._exchange:
            return self._simulate_order(signal)
        
        # 计算仓位大小
        if not signal.quantity:
            signal.quantity = self._calculate_position_size(signal)
        
        # 创建订单
        order = await self._exchange.create_order(
            symbol=signal.symbol,
            order_type='market',  # 市价单
            side=signal.side.value,
            amount=signal.quantity,
            params={
                'stopLoss': signal.stop_loss,
                'takeProfit': signal.take_profit,
            }
        )
        
        if order:
            self._stats['total_trades'] += 1
            self._stats['orders_this_hour'] += 1
            
            # 添加到持仓追踪
            position = Position(
                symbol=signal.symbol,
                side='long' if signal.side == OrderSide.BUY else 'short',
                quantity=signal.quantity,
                entry_price=order.get('price', signal.price or 0),
                current_price=order.get('price', signal.price or 0),
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
                margin=self._calculate_margin(signal.quantity, signal.price),
                leverage=self.config.get('leverage', 1),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
            
            self._positions[signal.symbol] = position
            self._notify_position(position)
        
        return order
    
    async def _close_position(self, signal: TradeSignal) -> Optional[Dict]:
        """平仓"""
        position = self._positions.get(signal.symbol)
        if not position:
            self.logger.warning(f"No position to close: {signal.symbol}")
            return None
        
        if not self._exchange:
            return self._simulate_order(signal)
        
        # 平仓方向相反
        close_side = OrderSide.SELL if position.side == 'long' else OrderSide.BUY
        
        order = await self._exchange.create_order(
            symbol=signal.symbol,
            order_type='market',
            side=close_side.value,
            amount=position.quantity,
        )
        
        if order:
            # 计算盈亏
            pnl = self._calculate_pnl(position, order.get('price', 0))
            
            # 更新统计
            self._stats['total_pnl'] += pnl
            if pnl > 0:
                self._stats['winning_trades'] += 1
            else:
                self._stats['losing_trades'] += 1
            
            # 移除持仓
            del self._positions[signal.symbol]
            
            # 通知
            self._notify_position(None, closed=True, pnl=pnl)
        
        return order
    
    def _simulate_order(self, signal: TradeSignal) -> Dict:
        """模拟订单"""
        return {
            'order_id': f'sim_{int(datetime.now().timestamp())}',
            'symbol': signal.symbol,
            'side': signal.side.value,
            'type': 'market',
            'quantity': signal.quantity or 0,
            'price': signal.price or 0,
            'status': 'filled',
            'simulated': True,
            'timestamp': datetime.now().isoformat(),
        }
    
    # ==================== 风控检查 ====================
    
    async def _check_risk(self, signal: TradeSignal) -> Dict[str, Any]:
        """风控检查"""
        reasons = []
        
        # 检查订单频率
        if self._stats['orders_this_hour'] >= self.RISK_CONFIG['max_orders_per_hour']:
            reasons.append(f"订单频率超限: {self._stats['orders_this_hour']}/h")
        
        # 检查单仓大小
        if signal.quantity:
            position_value = signal.quantity * (signal.price or 1)
            single_pct = position_value / self._total_equity if self._total_equity > 0 else 1
            
            if single_pct > self.RISK_CONFIG['max_single_position_pct']:
                reasons.append(f"单仓过大: {single_pct*100:.1f}%")
        
        # 检查总仓位
        total_position_pct = self._get_total_position_pct()
        if total_position_pct > self.RISK_CONFIG['max_total_position_pct']:
            reasons.append(f"总仓过大: {total_position_pct*100:.1f}%")
        
        # 检查累计亏损
        if self._stats['total_pnl'] < 0:
            loss_pct = abs(self._stats['total_pnl']) / self._total_equity if self._total_equity > 0 else 0
            if loss_pct > self.RISK_CONFIG['max_loss_pct']:
                reasons.append(f"累计亏损过大: {loss_pct*100:.1f}%")
        
        return {
            'allowed': len(reasons) == 0,
            'reasons': reasons,
            'reason': '; '.join(reasons) if reasons else None,
        }
    
    def _get_total_position_pct(self) -> float:
        """获取总仓位比例"""
        total_position_value = sum(
            p.quantity * p.current_price for p in self._positions.values()
        )
        return total_position_value / self._total_equity if self._total_equity > 0 else 0
    
    # ==================== 工具方法 ====================
    
    def _calculate_position_size(self, signal: TradeSignal) -> float:
        """计算仓位大小"""
        # 基于信号强度和可用资金
        risk_per_trade = self.config.get('risk_per_trade', 0.02)  # 每笔交易风险 2%
        risk_amount = self._total_equity * risk_per_trade * signal.strength
        
        # 计算数量
        price = signal.price or 1
        if signal.stop_loss:
            stop_distance = abs(price - signal.stop_loss)
            quantity = risk_amount / stop_distance if stop_distance > 0 else 0
        else:
            # 默认使用 2% 止损距离
            quantity = risk_amount / (price * 0.02)
        
        return round(quantity, 6)
    
    def _calculate_margin(self, quantity: float, price: float) -> float:
        """计算保证金"""
        leverage = self.config.get('leverage', 1)
        return (quantity * price) / leverage
    
    def _calculate_pnl(self, position: Position, exit_price: float) -> float:
        """计算盈亏"""
        if position.side == 'long':
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        return pnl
    
    def _log_trade(self, signal: TradeSignal, order: Optional[Dict] = None, simulated: bool = False):
        """记录交易"""
        trade = {
            'signal': signal.to_dict(),
            'order': order,
            'simulated': simulated,
            'timestamp': datetime.now().isoformat(),
        }
        self._order_history.append(trade)
        
        # 保持历史记录在合理范围
        if len(self._order_history) > 1000:
            self._order_history = self._order_history[-500:]
    
    # ==================== 状态更新 ====================
    
    async def update_positions(self, prices: Dict[str, float]):
        """更新持仓价格"""
        for symbol, position in self._positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                
                # 计算未实现盈亏
                if position.side == 'long':
                    position.unrealized_pnl = (
                        position.current_price - position.entry_price
                    ) * position.quantity
                else:
                    position.unrealized_pnl = (
                        position.entry_price - position.current_price
                    ) * position.quantity
                
                position.unrealized_pnl_pct = (
                    position.unrealized_pnl / (position.entry_price * position.quantity)
                ) if position.quantity > 0 else 0
                
                # 检查止盈止损
                await self._check_stop_orders(position)
    
    async def _check_stop_orders(self, position: Position):
        """检查止盈止损"""
        # 止损检查
        if position.stop_loss:
            if position.side == 'long' and position.current_price <= position.stop_loss:
                await self.process_signal(TradeSignal(
                    symbol=position.symbol,
                    side=OrderSide.SELL,
                    signal_type='stop_loss',
                    strength=1.0,
                    reason=f'止损触发: {position.stop_loss}',
                ))
            elif position.side == 'short' and position.current_price >= position.stop_loss:
                await self.process_signal(TradeSignal(
                    symbol=position.symbol,
                    side=OrderSide.BUY,
                    signal_type='stop_loss',
                    strength=1.0,
                    reason=f'止损触发: {position.stop_loss}',
                ))
        
        # 止盈检查
        if position.take_profit:
            if position.side == 'long' and position.current_price >= position.take_profit:
                await self.process_signal(TradeSignal(
                    symbol=position.symbol,
                    side=OrderSide.SELL,
                    signal_type='take_profit',
                    strength=1.0,
                    reason=f'止盈触发: {position.take_profit}',
                ))
            elif position.side == 'short' and position.current_price <= position.take_profit:
                await self.process_signal(TradeSignal(
                    symbol=position.symbol,
                    side=OrderSide.BUY,
                    signal_type='take_profit',
                    strength=1.0,
                    reason=f'止盈触发: {position.take_profit}',
                ))
    
    async def update_balance(self, balance: Dict[str, Any]):
        """更新资金"""
        self._total_equity = balance.get('total', {}).get('USDT', 0)
        self._available_balance = balance.get('free', {}).get('USDT', 0)
        self._margin_used = balance.get('used', {}).get('USDT', 0)
    
    # ==================== 心跳和健康检查 ====================
    
    async def heartbeat(self):
        """心跳"""
        # 更新每小时计数
        now = datetime.now()
        if (now - self._stats['last_hour_reset']).total_seconds() > 3600:
            self._stats['orders_this_hour'] = 0
            self._stats['last_hour_reset'] = now
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'healthy': self._running,
            'mode': self._mode.value,
            'positions': len(self._positions),
            'pending_signals': len(self._pending_signals),
            'total_equity': self._total_equity,
            'total_pnl': self._stats['total_pnl'],
            'win_rate': (
                self._stats['winning_trades'] / self._stats['total_trades'] * 100
                if self._stats['total_trades'] > 0 else 0
            ),
        }
    
    # ==================== 回调注册 ====================
    
    def on_signal(self, callback: Callable):
        """注册信号回调"""
        self._on_signal_callbacks.append(callback)
    
    def on_order(self, callback: Callable):
        """注册订单回调"""
        self._on_order_callbacks.append(callback)
    
    def on_position(self, callback: Callable):
        """注册持仓回调"""
        self._on_position_callbacks.append(callback)
    
    def _notify_signal(self, signal: TradeSignal):
        """通知信号"""
        for callback in self._on_signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(signal))
                else:
                    callback(signal)
            except Exception as e:
                self.logger.error(f"Signal callback error: {e}")
    
    def _notify_order(self, order: Dict):
        """通知订单"""
        for callback in self._on_order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(order))
                else:
                    callback(order)
            except Exception as e:
                self.logger.error(f"Order callback error: {e}")
    
    def _notify_position(self, position: Optional[Position], closed: bool = False, pnl: float = 0):
        """通知持仓"""
        for callback in self._on_position_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(position, closed, pnl))
                else:
                    callback(position, closed, pnl)
            except Exception as e:
                self.logger.error(f"Position callback error: {e}")
    
    # ==================== 状态查询 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'mode': self._mode.value,
            'running': self._running,
            'total_equity': self._total_equity,
            'available_balance': self._available_balance,
            'margin_used': self._margin_used,
            'positions': {
                symbol: pos.to_dict() for symbol, pos in self._positions.items()
            },
            'pending_signals': [s.to_dict() for s in self._pending_signals],
            'stats': {
                **self._stats,
                'win_rate': (
                    self._stats['winning_trades'] / self._stats['total_trades'] * 100
                    if self._stats['total_trades'] > 0 else 0
                ),
            },
        }
    
    def get_positions(self) -> Dict[str, Position]:
        """获取持仓"""
        return self._positions.copy()
    
    def get_pending_signals(self) -> List[TradeSignal]:
        """获取待确认信号"""
        return self._pending_signals.copy()
    
    def get_order_history(self, limit: int = 100) -> List[Dict]:
        """获取订单历史"""
        return self._order_history[-limit:]
    
    def set_mode(self, mode: TradeMode):
        """设置交易模式"""
        self._mode = mode
        self.logger.info(f"Trade mode changed to: {mode.value}")
    
    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return self._stats['total_pnl']


# 导出
__all__ = [
    'TradeAgent',
    'TradeMode',
    'TradeSignal',
    'Position',
    'OrderStatus'
]
