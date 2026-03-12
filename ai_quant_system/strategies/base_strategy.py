"""
Base Strategy class for all trading strategies.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from core.base import BaseModule
from core.constants import OrderSide, OrderType


class StrategyStatus(Enum):
    """Strategy status enumeration."""
    CREATED = 'created'
    INITIALIZING = 'initializing'
    READY = 'ready'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    ERROR = 'error'


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    version: str = '1.0.0'
    timeframe: str = '1h'
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT'])
    initial_capital: float = 100000.0
    
    # Risk parameters
    max_position_size: float = 0.02  # 2% of portfolio
    max_drawdown: float = 0.15  # 15% max drawdown
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    
    # Execution parameters
    order_type: str = 'market'
    slippage_tolerance: float = 0.001
    
    # Custom parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'timeframe': self.timeframe,
            'symbols': self.symbols,
            'initial_capital': self.initial_capital,
            'max_position_size': self.max_position_size,
            'max_drawdown': self.max_drawdown,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'order_type': self.order_type,
            'parameters': self.parameters
        }


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    side: OrderSide
    signal_type: str  # entry, exit, stop_loss, take_profit
    strength: float  # 0-1
    price: Optional[float] = None
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'signal_type': self.signal_type,
            'strength': self.strength,
            'price': self.price,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class BaseStrategy(BaseModule):
    """
    Base class for all trading strategies.
    Provides common functionality for signal generation and position management.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        data_provider: Optional[Callable] = None,
        order_executor: Optional[Callable] = None
    ):
        super().__init__(config.name, config.to_dict())
        self._config = config
        self._status = StrategyStatus.CREATED
        self._data_provider = data_provider
        self._order_executor = order_executor
        
        # State
        self._positions: Dict[str, Dict] = {}
        self._signals: List[Signal] = []
        self._trades: List[Dict] = []
        self._equity_curve: List[float] = []
        
        # Metrics
        self._metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Callbacks
        self._on_signal_callbacks: List[Callable] = []
        self._on_trade_callbacks: List[Callable] = []
    
    @property
    def status(self) -> StrategyStatus:
        return self._status
    
    @property
    def config(self) -> StrategyConfig:
        return self._config
    
    @abstractmethod
    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from market data.
        Must be implemented by subclasses.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            List of signals
        """
        pass
    
    async def initialize(self) -> bool:
        """Initialize the strategy."""
        self.logger.info(f"Initializing strategy: {self._config.name}")
        self._status = StrategyStatus.READY
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the strategy."""
        if self._status != StrategyStatus.READY:
            await self.initialize()
        
        self._status = StrategyStatus.RUNNING
        self._running = True
        self._start_time = datetime.now()
        self.logger.info(f"Strategy started: {self._config.name}")
        return True
    
    async def stop(self) -> bool:
        """Stop the strategy."""
        self._status = StrategyStatus.STOPPED
        self._running = False
        
        # Close all positions
        await self.close_all_positions()
        
        self.logger.info(f"Strategy stopped: {self._config.name}")
        return True
    
    async def pause(self) -> bool:
        """Pause the strategy."""
        self._status = StrategyStatus.PAUSED
        self._running = False
        return True
    
    async def resume(self) -> bool:
        """Resume the strategy."""
        if self._status == StrategyStatus.PAUSED:
            self._status = StrategyStatus.RUNNING
            self._running = True
            return True
        return False
    
    async def on_data(self, data: pd.DataFrame) -> List[Signal]:
        """
        Process new market data and generate signals.
        
        Args:
            data: New market data
        
        Returns:
            List of generated signals
        """
        if not self._running:
            return []
        
        try:
            signals = await self.generate_signals(data)
            
            # Validate and filter signals
            valid_signals = []
            for signal in signals:
                if self._validate_signal(signal):
                    valid_signals.append(signal)
                    self._signals.append(signal)
            
            # Notify callbacks
            for signal in valid_signals:
                for callback in self._on_signal_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(signal)
                        else:
                            callback(signal)
                    except Exception as e:
                        self.logger.error(f"Signal callback error: {e}")
            
            return valid_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            self._status = StrategyStatus.ERROR
            return []
    
    def _validate_signal(self, signal: Signal) -> bool:
        """Validate a trading signal."""
        # Check symbol
        if signal.symbol not in self._config.symbols:
            self.logger.warning(f"Invalid symbol in signal: {signal.symbol}")
            return False
        
        # Check strength
        if not 0 <= signal.strength <= 1:
            self.logger.warning(f"Invalid signal strength: {signal.strength}")
            return False
        
        return True
    
    async def execute_signal(self, signal: Signal) -> Optional[Dict]:
        """Execute a trading signal."""
        if self._order_executor:
            try:
                order = await self._order_executor(signal)
                self._trades.append(order)
                self._metrics['total_trades'] += 1
                return order
            except Exception as e:
                self.logger.error(f"Error executing signal: {e}")
        return None
    
    async def close_all_positions(self) -> None:
        """Close all open positions."""
        for symbol, position in self._positions.items():
            if position.get('quantity', 0) != 0:
                # Generate close signal
                signal = Signal(
                    symbol=symbol,
                    side=OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY,
                    signal_type='exit',
                    strength=1.0
                )
                await self.execute_signal(signal)
        
        self._positions.clear()
    
    def update_position(self, symbol: str, quantity: float, price: float) -> None:
        """Update position after a trade."""
        if symbol not in self._positions:
            self._positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'unrealized_pnl': 0
            }
        
        current = self._positions[symbol]
        new_quantity = current['quantity'] + quantity
        
        if new_quantity == 0:
            # Position closed
            realized_pnl = (price - current['avg_price']) * current['quantity']
            self._metrics['total_pnl'] += realized_pnl
            if realized_pnl > 0:
                self._metrics['winning_trades'] += 1
            else:
                self._metrics['losing_trades'] += 1
            del self._positions[symbol]
        else:
            # Update position
            if quantity * current['quantity'] > 0:
                # Adding to position
                total_cost = current['avg_price'] * abs(current['quantity']) + price * abs(quantity)
                current['avg_price'] = total_cost / (abs(new_quantity))
            current['quantity'] = new_quantity
            current['unrealized_pnl'] = (price - current['avg_price']) * current['quantity']
    
    def calculate_position_size(
        self,
        price: float,
        portfolio_value: float
    ) -> float:
        """Calculate position size based on risk parameters."""
        max_position_value = portfolio_value * self._config.max_position_size
        return max_position_value / price
    
    def calculate_stop_loss(self, entry_price: float, side: OrderSide) -> float:
        """Calculate stop loss price."""
        if side == OrderSide.BUY:
            return entry_price * (1 - self._config.stop_loss_pct)
        else:
            return entry_price * (1 + self._config.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, side: OrderSide) -> float:
        """Calculate take profit price."""
        if side == OrderSide.BUY:
            return entry_price * (1 + self._config.take_profit_pct)
        else:
            return entry_price * (1 - self._config.take_profit_pct)
    
    def add_signal_callback(self, callback: Callable) -> None:
        """Add a callback for signal events."""
        self._on_signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable) -> None:
        """Add a callback for trade events."""
        self._on_trade_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy metrics."""
        win_rate = (
            self._metrics['winning_trades'] / self._metrics['total_trades']
            if self._metrics['total_trades'] > 0 else 0
        )
        
        return {
            **self._metrics,
            'win_rate': win_rate,
            'status': self._status.value
        }
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions."""
        return self._positions.copy()
    
    def get_recent_signals(self, limit: int = 20) -> List[Signal]:
        """Get recent signals."""
        return self._signals[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status."""
        return {
            'name': self._config.name,
            'status': self._status.value,
            'running': self._running,
            'positions': len(self._positions),
            'metrics': self.get_metrics()
        }


import asyncio
