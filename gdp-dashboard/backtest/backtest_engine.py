"""
Backtest Engine for strategy simulation.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from core.base import BaseModule
from core.constants import OrderSide, OrderType
from backtest.historical_loader import HistoricalDataLoader
from backtest.slippage_model import SlippageModel
from backtest.fee_model import FeeModel
from backtest.performance_analyzer import PerformanceAnalyzer


class BacktestEngine(BaseModule):
    """
    Backtest engine for simulating trading strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('backtest_engine', config)
        
        self._data_loader: Optional[HistoricalDataLoader] = None
        self._slippage_model = SlippageModel(
            self.config.get('slippage_model', 'percentage'),
            self.config.get('slippage_params', {})
        )
        self._fee_model = FeeModel(
            self.config.get('fee_model', 'percentage'),
            self.config.get('fee_params', {})
        )
        self._performance_analyzer = PerformanceAnalyzer()
        
        # Backtest state
        self._initial_capital = self.config.get('initial_capital', 100000)
        self._portfolio_value = 0
        self._cash = 0
        self._positions: Dict[str, Dict] = {}
        self._trades: List[Dict] = []
        self._equity_curve: List[Dict] = []
    
    def set_data_loader(self, loader: HistoricalDataLoader) -> None:
        """Set the data loader."""
        self._data_loader = loader
    
    async def initialize(self) -> bool:
        """Initialize the backtest engine."""
        self.logger.info("Initializing backtest engine...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the backtest engine."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the backtest engine."""
        self._running = False
        return True
    
    async def run(
        self,
        strategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: Optional[float] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Strategy instance
            start_date: Start date
            end_date: End date
            initial_capital: Initial capital
            symbols: List of symbols
        
        Returns:
            Backtest results
        """
        self._initial_capital = initial_capital or self._initial_capital
        self._cash = self._initial_capital
        self._portfolio_value = self._initial_capital
        self._positions.clear()
        self._trades.clear()
        self._equity_curve.clear()
        
        symbols = symbols or strategy.config.symbols
        timeframe = strategy.config.timeframe
        
        # Initialize strategy
        await strategy.initialize()
        strategy._status = strategy._status.__class__('running')
        
        # Load data
        all_data = {}
        for symbol in symbols:
            if self._data_loader:
                data = await self._data_loader.load_data(
                    symbol=symbol,
                    start_date=start_date or datetime.now() - timedelta(days=365),
                    end_date=end_date or datetime.now(),
                    timeframe=timeframe
                )
            else:
                # Generate synthetic data
                data = await self._generate_synthetic_data(
                    symbol, start_date, end_date, timeframe
                )
            all_data[symbol] = data
        
        # Combine data
        combined_data = self._combine_data(all_data)
        
        # Run backtest
        for i, (timestamp, row) in enumerate(combined_data.iterrows()):
            # Get current data slice
            current_data = combined_data.iloc[:i+1]
            
            # Generate signals
            signals = await strategy.generate_signals(current_data)
            
            # Execute signals
            for signal in signals:
                await self._execute_signal(signal, row, timestamp)
            
            # Update equity
            self._update_equity(row, timestamp)
        
        # Close all positions at end
        await self._close_all_positions(combined_data.iloc[-1], combined_data.index[-1])
        
        # Calculate performance metrics
        results = self._performance_analyzer.analyze(
            trades=self._trades,
            equity_curve=self._equity_curve,
            initial_capital=self._initial_capital
        )
        
        return results
    
    async def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        timeframe: str
    ) -> pd.DataFrame:
        """Generate synthetic data for backtesting."""
        start = start_date or datetime.now() - timedelta(days=365)
        end = end_date or datetime.now()
        
        # Generate dates
        freq_map = {'1m': 'min', '5m': '5min', '15m': '15min', '1h': 'h', '1d': 'D'}
        freq = freq_map.get(timeframe, 'h')
        
        dates = pd.date_range(start=start, end=end, freq=freq)
        n = len(dates)
        
        # Generate prices
        np.random.seed(hash(symbol) % (2**32))
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        returns = np.random.normal(0.0001, 0.02, n)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': prices * (1 + np.random.uniform(0, 0.01, n)),
            'low': prices * (1 - np.random.uniform(0, 0.01, n)),
            'close': prices,
            'volume': np.random.uniform(100, 10000, n) * prices,
            'symbol': symbol
        }, index=dates)
        
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    def _combine_data(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multi-symbol data."""
        if len(all_data) == 1:
            return list(all_data.values())[0]
        
        # Combine all dataframes
        combined = pd.concat(all_data.values())
        combined = combined.sort_index()
        
        return combined
    
    async def _execute_signal(self, signal, row: pd.Series, timestamp: datetime) -> None:
        """Execute a trading signal."""
        symbol = signal.symbol
        side = signal.side
        price = signal.price or row['close']
        quantity = signal.quantity
        
        # Calculate position size
        if quantity is None:
            quantity = self._calculate_position_size(price)
        
        # Apply slippage
        fill_price = self._slippage_model.apply(price, side)
        
        # Calculate costs
        trade_value = quantity * fill_price
        fee = self._fee_model.calculate(trade_value)
        
        # Execute trade
        if side == OrderSide.BUY or side.value == 'buy':
            if self._cash >= trade_value + fee:
                self._cash -= (trade_value + fee)
                
                if symbol not in self._positions:
                    self._positions[symbol] = {'quantity': 0, 'avg_price': 0}
                
                pos = self._positions[symbol]
                total_quantity = pos['quantity'] + quantity
                pos['avg_price'] = (
                    (pos['avg_price'] * pos['quantity'] + fill_price * quantity) /
                    total_quantity
                )
                pos['quantity'] = total_quantity
                
                self._trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'buy',
                    'price': fill_price,
                    'quantity': quantity,
                    'fee': fee,
                    'value': trade_value
                })
        
        else:  # SELL
            if symbol in self._positions and self._positions[symbol]['quantity'] >= quantity:
                pos = self._positions[symbol]
                pos['quantity'] -= quantity
                self._cash += trade_value - fee
                
                self._trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'sell',
                    'price': fill_price,
                    'quantity': quantity,
                    'fee': fee,
                    'value': trade_value,
                    'pnl': (fill_price - pos['avg_price']) * quantity
                })
                
                if pos['quantity'] == 0:
                    del self._positions[symbol]
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on portfolio."""
        position_value = self._portfolio_value * 0.02  # 2% position size
        return position_value / price
    
    def _update_equity(self, row: pd.Series, timestamp: datetime) -> None:
        """Update portfolio equity."""
        positions_value = 0
        
        for symbol, pos in self._positions.items():
            # Get current price
            current_price = row.get('close', row.get(f'{symbol}_close', 0))
            positions_value += pos['quantity'] * current_price
        
        self._portfolio_value = self._cash + positions_value
        
        self._equity_curve.append({
            'timestamp': timestamp,
            'equity': self._portfolio_value,
            'cash': self._cash,
            'positions_value': positions_value
        })
    
    async def _close_all_positions(self, row: pd.Series, timestamp: datetime) -> None:
        """Close all open positions."""
        for symbol, pos in list(self._positions.items()):
            if pos['quantity'] > 0:
                price = row.get('close', row.get(f'{symbol}_close', 0))
                fill_price = self._slippage_model.apply(price, OrderSide.SELL)
                trade_value = pos['quantity'] * fill_price
                fee = self._fee_model.calculate(trade_value)
                
                self._cash += trade_value - fee
                
                self._trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'sell',
                    'price': fill_price,
                    'quantity': pos['quantity'],
                    'fee': fee,
                    'value': trade_value,
                    'pnl': (fill_price - pos['avg_price']) * pos['quantity']
                })
        
        self._positions.clear()
        self._portfolio_value = self._cash
