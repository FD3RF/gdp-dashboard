"""
Strategy Combiner for multi-strategy portfolio management.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from core.base import BaseModule
from strategies.base_strategy import BaseStrategy, Signal, StrategyConfig


class StrategyCombiner(BaseModule):
    """
    Combines multiple strategies and manages signal aggregation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('strategy_combiner', config)
        
        self._strategies: Dict[str, BaseStrategy] = {}
        self._weights: Dict[str, float] = {}
        self._signal_buffer: List[Signal] = []
        
        # Combination parameters
        self._combination_method = self.config.get('method', 'weighted_vote')
        self._min_signal_agreement = self.config.get('min_agreement', 0.5)
        self._max_total_position = self.config.get('max_total_position', 0.1)
    
    async def initialize(self) -> bool:
        """Initialize the combiner."""
        self.logger.info("Initializing strategy combiner...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the combiner."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the combiner."""
        for strategy in self._strategies.values():
            await strategy.stop()
        self._running = False
        return True
    
    def register_strategy(
        self,
        strategy: BaseStrategy,
        weight: float = 1.0
    ) -> None:
        """Register a strategy with a weight."""
        self._strategies[strategy.config.name] = strategy
        self._weights[strategy.config.name] = weight
        self.logger.info(f"Registered strategy: {strategy.config.name} (weight: {weight})")
    
    def unregister_strategy(self, name: str) -> None:
        """Unregister a strategy."""
        if name in self._strategies:
            del self._strategies[name]
            del self._weights[name]
    
    def set_weight(self, name: str, weight: float) -> None:
        """Update strategy weight."""
        if name in self._weights:
            self._weights[name] = weight
    
    async def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate combined signals from all strategies."""
        all_signals = []
        
        # Collect signals from all strategies
        for name, strategy in self._strategies.items():
            if strategy.status.value == 'running':
                try:
                    signals = await strategy.on_data(data)
                    for signal in signals:
                        signal.metadata['strategy'] = name
                        signal.metadata['weight'] = self._weights.get(name, 1.0)
                    all_signals.extend(signals)
                except Exception as e:
                    self.logger.error(f"Error getting signals from {name}: {e}")
        
        # Combine signals based on method
        if self._combination_method == 'weighted_vote':
            combined = self._weighted_vote_combine(all_signals)
        elif self._combination_method == 'average':
            combined = self._average_combine(all_signals)
        elif self._combination_method == 'max_strength':
            combined = self._max_strength_combine(all_signals)
        else:
            combined = all_signals
        
        # Store in buffer
        self._signal_buffer.extend(combined)
        if len(self._signal_buffer) > 1000:
            self._signal_buffer = self._signal_buffer[-1000:]
        
        return combined
    
    def _weighted_vote_combine(self, signals: List[Signal]) -> List[Signal]:
        """Combine signals using weighted voting."""
        if not signals:
            return []
        
        # Group signals by symbol
        symbol_signals: Dict[str, List[Signal]] = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        combined = []
        
        for symbol, sigs in symbol_signals.items():
            # Calculate weighted votes
            buy_votes = sum(s.metadata.get('weight', 1) for s in sigs if s.side.value == 'buy')
            sell_votes = sum(s.metadata.get('weight', 1) for s in sigs if s.side.value == 'sell')
            total_votes = buy_votes + sell_votes
            
            if total_votes == 0:
                continue
            
            # Check if there's enough agreement
            buy_ratio = buy_votes / total_votes
            sell_ratio = sell_votes / total_votes
            
            if buy_ratio >= self._min_signal_agreement:
                # Create combined buy signal
                avg_strength = np.mean([s.strength for s in sigs if s.side.value == 'buy'])
                combined.append(Signal(
                    symbol=symbol,
                    side=signals[0].side.__class__('buy'),
                    signal_type='combined_entry',
                    strength=avg_strength * buy_ratio,
                    metadata={
                        'method': 'weighted_vote',
                        'agreement': buy_ratio,
                        'strategies': [s.metadata.get('strategy') for s in sigs if s.side.value == 'buy']
                    }
                ))
            
            elif sell_ratio >= self._min_signal_agreement:
                # Create combined sell signal
                avg_strength = np.mean([s.strength for s in sigs if s.side.value == 'sell'])
                combined.append(Signal(
                    symbol=symbol,
                    side=signals[0].side.__class__('sell'),
                    signal_type='combined_entry',
                    strength=avg_strength * sell_ratio,
                    metadata={
                        'method': 'weighted_vote',
                        'agreement': sell_ratio,
                        'strategies': [s.metadata.get('strategy') for s in sigs if s.side.value == 'sell']
                    }
                ))
        
        return combined
    
    def _average_combine(self, signals: List[Signal]) -> List[Signal]:
        """Combine signals by averaging."""
        if not signals:
            return []
        
        symbol_signals: Dict[str, List[Signal]] = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        combined = []
        
        for symbol, sigs in symbol_signals.items():
            # Calculate weighted average strength
            total_weight = sum(s.metadata.get('weight', 1) for s in sigs)
            
            buy_strength = sum(
                s.strength * s.metadata.get('weight', 1)
                for s in sigs if s.side.value == 'buy'
            ) / total_weight
            
            sell_strength = sum(
                s.strength * s.metadata.get('weight', 1)
                for s in sigs if s.side.value == 'sell'
            ) / total_weight
            
            net_strength = buy_strength - sell_strength
            
            from core.constants import OrderSide
            if abs(net_strength) > 0.2:  # Threshold
                combined.append(Signal(
                    symbol=symbol,
                    side=OrderSide.BUY if net_strength > 0 else OrderSide.SELL,
                    signal_type='combined_entry',
                    strength=abs(net_strength),
                    metadata={
                        'method': 'average',
                        'buy_strength': buy_strength,
                        'sell_strength': sell_strength
                    }
                ))
        
        return combined
    
    def _max_strength_combine(self, signals: List[Signal]) -> List[Signal]:
        """Select signal with maximum strength."""
        if not signals:
            return []
        
        symbol_signals: Dict[str, List[Signal]] = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        combined = []
        
        for symbol, sigs in symbol_signals.items():
            # Select signal with max strength
            max_signal = max(sigs, key=lambda s: s.strength * s.metadata.get('weight', 1))
            
            combined.append(Signal(
                symbol=symbol,
                side=max_signal.side,
                signal_type='combined_entry',
                strength=max_signal.strength,
                metadata={
                    'method': 'max_strength',
                    'source_strategy': max_signal.metadata.get('strategy')
                }
            ))
        
        return combined
    
    def get_strategy_status(self) -> Dict[str, Dict]:
        """Get status of all strategies."""
        return {
            name: strategy.get_status()
            for name, strategy in self._strategies.items()
        }
    
    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent combined signals."""
        return [s.to_dict() for s in self._signal_buffer[-limit:]]
    
    def get_status(self) -> Dict[str, Any]:
        """Get combiner status."""
        return {
            **super().get_status(),
            'strategies': list(self._strategies.keys()),
            'weights': self._weights,
            'combination_method': self._combination_method,
            'signal_buffer_size': len(self._signal_buffer)
        }
