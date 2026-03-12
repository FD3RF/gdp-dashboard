"""
Strategy Performance Monitor.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from core.base import BaseModule


class StrategyPerformanceMonitor(BaseModule):
    """
    Monitors strategy performance metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('strategy_performance_monitor', config)
        
        self._strategies: Dict[str, Any] = {}
        self._performance_history: Dict[str, List[Dict]] = {}
        self._metrics_window = self.config.get('metrics_window', 100)
    
    def register_strategy(self, name: str, strategy) -> None:
        """Register a strategy for monitoring."""
        self._strategies[name] = strategy
        self._performance_history[name] = []
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    async def update_metrics(
        self,
        strategy_name: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Update strategy metrics."""
        if strategy_name not in self._performance_history:
            self._performance_history[strategy_name] = []
        
        entry = {
            'timestamp': datetime.now(),
            **metrics
        }
        
        self._performance_history[strategy_name].append(entry)
        
        # Trim history
        if len(self._performance_history[strategy_name]) > self._metrics_window:
            self._performance_history[strategy_name] = \
                self._performance_history[strategy_name][-self._metrics_window:]
    
    def get_strategy_metrics(self, strategy_name: str) -> Dict[str, Any]:
        """Get current metrics for a strategy."""
        if strategy_name not in self._strategies:
            return {'error': 'Strategy not found'}
        
        strategy = self._strategies[strategy_name]
        
        metrics = {
            'name': strategy_name,
            'status': getattr(strategy, 'status', 'unknown'),
            'positions': len(strategy.get_positions()) if hasattr(strategy, 'get_positions') else 0,
        }
        
        if hasattr(strategy, 'get_metrics'):
            metrics.update(strategy.get_metrics())
        
        return metrics
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all strategies."""
        return {
            name: self.get_strategy_metrics(name)
            for name in self._strategies
        }
    
    def get_performance_summary(
        self,
        strategy_name: str,
        period: str = '1d'
    ) -> Dict[str, Any]:
        """Get performance summary for a period."""
        if strategy_name not in self._performance_history:
            return {'error': 'No data'}
        
        history = self._performance_history[strategy_name]
        
        # Filter by period
        period_map = {'1h': 1, '1d': 24, '7d': 168, '30d': 720}
        hours = period_map.get(period, 24)
        
        cutoff = datetime.now() - timedelta(hours=hours)
        filtered = [h for h in history if h['timestamp'] > cutoff]
        
        if not filtered:
            return {'error': 'No data for period'}
        
        # Calculate summary statistics
        df = pd.DataFrame(filtered)
        
        summary = {
            'strategy': strategy_name,
            'period': period,
            'data_points': len(filtered),
            'start': filtered[0]['timestamp'].isoformat(),
            'end': filtered[-1]['timestamp'].isoformat()
        }
        
        # Add numeric metrics
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'timestamp':
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        return summary
    
    def compare_strategies(
        self,
        metric: str = 'total_pnl'
    ) -> List[Dict[str, Any]]:
        """Compare strategies by a metric."""
        comparison = []
        
        for name, history in self._performance_history.items():
            if history:
                latest = history[-1]
                comparison.append({
                    'strategy': name,
                    metric: latest.get(metric, 0),
                    'timestamp': latest['timestamp'].isoformat()
                })
        
        comparison.sort(key=lambda x: x[metric], reverse=True)
        return comparison
    
    def get_performance_chart_data(
        self,
        strategy_name: str,
        metric: str = 'equity',
        hours: int = 24
    ) -> Dict[str, List]:
        """Get data for performance charts."""
        if strategy_name not in self._performance_history:
            return {'timestamps': [], 'values': []}
        
        cutoff = datetime.now() - timedelta(hours=hours)
        filtered = [
            h for h in self._performance_history[strategy_name]
            if h['timestamp'] > cutoff
        ]
        
        return {
            'timestamps': [h['timestamp'].isoformat() for h in filtered],
            'values': [h.get(metric, 0) for h in filtered]
        }
