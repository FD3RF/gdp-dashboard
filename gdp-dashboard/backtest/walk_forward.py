"""
Walk Forward Analysis for robust strategy validation.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from core.base import BaseModule
from backtest.backtest_engine import BacktestEngine
from backtest.historical_loader import HistoricalDataLoader


class WalkForwardAnalysis(BaseModule):
    """
    Walk-forward analysis for strategy validation and optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('walk_forward_analysis', config)
        
        self._in_sample_pct = self.config.get('in_sample_pct', 0.7)
        self._num_steps = self.config.get('num_steps', 5)
        self._anchor = self.config.get('anchor', False)
        
        self._backtest_engine: Optional[BacktestEngine] = None
        self._data_loader: Optional[HistoricalDataLoader] = None
        self._results: List[Dict] = []
    
    def set_backtest_engine(self, engine: BacktestEngine) -> None:
        """Set the backtest engine."""
        self._backtest_engine = engine
    
    def set_data_loader(self, loader: HistoricalDataLoader) -> None:
        """Set the data loader."""
        self._data_loader = loader
    
    async def initialize(self) -> bool:
        """Initialize the analysis."""
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the analysis."""
        self._running = True
        return True
    
    async def stop(self) -> bool:
        """Stop the analysis."""
        self._running = False
        return True
    
    async def run(
        self,
        strategy_class,
        strategy_config: Dict,
        start_date: datetime,
        end_date: datetime,
        optimization_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis.
        
        Args:
            strategy_class: Strategy class
            strategy_config: Strategy configuration
            start_date: Start date
            end_date: End date
            optimization_func: Optional optimization function
        
        Returns:
            Walk-forward results
        """
        total_days = (end_date - start_date).days
        step_days = total_days // self._num_steps
        in_sample_days = int(step_days * self._in_sample_pct)
        out_sample_days = step_days - in_sample_days
        
        results = []
        
        for i in range(self._num_steps):
            # Calculate periods
            if self._anchor:
                is_start = start_date
            else:
                is_start = start_date + timedelta(days=i * step_days)
            
            is_end = is_start + timedelta(days=in_sample_days)
            os_start = is_end
            os_end = os_start + timedelta(days=out_sample_days)
            
            if os_end > end_date:
                os_end = end_date
            
            # Run in-sample optimization
            if optimization_func:
                optimal_params = await optimization_func(
                    strategy_class,
                    strategy_config,
                    is_start,
                    is_end
                )
                strategy_config['parameters'].update(optimal_params)
            
            # Create strategy instance
            from strategies.base_strategy import StrategyConfig
            config = StrategyConfig(**strategy_config)
            strategy = strategy_class(config)
            
            # Run out-of-sample test
            if self._backtest_engine:
                os_result = await self._backtest_engine.run(
                    strategy=strategy,
                    start_date=os_start,
                    end_date=os_end
                )
            else:
                os_result = self._simulate_result()
            
            results.append({
                'step': i + 1,
                'in_sample': {
                    'start': is_start.isoformat(),
                    'end': is_end.isoformat()
                },
                'out_sample': {
                    'start': os_start.isoformat(),
                    'end': os_end.isoformat()
                },
                'result': os_result
            })
        
        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate(results)
        
        self._results = results
        
        return {
            'steps': self._num_steps,
            'in_sample_pct': self._in_sample_pct,
            'results': results,
            'aggregate': aggregate
        }
    
    def _simulate_result(self) -> Dict[str, float]:
        """Simulate backtest result for demonstration."""
        return {
            'total_return': np.random.uniform(-0.1, 0.3),
            'sharpe_ratio': np.random.uniform(0.5, 2.0),
            'max_drawdown': np.random.uniform(-0.2, -0.05),
            'win_rate': np.random.uniform(0.4, 0.6)
        }
    
    def _calculate_aggregate(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate walk-forward metrics."""
        returns = [r['result']['total_return'] for r in results]
        sharpe_ratios = [r['result']['sharpe_ratio'] for r in results]
        max_drawdowns = [r['result']['max_drawdown'] for r in results]
        win_rates = [r['result']['win_rate'] for r in results]
        
        # Calculate WFO efficiency
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        consistency = len([r for r in returns if r > 0]) / len(returns)
        
        return {
            'avg_return': np.mean(returns),
            'std_return': std_return,
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'consistency': consistency,
            'wfo_efficiency': avg_return / (std_return + 0.0001)
        }
    
    def get_results(self) -> List[Dict]:
        """Get all walk-forward results."""
        return self._results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of walk-forward analysis."""
        if not self._results:
            return {}
        
        return {
            'total_steps': len(self._results),
            'best_step': max(self._results, key=lambda x: x['result']['total_return']),
            'worst_step': min(self._results, key=lambda x: x['result']['total_return'])
        }
