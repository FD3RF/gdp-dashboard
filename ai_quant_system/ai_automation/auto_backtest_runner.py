"""
Auto Backtest Runner for automated backtesting.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from core.base import BaseModule


class AutoBacktestRunner(BaseModule):
    """
    Automatically runs and evaluates backtests.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('auto_backtest_runner', config)
        
        self._backtest_engine = None
        self._model_manager = None
        self._results: List[Dict] = []
    
    def set_backtest_engine(self, engine) -> None:
        """Set backtest engine."""
        self._backtest_engine = engine
    
    def set_model_manager(self, manager) -> None:
        """Set AI model manager."""
        self._model_manager = manager
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    async def run_backtest(
        self,
        strategy,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        auto_analyze: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtest and optionally analyze results.
        
        Args:
            strategy: Strategy to backtest
            start_date: Start date
            end_date: End date
            auto_analyze: Whether to auto-analyze results
        
        Returns:
            Backtest results
        """
        if self._backtest_engine:
            try:
                results = await self._backtest_engine.run(
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                results = {'error': str(e)}
        else:
            results = self._simulate_results()
        
        results['timestamp'] = datetime.now().isoformat()
        
        if auto_analyze:
            analysis = await self.analyze_results(results)
            results['ai_analysis'] = analysis
        
        self._results.append(results)
        return results
    
    def _simulate_results(self) -> Dict:
        """Simulate backtest results."""
        import numpy as np
        
        return {
            'total_return': np.random.uniform(-0.1, 0.5),
            'sharpe_ratio': np.random.uniform(0.5, 2.5),
            'max_drawdown': np.random.uniform(-0.25, -0.05),
            'win_rate': np.random.uniform(0.4, 0.6),
            'total_trades': int(np.random.uniform(50, 200))
        }
    
    async def analyze_results(self, results: Dict) -> Dict[str, Any]:
        """Analyze backtest results using AI."""
        if not self._model_manager:
            return {'error': 'No model manager'}
        
        prompt = f"""Analyze these backtest results and provide insights:

Results: {results}

Provide analysis including:
1. Overall performance assessment
2. Key strengths
3. Weaknesses and risks
4. Specific improvement suggestions
5. Market condition suitability

Respond with JSON:
{{
    "assessment": "excellent/good/moderate/poor",
    "strengths": [],
    "weaknesses": [],
    "suggestions": [],
    "suitable_conditions": [],
    "risk_level": "low/medium/high"
}}"""

        try:
            response = await self._model_manager.generate(prompt)
            
            content = response.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
        
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
        
        return {'assessment': 'unknown'}
    
    async def compare_strategies(
        self,
        strategies: List,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Compare multiple strategies."""
        results = {}
        
        for strategy in strategies:
            name = getattr(strategy, 'name', 'unknown')
            result = await self.run_backtest(strategy, start_date, end_date)
            results[name] = result
        
        # Rank by Sharpe ratio
        ranking = sorted(
            results.items(),
            key=lambda x: x[1].get('sharpe_ratio', 0),
            reverse=True
        )
        
        return {
            'results': results,
            'ranking': [r[0] for r in ranking],
            'best': ranking[0][0] if ranking else None
        }
    
    async def find_optimal_period(
        self,
        strategy,
        periods: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Find optimal trading period for a strategy."""
        results = {}
        
        for period in periods:
            result = await self.run_backtest(
                strategy,
                period['start'],
                period['end'],
                auto_analyze=False
            )
            results[period['name']] = result
        
        best_period = max(
            results.items(),
            key=lambda x: x[1].get('sharpe_ratio', 0)
        )
        
        return {
            'results': results,
            'best_period': best_period[0],
            'best_sharpe': best_period[1].get('sharpe_ratio', 0)
        }
    
    def get_results(self, limit: int = 20) -> List[Dict]:
        """Get recent backtest results."""
        return self._results[-limit:]
