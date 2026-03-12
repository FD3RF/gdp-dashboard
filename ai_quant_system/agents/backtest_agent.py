"""
Backtest Agent for automated strategy backtesting.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask


class BacktestAgent(BaseAgent):
    """
    Backtest Agent responsible for:
    - Running strategy backtests
    - Analyzing backtest results
    - Generating performance reports
    """
    
    def __init__(
        self,
        name: str = 'backtest',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None,
        backtest_engine = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._backtest_engine = backtest_engine
        self._results_cache: Dict[str, Dict] = {}
    
    def set_backtest_engine(self, engine) -> None:
        """Set the backtest engine."""
        self._backtest_engine = engine
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Backtest Agent in a quantitative trading system.
Your role is to:
1. Execute strategy backtests
2. Analyze performance metrics
3. Identify strategy strengths and weaknesses
4. Recommend improvements based on backtest results

Focus on providing objective, data-driven analysis."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process a backtest task."""
        if task.type == 'run_backtest':
            return await self._run_backtest(task)
        elif task.type == 'analyze_results':
            return await self._analyze_results(task)
        elif task.type == 'compare_strategies':
            return await self._compare_strategies(task)
        elif task.type == 'generate_report':
            return await self._generate_report(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _run_backtest(self, task: AgentTask) -> Dict[str, Any]:
        """Run a strategy backtest."""
        strategy = task.parameters.get('strategy', {})
        start_date = task.parameters.get('start_date')
        end_date = task.parameters.get('end_date')
        initial_capital = task.parameters.get('initial_capital', 100000)
        
        # Run backtest using engine if available
        if self._backtest_engine:
            try:
                result = await self._backtest_engine.run(
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital
                )
                
                # Cache results
                result_id = f"bt_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                self._results_cache[result_id] = result
                
                return result
            except Exception as e:
                return {'error': str(e)}
        
        # Simulate backtest results
        return await self._simulate_backtest(strategy, initial_capital)
    
    async def _simulate_backtest(
        self,
        strategy: Dict,
        initial_capital: float
    ) -> Dict[str, Any]:
        """Simulate backtest results for demonstration."""
        import numpy as np
        
        # Generate simulated performance
        days = 365
        daily_returns = np.random.normal(0.0005, 0.02, days)
        cumulative_returns = (1 + daily_returns).cumprod()
        
        final_value = initial_capital * cumulative_returns[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate metrics
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        win_rate = np.sum(daily_returns > 0) / len(daily_returns)
        
        return {
            'strategy_name': strategy.get('name', 'Unknown'),
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': int(np.random.uniform(50, 200)),
            'profit_factor': np.random.uniform(1.1, 2.5),
            'avg_trade_duration': '4.5 hours',
            'backtest_period': '1 year',
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_max_drawdown(self, cumulative_returns) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return float(np.min(drawdown))
    
    async def _analyze_results(self, task: AgentTask) -> Dict[str, Any]:
        """Analyze backtest results."""
        result_id = task.parameters.get('result_id')
        results = task.parameters.get('results') or self._results_cache.get(result_id)
        
        if not results:
            return {'error': 'No results to analyze'}
        
        prompt = f"""Analyze these backtest results:

{json.dumps(results, indent=2, default=str)}

Provide analysis including:
1. Overall assessment
2. Strengths of the strategy
3. Weaknesses and risks
4. Specific recommendations for improvement
5. Risk considerations

Respond with JSON:
{{
    "overall_assessment": "good/poor/excellent",
    "strengths": [],
    "weaknesses": [],
    "recommendations": [],
    "risk_notes": [],
    "suitable_conditions": []
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            return {'error': str(e), 'raw_analysis': response}
    
    async def _compare_strategies(self, task: AgentTask) -> Dict[str, Any]:
        """Compare multiple strategy backtest results."""
        results_ids = task.parameters.get('result_ids', [])
        
        results = []
        for rid in results_ids:
            if rid in self._results_cache:
                results.append(self._results_cache[rid])
        
        if len(results) < 2:
            return {'error': 'Need at least 2 strategies to compare'}
        
        # Create comparison table
        comparison = {
            'strategies': [],
            'best_by_metric': {}
        }
        
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for i, result in enumerate(results):
            comparison['strategies'].append({
                'name': result.get('strategy_name', f'Strategy {i+1}'),
                'metrics': {m: result.get(m) for m in metrics}
            })
        
        # Determine best by each metric
        for metric in metrics:
            if metric == 'max_drawdown':
                # Lower is better
                best = min(comparison['strategies'], 
                          key=lambda x: abs(x['metrics'].get(metric, 0)))
            else:
                # Higher is better
                best = max(comparison['strategies'],
                          key=lambda x: x['metrics'].get(metric, 0))
            comparison['best_by_metric'][metric] = best['name']
        
        return comparison
    
    async def _generate_report(self, task: AgentTask) -> Dict[str, Any]:
        """Generate a detailed backtest report."""
        result_id = task.parameters.get('result_id')
        results = self._results_cache.get(result_id) or task.parameters.get('results')
        
        if not results:
            return {'error': 'No results for report'}
        
        prompt = f"""Generate a comprehensive backtest report:

Results: {json.dumps(results, indent=2, default=str)}

Create a professional report with:
1. Executive Summary
2. Strategy Overview
3. Performance Analysis
4. Risk Analysis
5. Trade Analysis
6. Recommendations

Format as a detailed text report."""

        report = await self.generate_response(prompt)
        
        return {
            'result_id': result_id,
            'report': report,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_cached_results(self, result_id: str) -> Optional[Dict]:
        """Get cached backtest results."""
        return self._results_cache.get(result_id)
    
    def list_cached_results(self) -> List[str]:
        """List all cached result IDs."""
        return list(self._results_cache.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """Get backtest agent status."""
        return {
            **super().get_status(),
            'cached_results': len(self._results_cache),
            'has_backtest_engine': self._backtest_engine is not None
        }
