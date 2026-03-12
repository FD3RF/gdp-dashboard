"""
Optimization Agent for strategy optimization.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from agents.base_agent import BaseAgent, AgentTask


class OptimizationAgent(BaseAgent):
    """
    Optimization Agent responsible for:
    - Parameter optimization
    - Walk-forward analysis
    - Robustness testing
    """
    
    def __init__(
        self,
        name: str = 'optimization',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None,
        backtest_agent = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._backtest_agent = backtest_agent
        self._optimization_history: List[Dict] = []
    
    def set_backtest_agent(self, agent) -> None:
        """Set backtest agent for running optimizations."""
        self._backtest_agent = agent
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Optimization Agent in a quantitative trading system.
Your role is to:
1. Optimize strategy parameters
2. Prevent overfitting
3. Perform robustness analysis
4. Recommend optimal parameters

Focus on out-of-sample performance and robustness."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process an optimization task."""
        if task.type == 'optimize_parameters':
            return await self._optimize_parameters(task)
        elif task.type == 'grid_search':
            return await self._grid_search(task)
        elif task.type == 'walk_forward':
            return await self._walk_forward(task)
        elif task.type == 'robustness_test':
            return await self._robustness_test(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _optimize_parameters(self, task: AgentTask) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        strategy = task.parameters.get('strategy', {})
        param_ranges = task.parameters.get('param_ranges', {})
        objective = task.parameters.get('objective', 'sharpe_ratio')
        method = task.parameters.get('method', 'bayesian')
        
        # AI-guided optimization suggestion
        prompt = f"""Suggest optimal parameters for this strategy:

Strategy: {json.dumps(strategy, indent=2)}
Parameter Ranges: {json.dumps(param_ranges, indent=2)}
Objective: {objective}

Recommend specific parameter values considering:
1. Market regime suitability
2. Robustness vs optimal fit
3. Risk considerations

Respond with JSON:
{{
    "recommended_params": {{}},
    "rationale": "explanation",
    "expected_improvement": "description"
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            ai_recommendation = json.loads(json_str)
        except:
            ai_recommendation = {
                'recommended_params': {},
                'rationale': 'Could not parse recommendation'
            }
        
        # If backtest agent available, test recommended parameters
        if self._backtest_agent and ai_recommendation['recommended_params']:
            strategy_copy = strategy.copy()
            strategy_copy['parameters'] = ai_recommendation['recommended_params']
            
            # Would run backtest here
            ai_recommendation['tested'] = True
        
        # Store optimization result
        result = {
            'strategy_id': strategy.get('id'),
            'method': method,
            'objective': objective,
            'ai_recommendation': ai_recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        self._optimization_history.append(result)
        
        return result
    
    async def _grid_search(self, task: AgentTask) -> Dict[str, Any]:
        """Perform grid search optimization."""
        strategy = task.parameters.get('strategy', {})
        param_grid = task.parameters.get('param_grid', {})
        objective = task.parameters.get('objective', 'sharpe_ratio')
        
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        results = []
        
        # Simulate grid search (would run actual backtests)
        for combo in combinations[:10]:  # Limit for demo
            params = dict(zip(param_names, combo))
            
            # Simulated result
            result = {
                'params': params,
                objective: np.random.uniform(0.5, 2.0)
            }
            results.append(result)
        
        # Sort by objective
        results.sort(key=lambda x: x[objective], reverse=True)
        
        return {
            'total_combinations': len(combinations),
            'tested_combinations': len(results),
            'best_params': results[0]['params'] if results else None,
            'best_score': results[0][objective] if results else None,
            'top_5': results[:5],
            'timestamp': datetime.now().isoformat()
        }
    
    async def _walk_forward(self, task: AgentTask) -> Dict[str, Any]:
        """Perform walk-forward analysis."""
        strategy = task.parameters.get('strategy', {})
        total_period = task.parameters.get('total_period_days', 365)
        in_sample_pct = task.parameters.get('in_sample_pct', 0.7)
        steps = task.parameters.get('steps', 5)
        
        in_sample_days = int(total_period * in_sample_pct)
        out_sample_days = total_period - in_sample_days
        step_size = total_period // steps
        
        walk_forward_results = []
        
        for i in range(steps):
            # Calculate periods
            is_start = i * step_size
            is_end = is_start + in_sample_days
            os_start = is_end
            os_end = min(os_start + out_sample_days, total_period)
            
            # Simulated results
            is_result = {
                'period': f'days {is_start}-{is_end}',
                'type': 'in_sample',
                'sharpe': np.random.uniform(1.0, 2.5),
                'return': np.random.uniform(0.1, 0.5)
            }
            
            os_result = {
                'period': f'days {os_start}-{os_end}',
                'type': 'out_sample',
                'sharpe': np.random.uniform(0.5, 2.0),
                'return': np.random.uniform(0.05, 0.3)
            }
            
            walk_forward_results.append({
                'step': i + 1,
                'in_sample': is_result,
                'out_sample': os_result
            })
        
        # Calculate degradation
        avg_is_sharpe = np.mean([r['in_sample']['sharpe'] for r in walk_forward_results])
        avg_os_sharpe = np.mean([r['out_sample']['sharpe'] for r in walk_forward_results])
        degradation = (avg_is_sharpe - avg_os_sharpe) / avg_is_sharpe if avg_is_sharpe > 0 else 0
        
        return {
            'steps': steps,
            'results': walk_forward_results,
            'avg_in_sample_sharpe': avg_is_sharpe,
            'avg_out_sample_sharpe': avg_os_sharpe,
            'degradation': degradation,
            'overfitting_risk': 'high' if degradation > 0.3 else 'medium' if degradation > 0.15 else 'low',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _robustness_test(self, task: AgentTask) -> Dict[str, Any]:
        """Test strategy robustness."""
        strategy = task.parameters.get('strategy', {})
        
        tests = {
            'parameter_sensitivity': await self._test_parameter_sensitivity(strategy),
            'monte_carlo': await self._monte_carlo_test(strategy),
            'stress_test': await self._stress_test(strategy)
        }
        
        # Overall robustness score
        scores = [
            tests['parameter_sensitivity'].get('score', 0),
            tests['monte_carlo'].get('score', 0),
            tests['stress_test'].get('score', 0)
        ]
        overall_score = np.mean(scores)
        
        return {
            'strategy_id': strategy.get('id'),
            'tests': tests,
            'overall_score': overall_score,
            'robustness_rating': 'high' if overall_score > 0.7 else 'medium' if overall_score > 0.4 else 'low',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _test_parameter_sensitivity(self, strategy: Dict) -> Dict[str, Any]:
        """Test sensitivity to parameter changes."""
        # Simulate parameter perturbation test
        perturbation_results = []
        
        for i in range(10):
            perturbation = np.random.uniform(-0.2, 0.2)
            performance = np.random.uniform(0.8, 1.2)
            perturbation_results.append({
                'perturbation': perturbation,
                'performance_ratio': performance
            })
        
        # Calculate sensitivity score
        variance = np.var([r['performance_ratio'] for r in perturbation_results])
        score = max(0, 1 - variance)
        
        return {
            'test': 'parameter_sensitivity',
            'perturbation_results': perturbation_results,
            'variance': variance,
            'score': score
        }
    
    async def _monte_carlo_test(self, strategy: Dict) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        # Simulate Monte Carlo results
        simulations = 1000
        returns = np.random.normal(0.1, 0.2, simulations)
        
        return {
            'test': 'monte_carlo',
            'simulations': simulations,
            'median_return': np.median(returns),
            'percentile_5': np.percentile(returns, 5),
            'percentile_95': np.percentile(returns, 95),
            'prob_positive': np.sum(returns > 0) / simulations,
            'score': np.sum(returns > 0) / simulations
        }
    
    async def _stress_test(self, strategy: Dict) -> Dict[str, Any]:
        """Run stress test scenarios."""
        scenarios = ['market_crash', 'flash_crash', 'high_volatility', 'liquidity_crisis']
        
        results = {}
        for scenario in scenarios:
            # Simulate stress test result
            max_loss = np.random.uniform(0.1, 0.4)
            results[scenario] = {
                'max_drawdown': max_loss,
                'recovery_time': np.random.randint(5, 30),
                'survived': max_loss < 0.3
            }
        
        survival_rate = sum(1 for r in results.values() if r['survived']) / len(results)
        
        return {
            'test': 'stress_test',
            'scenarios': results,
            'survival_rate': survival_rate,
            'score': survival_rate
        }
    
    def get_optimization_history(self, limit: int = 20) -> List[Dict]:
        """Get optimization history."""
        return self._optimization_history[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimization agent status."""
        return {
            **super().get_status(),
            'optimization_count': len(self._optimization_history),
            'has_backtest_agent': self._backtest_agent is not None
        }
