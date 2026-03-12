"""
Auto Parameter Optimizer for automated parameter tuning.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from core.base import BaseModule


class AutoParameterOptimizer(BaseModule):
    """
    Automatically optimizes strategy parameters using AI.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('auto_parameter_optimizer', config)
        
        self._model_manager = None
        self._backtest_agent = None
        self._optimization_history: List[Dict] = []
    
    def set_model_manager(self, manager) -> None:
        """Set AI model manager."""
        self._model_manager = manager
    
    def set_backtest_agent(self, agent) -> None:
        """Set backtest agent."""
        self._backtest_agent = agent
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    async def optimize(
        self,
        strategy: Dict,
        param_ranges: Dict[str, Tuple],
        objective: str = 'sharpe_ratio',
        method: str = 'ai_guided',
        max_iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters.
        
        Args:
            strategy: Strategy configuration
            param_ranges: Parameter ranges {param: (min, max)}
            objective: Optimization objective
            method: Optimization method
            max_iterations: Maximum iterations
        
        Returns:
            Optimization results
        """
        if method == 'ai_guided':
            return await self._ai_guided_optimization(
                strategy, param_ranges, objective, max_iterations
            )
        elif method == 'grid_search':
            return await self._grid_search_optimization(
                strategy, param_ranges, objective
            )
        else:
            return await self._random_search_optimization(
                strategy, param_ranges, objective, max_iterations
            )
    
    async def _ai_guided_optimization(
        self,
        strategy: Dict,
        param_ranges: Dict[str, Tuple],
        objective: str,
        max_iterations: int
    ) -> Dict[str, Any]:
        """AI-guided parameter optimization."""
        best_params = {}
        best_score = -np.inf
        
        # Initialize with middle values
        current_params = {
            k: (v[0] + v[1]) / 2
            for k, v in param_ranges.items()
        }
        
        history = []
        
        for i in range(max_iterations):
            # Get AI suggestion
            suggested_params = await self._get_ai_suggestion(
                strategy, current_params, history, objective
            )
            
            # Validate and clip parameters
            for param, value in suggested_params.items():
                if param in param_ranges:
                    min_val, max_val = param_ranges[param]
                    suggested_params[param] = np.clip(value, min_val, max_val)
            
            # Run backtest
            strategy_copy = strategy.copy()
            strategy_copy['parameters'] = suggested_params
            
            # Evaluate (simulated)
            score = await self._evaluate_params(strategy_copy, objective)
            
            history.append({
                'iteration': i,
                'params': suggested_params.copy(),
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = suggested_params.copy()
            
            current_params = suggested_params
        
        result = {
            'best_params': best_params,
            'best_score': best_score,
            'iterations': max_iterations,
            'history': history,
            'timestamp': datetime.now().isoformat()
        }
        
        self._optimization_history.append(result)
        return result
    
    async def _get_ai_suggestion(
        self,
        strategy: Dict,
        current_params: Dict,
        history: List[Dict],
        objective: str
    ) -> Dict[str, float]:
        """Get parameter suggestion from AI."""
        prompt = f"""Suggest improved parameters for this trading strategy:

Strategy: {json.dumps(strategy.get('name', 'unknown'), indent=2)}
Current Parameters: {json.dumps(current_params, indent=2)}
Optimization History: {json.dumps(history[-5:], indent=2, default=str)}
Objective: Maximize {objective}

Suggest new parameter values that might improve performance.
Respond with JSON: {{ "param_name": value, ... }}"""

        if self._model_manager:
            try:
                response = await self._model_manager.generate(prompt)
                
                content = response.content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    return json.loads(content[json_start:json_end])
            
            except Exception as e:
                self.logger.error(f"AI suggestion error: {e}")
        
        # Return random adjustment
        return {
            k: v * (1 + np.random.uniform(-0.1, 0.1))
            for k, v in current_params.items()
        }
    
    async def _evaluate_params(
        self,
        strategy: Dict,
        objective: str
    ) -> float:
        """Evaluate parameters (run backtest)."""
        # Simulated evaluation
        return np.random.uniform(0.5, 2.0)
    
    async def _grid_search_optimization(
        self,
        strategy: Dict,
        param_ranges: Dict[str, Tuple],
        objective: str
    ) -> Dict[str, Any]:
        """Grid search optimization."""
        # Generate grid points
        grid_points = 5
        
        param_values = {}
        for param, (min_val, max_val) in param_ranges.items():
            param_values[param] = np.linspace(min_val, max_val, grid_points)
        
        # Generate combinations
        from itertools import product
        combinations = list(product(*param_values.values()))
        
        best_params = {}
        best_score = -np.inf
        
        for combo in combinations[:50]:  # Limit iterations
            params = dict(zip(param_values.keys(), combo))
            
            strategy_copy = strategy.copy()
            strategy_copy['parameters'] = params
            
            score = await self._evaluate_params(strategy_copy, objective)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'method': 'grid_search',
            'combinations_tested': min(len(combinations), 50)
        }
    
    async def _random_search_optimization(
        self,
        strategy: Dict,
        param_ranges: Dict[str, Tuple],
        objective: str,
        max_iterations: int
    ) -> Dict[str, Any]:
        """Random search optimization."""
        best_params = {}
        best_score = -np.inf
        
        for _ in range(max_iterations):
            params = {
                k: np.random.uniform(v[0], v[1])
                for k, v in param_ranges.items()
            }
            
            strategy_copy = strategy.copy()
            strategy_copy['parameters'] = params
            
            score = await self._evaluate_params(strategy_copy, objective)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'method': 'random_search'
        }
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history."""
        return self._optimization_history
