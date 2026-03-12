"""
Strategy Agent for strategy generation and management.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask


class StrategyAgent(BaseAgent):
    """
    Strategy Agent responsible for:
    - Generating trading strategies
    - Managing strategy configurations
    - Evaluating strategy performance
    """
    
    def __init__(
        self,
        name: str = 'strategy',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None,
        strategy_manager = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._strategy_manager = strategy_manager
        self._strategies: Dict[str, Dict] = {}
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Strategy Agent in a quantitative trading system.
Your role is to:
1. Generate trading strategies based on market research
2. Define strategy parameters and rules
3. Optimize strategy configurations
4. Evaluate strategy performance metrics

Focus on creating well-defined, testable strategies with clear entry/exit rules.
Always include risk management parameters."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process a strategy task."""
        if task.type == 'generate_strategy':
            return await self._generate_strategy(task)
        elif task.type == 'optimize_strategy':
            return await self._optimize_strategy(task)
        elif task.type == 'evaluate_strategy':
            return await self._evaluate_strategy(task)
        elif task.type == 'list_strategies':
            return self._list_strategies()
        elif task.type == 'modify_strategy':
            return await self._modify_strategy(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _generate_strategy(self, task: AgentTask) -> Dict[str, Any]:
        """Generate a new trading strategy."""
        strategy_type = task.parameters.get('type', 'trend_following')
        market_condition = task.parameters.get('market_condition', 'trending')
        risk_profile = task.parameters.get('risk_profile', 'moderate')
        research_data = task.parameters.get('research_data', {})
        
        prompt = f"""Generate a {strategy_type} trading strategy for a {market_condition} market.

Risk Profile: {risk_profile}
Research Data: {json.dumps(research_data, indent=2, default=str)}

Create a complete strategy definition with:
1. Strategy name and description
2. Entry conditions (specific, measurable)
3. Exit conditions (take profit, stop loss)
4. Position sizing rules
5. Risk management parameters
6. Timeframe and symbols
7. Indicator parameters

Respond with JSON:
{{
    "name": "strategy_name",
    "type": "strategy_type",
    "description": "brief description",
    "timeframe": "1h",
    "symbols": ["BTC/USDT"],
    "entry_conditions": {{
        "indicators": {{}},
        "rules": []
    }},
    "exit_conditions": {{
        "take_profit": {{}},
        "stop_loss": {{}}
    }},
    "risk_management": {{
        "max_position_size": 0.02,
        "max_drawdown": 0.15
    }},
    "parameters": {{}}
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            strategy = json.loads(json_str)
            strategy['id'] = f"strat_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            strategy['created_at'] = datetime.now().isoformat()
            
            self._strategies[strategy['id']] = strategy
            
            return strategy
        except Exception as e:
            self.logger.error(f"Error parsing strategy: {e}")
            return {'error': str(e), 'raw_response': response}
    
    async def _optimize_strategy(self, task: AgentTask) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        strategy_id = task.parameters.get('strategy_id')
        backtest_results = task.parameters.get('backtest_results', {})
        optimization_target = task.parameters.get('target', 'sharpe_ratio')
        
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            return {'error': f'Strategy {strategy_id} not found'}
        
        prompt = f"""Optimize this trading strategy based on backtest results:

Strategy: {json.dumps(strategy, indent=2)}
Backtest Results: {json.dumps(backtest_results, indent=2, default=str)}
Optimization Target: {optimization_target}

Suggest improved parameters. Respond with JSON:
{{
    "original_params": {{}},
    "optimized_params": {{}},
    "expected_improvement": "description",
    "changes_made": []
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            optimization = json.loads(json_str)
            
            # Apply optimizations
            if 'optimized_params' in optimization:
                strategy['parameters'].update(optimization['optimized_params'])
                strategy['optimized_at'] = datetime.now().isoformat()
            
            return optimization
        except Exception as e:
            return {'error': str(e)}
    
    async def _evaluate_strategy(self, task: AgentTask) -> Dict[str, Any]:
        """Evaluate a strategy's potential."""
        strategy_id = task.parameters.get('strategy_id')
        
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            return {'error': f'Strategy {strategy_id} not found'}
        
        prompt = f"""Evaluate this trading strategy:

{json.dumps(strategy, indent=2)}

Provide evaluation on:
1. Strategy quality (1-10)
2. Risk-reward profile
3. Potential weaknesses
4. Suggested improvements
5. Recommended market conditions

Respond with JSON:
{{
    "quality_score": 7,
    "risk_level": "medium",
    "strengths": [],
    "weaknesses": [],
    "improvements": [],
    "recommended_conditions": []
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            return json.loads(json_str)
        except Exception as e:
            return {'error': str(e)}
    
    def _list_strategies(self) -> List[Dict[str, Any]]:
        """List all stored strategies."""
        return [
            {
                'id': sid,
                'name': s.get('name'),
                'type': s.get('type'),
                'created_at': s.get('created_at')
            }
            for sid, s in self._strategies.items()
        ]
    
    async def _modify_strategy(self, task: AgentTask) -> Dict[str, Any]:
        """Modify an existing strategy."""
        strategy_id = task.parameters.get('strategy_id')
        modifications = task.parameters.get('modifications', {})
        
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            return {'error': f'Strategy {strategy_id} not found'}
        
        # Apply modifications
        for key, value in modifications.items():
            if key in strategy:
                if isinstance(strategy[key], dict) and isinstance(value, dict):
                    strategy[key].update(value)
                else:
                    strategy[key] = value
        
        strategy['modified_at'] = datetime.now().isoformat()
        
        return strategy
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Get a strategy by ID."""
        return self._strategies.get(strategy_id)
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a strategy."""
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy agent status."""
        return {
            **super().get_status(),
            'strategies_count': len(self._strategies),
            'strategy_ids': list(self._strategies.keys())
        }
