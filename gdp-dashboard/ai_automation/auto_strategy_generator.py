"""
Auto Strategy Generator for automated strategy creation.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from core.base import BaseModule


class AutoStrategyGenerator(BaseModule):
    """
    Automatically generates trading strategies using AI.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('auto_strategy_generator', config)
        
        self._model_manager = None
        self._generated_strategies: List[Dict] = []
        self._template_dir = self.config.get('template_dir', 'templates/strategies')
    
    def set_model_manager(self, manager) -> None:
        """Set the AI model manager."""
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
    
    async def generate_strategy(
        self,
        market_condition: str = 'trending',
        risk_profile: str = 'moderate',
        timeframe: str = '1h',
        symbols: Optional[List[str]] = None,
        custom_requirements: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a new trading strategy.
        
        Args:
            market_condition: Market condition ('trending', 'ranging', 'volatile')
            risk_profile: Risk profile ('conservative', 'moderate', 'aggressive')
            timeframe: Trading timeframe
            symbols: Target symbols
            custom_requirements: Additional requirements
        
        Returns:
            Generated strategy
        """
        symbols = symbols or ['BTC/USDT']
        
        prompt = f"""Generate a complete trading strategy with the following requirements:

Market Condition: {market_condition}
Risk Profile: {risk_profile}
Timeframe: {timeframe}
Symbols: {symbols}
Custom Requirements: {custom_requirements or 'None'}

Generate a complete strategy with:
1. Strategy name and description
2. Entry conditions (specific indicators and thresholds)
3. Exit conditions (take profit, stop loss)
4. Position sizing rules
5. Risk management parameters
6. Default parameter values

Respond with JSON format:
{{
    "name": "strategy_name",
    "description": "description",
    "type": "trend/mean_reversion/momentum/etc",
    "timeframe": "{timeframe}",
    "symbols": {symbols},
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
        "max_drawdown": 0.15,
        "stop_loss_pct": 0.05
    }},
    "parameters": {{}}
}}"""

        if self._model_manager:
            try:
                response = await self._model_manager.generate(prompt)
                
                # Extract JSON
                content = response.content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    strategy_json = content[json_start:json_end]
                    strategy = json.loads(strategy_json)
                    
                    strategy['generated_at'] = datetime.now().isoformat()
                    strategy['id'] = f"gen_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    self._generated_strategies.append(strategy)
                    
                    return strategy
            
            except Exception as e:
                self.logger.error(f"Error generating strategy: {e}")
        
        # Return default strategy if generation fails
        return self._get_default_strategy(market_condition, timeframe, symbols)
    
    def _get_default_strategy(
        self,
        market_condition: str,
        timeframe: str,
        symbols: List[str]
    ) -> Dict:
        """Get default strategy template."""
        return {
            'name': f'auto_{market_condition}_strategy',
            'description': f'Auto-generated {market_condition} strategy',
            'type': 'trend' if market_condition == 'trending' else 'mean_reversion',
            'timeframe': timeframe,
            'symbols': symbols,
            'entry_conditions': {
                'indicators': {'sma_fast': 10, 'sma_slow': 30},
                'rules': ['price > sma_fast', 'sma_fast > sma_slow']
            },
            'exit_conditions': {
                'take_profit': {'pct': 0.05},
                'stop_loss': {'pct': 0.02}
            },
            'risk_management': {
                'max_position_size': 0.02,
                'max_drawdown': 0.15
            },
            'parameters': {}
        }
    
    async def generate_code(self, strategy: Dict) -> str:
        """Generate Python code for a strategy."""
        prompt = f"""Generate Python code for this trading strategy:

{json.dumps(strategy, indent=2)}

Create a complete strategy class that inherits from BaseStrategy.
Include all necessary imports and type hints.
Make the code production-ready.

```python
# Your code here
```"""

        if self._model_manager:
            try:
                response = await self._model_manager.generate(prompt)
                return response.content
            except Exception as e:
                self.logger.error(f"Error generating code: {e}")
        
        return ""
    
    def get_generated_strategies(self) -> List[Dict]:
        """Get all generated strategies."""
        return self._generated_strategies
    
    def validate_strategy(self, strategy: Dict) -> Dict[str, Any]:
        """Validate a strategy configuration."""
        required_fields = ['name', 'type', 'timeframe', 'entry_conditions', 'exit_conditions']
        
        errors = []
        for field in required_fields:
            if field not in strategy:
                errors.append(f"Missing required field: {field}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
