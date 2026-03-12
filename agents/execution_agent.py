"""
Execution Agent for trade execution management.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask


class ExecutionAgent(BaseAgent):
    """
    Execution Agent responsible for:
    - Managing trade execution
    - Optimizing order routing
    - Minimizing slippage
    """
    
    def __init__(
        self,
        name: str = 'execution',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None,
        order_manager = None,
        exchange_adapter = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._order_manager = order_manager
        self._exchange_adapter = exchange_adapter
        self._execution_history: List[Dict] = []
    
    def set_order_manager(self, manager) -> None:
        """Set the order manager."""
        self._order_manager = manager
    
    def set_exchange_adapter(self, adapter) -> None:
        """Set the exchange adapter."""
        self._exchange_adapter = adapter
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Execution Agent in a quantitative trading system.
Your role is to:
1. Execute trades efficiently
2. Minimize market impact and slippage
3. Choose optimal execution algorithms
4. Monitor execution quality

Focus on achieving best execution prices while managing risk."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process an execution task."""
        if task.type == 'execute_order':
            return await self._execute_order(task)
        elif task.type == 'cancel_order':
            return await self._cancel_order(task)
        elif task.type == 'check_status':
            return await self._check_status(task)
        elif task.type == 'optimize_execution':
            return await self._optimize_execution(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _execute_order(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a trade order."""
        order = task.parameters.get('order', {})
        
        symbol = order.get('symbol')
        side = order.get('side')
        quantity = order.get('quantity')
        order_type = order.get('type', 'market')
        price = order.get('price')
        time_in_force = order.get('time_in_force', 'GTC')
        
        # Validate order
        if not symbol or not side or not quantity:
            return {'error': 'Missing required order parameters'}
        
        execution_result = {
            'order': order,
            'status': 'pending',
            'timestamp': datetime.now().isoformat()
        }
        
        # Use order manager if available
        if self._order_manager:
            try:
                result = await self._order_manager.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=order_type,
                    price=price
                )
                execution_result.update(result)
            except Exception as e:
                execution_result['error'] = str(e)
                execution_result['status'] = 'failed'
        else:
            # Simulate execution
            execution_result.update({
                'order_id': f"ord_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'status': 'filled',
                'filled_quantity': quantity,
                'filled_price': price or 50000.0,  # Simulated price
                'fee': quantity * 0.001,  # 0.1% fee
                'slippage': 0.0001
            })
        
        self._execution_history.append(execution_result)
        
        return execution_result
    
    async def _cancel_order(self, task: AgentTask) -> Dict[str, Any]:
        """Cancel an order."""
        order_id = task.parameters.get('order_id')
        
        if self._order_manager:
            try:
                result = await self._order_manager.cancel_order(order_id)
                return result
            except Exception as e:
                return {'error': str(e), 'order_id': order_id}
        
        return {'order_id': order_id, 'status': 'cancelled'}
    
    async def _check_status(self, task: AgentTask) -> Dict[str, Any]:
        """Check order status."""
        order_id = task.parameters.get('order_id')
        
        if self._order_manager:
            try:
                status = await self._order_manager.get_order_status(order_id)
                return status
            except Exception as e:
                return {'error': str(e), 'order_id': order_id}
        
        return {'order_id': order_id, 'status': 'unknown'}
    
    async def _optimize_execution(self, task: AgentTask) -> Dict[str, Any]:
        """Determine optimal execution strategy."""
        order = task.parameters.get('order', {})
        market_conditions = task.parameters.get('market_conditions', {})
        
        symbol = order.get('symbol')
        quantity = order.get('quantity')
        urgency = order.get('urgency', 'normal')
        
        prompt = f"""Recommend optimal execution strategy for:

Order: {json.dumps(order, indent=2)}
Market Conditions: {json.dumps(market_conditions, indent=2, default=str)}
Urgency: {urgency}

Recommend:
1. Execution algorithm (market, limit, TWAP, VWAP, etc.)
2. Order splitting strategy
3. Timing considerations
4. Risk mitigation

Respond with JSON:
{{
    "algorithm": "recommended_algorithm",
    "split_orders": true/false,
    "num_splits": number,
    "time_window": "duration",
    "parameters": {{}},
    "rationale": "explanation"
}}"""

        response = await self.generate_response(prompt)
        
        try:
            json_str = response[response.find('{'):response.rfind('}')+1]
            recommendation = json.loads(json_str)
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            self.logger.warning(f"Failed to parse execution recommendation JSON: {e}")
            recommendation = {
                'algorithm': 'market',
                'split_orders': False,
                'rationale': 'Default recommendation'
            }
        
        return recommendation
    
    def get_execution_history(self, limit: int = 100) -> List[Dict]:
        """Get recent execution history."""
        return self._execution_history[-limit:]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Calculate execution statistics."""
        if not self._execution_history:
            return {'total_executions': 0}
        
        total = len(self._execution_history)
        filled = sum(1 for e in self._execution_history if e.get('status') == 'filled')
        total_slippage = sum(e.get('slippage', 0) for e in self._execution_history)
        total_fees = sum(e.get('fee', 0) for e in self._execution_history)
        
        return {
            'total_executions': total,
            'filled': filled,
            'success_rate': filled / total if total > 0 else 0,
            'total_slippage': total_slippage,
            'total_fees': total_fees,
            'avg_slippage': total_slippage / total if total > 0 else 0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get execution agent status."""
        return {
            **super().get_status(),
            'execution_history_count': len(self._execution_history),
            'has_order_manager': self._order_manager is not None
        }
