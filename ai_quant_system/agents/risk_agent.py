"""
Risk Agent for risk analysis and management.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask


class RiskAgent(BaseModule):
    """
    Risk Agent responsible for:
    - Analyzing trade risks
    - Monitoring portfolio risk
    - Providing risk recommendations
    """
    
    def __init__(
        self,
        name: str = 'risk',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None,
        risk_manager = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._risk_manager = risk_manager
        self._risk_limits = self.config.get('risk_limits', {
            'max_position_size': 0.02,
            'max_portfolio_leverage': 3.0,
            'max_drawdown': 0.15,
            'max_daily_loss': 0.05,
            'max_single_trade_risk': 0.01
        })
    
    def set_risk_manager(self, manager) -> None:
        """Set the risk manager."""
        self._risk_manager = manager
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Risk Agent in a quantitative trading system.
Your role is to:
1. Assess risk of potential trades
2. Monitor portfolio-level risk
3. Ensure compliance with risk limits
4. Provide risk mitigation recommendations

Always prioritize capital preservation and risk management."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process a risk task."""
        if task.type == 'assess_trade':
            return await self._assess_trade(task)
        elif task.type == 'portfolio_risk':
            return await self._portfolio_risk(task)
        elif task.type == 'check_limits':
            return await self._check_limits(task)
        elif task.type == 'risk_report':
            return await self._risk_report(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _assess_trade(self, task: AgentTask) -> Dict[str, Any]:
        """Assess risk of a potential trade."""
        trade = task.parameters.get('trade', {})
        portfolio = task.parameters.get('portfolio', {})
        
        symbol = trade.get('symbol')
        side = trade.get('side')
        size = trade.get('size')
        price = trade.get('price')
        leverage = trade.get('leverage', 1.0)
        
        # Calculate position value
        position_value = size * price * leverage
        portfolio_value = portfolio.get('total_value', 100000)
        
        # Position size as % of portfolio
        position_pct = position_value / portfolio_value
        
        # Risk assessment
        risks = []
        risk_level = 'low'
        
        if position_pct > self._risk_limits['max_position_size']:
            risks.append(f"Position size ({position_pct:.2%}) exceeds limit")
            risk_level = 'high'
        
        if leverage > self._risk_limits['max_portfolio_leverage']:
            risks.append(f"Leverage ({leverage}x) exceeds limit")
            risk_level = 'high'
        
        # AI analysis
        prompt = f"""Assess the risk of this trade:

Trade: {json.dumps(trade, indent=2)}
Portfolio: {json.dumps(portfolio, indent=2, default=str)}
Risk Limits: {json.dumps(self._risk_limits, indent=2)}

Provide risk assessment with:
1. Overall risk level (low/medium/high/critical)
2. Specific risks identified
3. Risk score (0-100)
4. Recommendations

Respond with JSON."""

        try:
            response = await self.generate_response(prompt)
            json_str = response[response.find('{'):response.rfind('}')+1]
            ai_assessment = json.loads(json_str)
        except:
            ai_assessment = {
                'risk_level': risk_level,
                'risks': risks,
                'risk_score': 50
            }
        
        return {
            'trade': trade,
            'risk_level': ai_assessment.get('risk_level', risk_level),
            'risks': risks,
            'position_pct': position_pct,
            'leverage': leverage,
            'approved': risk_level != 'high' and len(risks) == 0,
            'recommendations': ai_assessment.get('recommendations', []),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _portfolio_risk(self, task: AgentTask) -> Dict[str, Any]:
        """Analyze portfolio-level risk."""
        portfolio = task.parameters.get('portfolio', {})
        
        positions = portfolio.get('positions', [])
        total_value = portfolio.get('total_value', 0)
        
        # Calculate basic metrics
        total_exposure = sum(p.get('value', 0) for p in positions)
        leverage = total_exposure / total_value if total_value > 0 else 0
        
        # Concentration risk
        position_values = [p.get('value', 0) for p in positions]
        if position_values:
            max_position = max(position_values)
            concentration = max_position / total_value if total_value > 0 else 0
        else:
            concentration = 0
        
        return {
            'total_value': total_value,
            'total_exposure': total_exposure,
            'leverage': leverage,
            'position_count': len(positions),
            'concentration_risk': concentration,
            'risk_limits': self._risk_limits,
            'leverage_status': 'within_limits' if leverage <= self._risk_limits['max_portfolio_leverage'] else 'exceeded',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _check_limits(self, task: AgentTask) -> Dict[str, Any]:
        """Check if portfolio is within risk limits."""
        portfolio = task.parameters.get('portfolio', {})
        
        violations = []
        warnings = []
        
        # Check various limits
        total_value = portfolio.get('total_value', 0)
        daily_pnl = portfolio.get('daily_pnl', 0)
        drawdown = portfolio.get('drawdown', 0)
        leverage = portfolio.get('leverage', 1.0)
        
        if abs(daily_pnl / total_value) > self._risk_limits['max_daily_loss']:
            violations.append('Daily loss limit exceeded')
        
        if abs(drawdown) > self._risk_limits['max_drawdown']:
            violations.append('Max drawdown exceeded')
        
        if leverage > self._risk_limits['max_portfolio_leverage']:
            violations.append('Leverage limit exceeded')
        
        # Warnings (80% of limit)
        if abs(daily_pnl / total_value) > self._risk_limits['max_daily_loss'] * 0.8:
            warnings.append('Approaching daily loss limit')
        
        if abs(drawdown) > self._risk_limits['max_drawdown'] * 0.8:
            warnings.append('Approaching max drawdown')
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'current_metrics': {
                'daily_pnl_pct': daily_pnl / total_value if total_value > 0 else 0,
                'drawdown': drawdown,
                'leverage': leverage
            },
            'limits': self._risk_limits,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _risk_report(self, task: AgentTask) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        portfolio = task.parameters.get('portfolio', {})
        
        # Gather all risk metrics
        limits_check = await self._check_limits(
            AgentTask(id='tmp', type='check_limits', description='', parameters={'portfolio': portfolio})
        )
        portfolio_risk = await self._portfolio_risk(
            AgentTask(id='tmp', type='portfolio_risk', description='', parameters={'portfolio': portfolio})
        )
        
        prompt = f"""Generate a comprehensive risk report:

Portfolio Risk Metrics: {json.dumps(portfolio_risk, indent=2)}
Limits Check: {json.dumps(limits_check, indent=2)}

Create a professional risk report with:
1. Executive Summary
2. Current Risk Exposure
3. Limit Compliance
4. Risk Recommendations
5. Action Items"""

        report = await self.generate_response(prompt)
        
        return {
            'report': report,
            'metrics': portfolio_risk,
            'compliance': limits_check,
            'generated_at': datetime.now().isoformat()
        }
    
    def update_limits(self, limits: Dict[str, Any]) -> None:
        """Update risk limits."""
        self._risk_limits.update(limits)
    
    def get_limits(self) -> Dict[str, Any]:
        """Get current risk limits."""
        return self._risk_limits.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get risk agent status."""
        return {
            **super().get_status(),
            'risk_limits': self._risk_limits
        }


# Fix import
from core.base import BaseModule
