"""
Monitoring Agent for system and strategy monitoring.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask


class MonitoringAgent(BaseAgent):
    """
    Monitoring Agent responsible for:
    - System health monitoring
    - Strategy performance tracking
    - Alert generation
    """
    
    def __init__(
        self,
        name: str = 'monitoring',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._monitored_items: Dict[str, Dict] = {}
        self._alerts: List[Dict] = []
        self._metrics_history: Dict[str, List[Dict]] = {}
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Monitoring Agent in a quantitative trading system.
Your role is to:
1. Monitor system health and performance
2. Track strategy metrics
3. Generate alerts for anomalies
4. Provide status updates

Be proactive in identifying potential issues before they become critical."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process a monitoring task."""
        if task.type == 'check_health':
            return await self._check_health(task)
        elif task.type == 'check_strategy':
            return await self._check_strategy(task)
        elif task.type == 'generate_alert':
            return await self._generate_alert(task)
        elif task.type == 'get_status_report':
            return await self._get_status_report(task)
        elif task.type == 'analyze_trends':
            return await self._analyze_trends(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    def register_monitor(self, name: str, check_func, interval: int = 60) -> None:
        """Register an item to monitor."""
        self._monitored_items[name] = {
            'check_func': check_func,
            'interval': interval,
            'last_check': None,
            'status': 'unknown'
        }
    
    async def _check_health(self, task: AgentTask) -> Dict[str, Any]:
        """Check system health."""
        components = task.parameters.get('components', list(self._monitored_items.keys()))
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        for component in components:
            if component in self._monitored_items:
                check_func = self._monitored_items[component]['check_func']
                try:
                    if asyncio.iscoroutinefunction(check_func):
                        status = await check_func()
                    else:
                        status = check_func()
                    
                    health_report['components'][component] = {
                        'status': 'healthy' if status else 'unhealthy',
                        'details': status
                    }
                except Exception as e:
                    health_report['components'][component] = {
                        'status': 'error',
                        'error': str(e)
                    }
            else:
                health_report['components'][component] = {
                    'status': 'not_monitored'
                }
        
        # Overall health
        all_healthy = all(
            c.get('status') == 'healthy'
            for c in health_report['components'].values()
        )
        health_report['overall'] = 'healthy' if all_healthy else 'degraded'
        
        return health_report
    
    async def _check_strategy(self, task: AgentTask) -> Dict[str, Any]:
        """Check strategy performance."""
        strategy_id = task.parameters.get('strategy_id')
        metrics = task.parameters.get('metrics', {})
        
        report = {
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'alerts': []
        }
        
        # Check for concerning metrics
        if metrics.get('drawdown', 0) > 0.1:
            report['alerts'].append({
                'level': 'warning',
                'message': f"High drawdown: {metrics['drawdown']:.2%}"
            })
        
        if metrics.get('daily_pnl', 0) < 0 and abs(metrics['daily_pnl']) > 1000:
            report['alerts'].append({
                'level': 'warning',
                'message': f"Significant daily loss: {metrics['daily_pnl']}"
            })
        
        # Store metrics history
        if strategy_id not in self._metrics_history:
            self._metrics_history[strategy_id] = []
        self._metrics_history[strategy_id].append({
            'timestamp': report['timestamp'],
            'metrics': metrics
        })
        
        return report
    
    async def _generate_alert(self, task: AgentTask) -> Dict[str, Any]:
        """Generate an alert."""
        alert_type = task.parameters.get('alert_type', 'info')
        message = task.parameters.get('message', '')
        details = task.parameters.get('details', {})
        
        alert = {
            'id': f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'type': alert_type,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False
        }
        
        self._alerts.append(alert)
        
        # Keep last 100 alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]
        
        return alert
    
    async def _get_status_report(self, task: AgentTask) -> Dict[str, Any]:
        """Get comprehensive status report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitored_items': len(self._monitored_items),
            'recent_alerts': self._alerts[-10:],
            'item_statuses': {}
        }
        
        for name, config in self._monitored_items.items():
            report['item_statuses'][name] = config['status']
        
        # AI summary
        prompt = f"""Generate a brief status summary:

Monitoring Data: {json.dumps(report, indent=2, default=str)}

Provide a concise 2-3 sentence summary of the current system status."""

        summary = await self.generate_response(prompt)
        report['ai_summary'] = summary
        
        return report
    
    async def _analyze_trends(self, task: AgentTask) -> Dict[str, Any]:
        """Analyze metric trends."""
        strategy_id = task.parameters.get('strategy_id')
        lookback_hours = task.parameters.get('lookback_hours', 24)
        
        history = self._metrics_history.get(strategy_id, [])
        
        if len(history) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Filter by time
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent = [
            h for h in history
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]
        
        # Calculate trends
        if not recent:
            return {'error': 'No data in lookback period'}
        
        # Simple trend analysis
        metrics_over_time = {}
        for entry in recent:
            for metric, value in entry.get('metrics', {}).items():
                if metric not in metrics_over_time:
                    metrics_over_time[metric] = []
                if isinstance(value, (int, float)):
                    metrics_over_time[metric].append(value)
        
        trends = {}
        for metric, values in metrics_over_time.items():
            if len(values) >= 2:
                change = values[-1] - values[0]
                change_pct = change / values[0] if values[0] != 0 else 0
                trends[metric] = {
                    'start': values[0],
                    'end': values[-1],
                    'change': change,
                    'change_pct': change_pct,
                    'direction': 'increasing' if change > 0 else 'decreasing'
                }
        
        return {
            'strategy_id': strategy_id,
            'lookback_hours': lookback_hours,
            'data_points': len(recent),
            'trends': trends
        }
    
    def get_alerts(self, unacknowledged_only: bool = False) -> List[Dict]:
        """Get alerts."""
        if unacknowledged_only:
            return [a for a in self._alerts if not a['acknowledged']]
        return self._alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring agent status."""
        return {
            **super().get_status(),
            'monitored_items': len(self._monitored_items),
            'total_alerts': len(self._alerts),
            'unacknowledged_alerts': sum(1 for a in self._alerts if not a['acknowledged'])
        }


import asyncio
