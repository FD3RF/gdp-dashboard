"""
Risk Dashboard for aggregated risk monitoring.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from core.base import BaseModule


class RiskDashboard(BaseModule):
    """
    Aggregated risk dashboard combining all risk metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('risk_dashboard', config)
        
        self._position_sizing = None
        self._stop_loss = None
        self._drawdown = None
        self._exposure = None
        self._volatility = None
        
        self._alerts: List[Dict] = []
    
    def set_modules(
        self,
        position_sizing=None,
        stop_loss=None,
        drawdown=None,
        exposure=None,
        volatility=None
    ) -> None:
        """Set risk sub-modules."""
        self._position_sizing = position_sizing
        self._stop_loss = stop_loss
        self._drawdown = drawdown
        self._exposure = exposure
        self._volatility = volatility
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    def get_dashboard(self, portfolio_value: float) -> Dict[str, Any]:
        """
        Get complete risk dashboard.
        
        Args:
            portfolio_value: Current portfolio value
        
        Returns:
            Complete risk dashboard data
        """
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': portfolio_value,
            'overall_risk_level': 'normal',
            'alerts': [],
            'metrics': {}
        }
        
        # Drawdown metrics
        if self._drawdown:
            dd_status = self._drawdown.get_status()
            dashboard['metrics']['drawdown'] = dd_status
            
            if dd_status.get('halted'):
                dashboard['overall_risk_level'] = 'critical'
                dashboard['alerts'].append({
                    'level': 'critical',
                    'message': 'Trading halted due to drawdown'
                })
            elif dd_status.get('current_drawdown', 0) > 0.1:
                dashboard['overall_risk_level'] = 'high'
        
        # Exposure metrics
        if self._exposure:
            exp_summary = self._exposure.get_exposure_summary(portfolio_value)
            dashboard['metrics']['exposure'] = exp_summary
            
            if exp_summary.get('leverage', 0) > 2.5:
                dashboard['alerts'].append({
                    'level': 'warning',
                    'message': f"High leverage: {exp_summary['leverage']:.2f}x"
                })
        
        # Volatility status
        if self._volatility:
            vol_status = self._volatility.get_status()
            dashboard['metrics']['volatility'] = vol_status
            
            if vol_status.get('high_vol_symbols'):
                dashboard['alerts'].append({
                    'level': 'warning',
                    'message': f"High volatility in: {vol_status['high_vol_symbols']}"
                })
        
        # Stop loss summary
        if self._stop_loss:
            stops = self._stop_loss.get_all_stops()
            dashboard['metrics']['stop_loss'] = {
                'active_stops': len(stops),
                'positions_protected': list(stops.keys())
            }
        
        # Determine overall risk level
        if dashboard['overall_risk_level'] == 'normal':
            if len([a for a in dashboard['alerts'] if a['level'] == 'warning']) > 2:
                dashboard['overall_risk_level'] = 'warning'
        
        return dashboard
    
    def add_alert(self, level: str, message: str, details: Dict = None) -> None:
        """Add a risk alert."""
        alert = {
            'level': level,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self._alerts.append(alert)
        
        # Keep last 100 alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts."""
        return self._alerts[-limit:]
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()
    
    def calculate_risk_score(self) -> float:
        """
        Calculate overall risk score (0-100).
        Higher score = higher risk.
        """
        score = 0
        
        # Drawdown contribution (0-40 points)
        if self._drawdown:
            dd = self._drawdown._current_drawdown
            score += min(dd * 200, 40)
        
        # Leverage contribution (0-30 points)
        if self._exposure:
            lev = self._exposure._calculate_leverage()
            score += min(lev * 10, 30)
        
        # Volatility contribution (0-30 points)
        if self._volatility:
            high_vol_count = len(self._volatility.get_status().get('high_vol_symbols', []))
            score += min(high_vol_count * 10, 30)
        
        return min(score, 100)
