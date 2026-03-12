"""
System Health Monitor for infrastructure monitoring.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import psutil
from core.base import BaseModule


class SystemHealthMonitor(BaseModule):
    """
    Monitors system health metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('system_health_monitor', config)
        
        self._check_interval = self.config.get('check_interval', 60)
        self._cpu_threshold = self.config.get('cpu_threshold', 80)
        self._memory_threshold = self.config.get('memory_threshold', 80)
        self._disk_threshold = self.config.get('disk_threshold', 90)
        
        self._health_history: List[Dict] = []
        self._modules: Dict[str, Any] = {}
        self._alerts: List[Dict] = []
    
    def register_module(self, name: str, module) -> None:
        """Register a module for health monitoring."""
        self._modules[name] = module
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        asyncio.create_task(self._monitoring_loop())
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                health = await self.check_health()
                self._health_history.append(health)
                
                if len(self._health_history) > 1000:
                    self._health_history = self._health_history[-1000:]
                
                await asyncio.sleep(self._check_interval)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self._check_interval)
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'system': self._check_system(),
            'modules': await self._check_modules(),
            'alerts': []
        }
        
        # Generate alerts
        if health['system']['cpu_pct'] > self._cpu_threshold:
            health['alerts'].append({
                'level': 'warning',
                'message': f"High CPU: {health['system']['cpu_pct']:.1f}%"
            })
        
        if health['system']['memory_pct'] > self._memory_threshold:
            health['alerts'].append({
                'level': 'warning',
                'message': f"High Memory: {health['system']['memory_pct']:.1f}%"
            })
        
        if health['system']['disk_pct'] > self._disk_threshold:
            health['alerts'].append({
                'level': 'warning',
                'message': f"High Disk: {health['system']['disk_pct']:.1f}%"
            })
        
        self._alerts.extend(health['alerts'])
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]
        
        return health
    
    def _check_system(self) -> Dict[str, Any]:
        """Check system metrics."""
        return {
            'cpu_pct': psutil.cpu_percent(interval=1),
            'memory_pct': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'disk_pct': psutil.disk_usage('/').percent,
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
            'network_connections': len(psutil.net_connections()),
            'process_count': len(psutil.pids())
        }
    
    async def _check_modules(self) -> Dict[str, Any]:
        """Check registered modules."""
        modules_status = {}
        
        for name, module in self._modules.items():
            try:
                if hasattr(module, 'health_check'):
                    status = await module.health_check()
                elif hasattr(module, 'is_running'):
                    status = {'running': module.is_running}
                else:
                    status = {'status': 'unknown'}
                
                modules_status[name] = status
            except Exception as e:
                modules_status[name] = {'error': str(e)}
        
        return modules_status
    
    def get_health_history(self, hours: int = 1) -> List[Dict]:
        """Get health history."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            h for h in self._health_history
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]
    
    def get_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts."""
        return self._alerts[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        if not self._health_history:
            return {'status': 'no_data'}
        
        latest = self._health_history[-1]
        
        return {
            'overall_status': 'healthy' if not latest['alerts'] else 'warning',
            'cpu': latest['system']['cpu_pct'],
            'memory': latest['system']['memory_pct'],
            'disk': latest['system']['disk_pct'],
            'modules_status': {
                k: v.get('healthy', v.get('running', 'unknown'))
                for k, v in latest['modules'].items()
            }
        }
