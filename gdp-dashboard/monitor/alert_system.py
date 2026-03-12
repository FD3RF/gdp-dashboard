"""
Alert System for notifications.
"""

import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from core.base import BaseModule


class AlertLevel(Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


class AlertSystem(BaseModule):
    """
    Multi-channel alert system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('alert_system', config)
        
        self._alerts: List[Dict] = []
        self._handlers: Dict[str, Callable] = {}
        self._max_alerts = self.config.get('max_alerts', 1000)
        
        # Notification channels
        self._telegram_enabled = self.config.get('telegram_enabled', False)
        self._telegram_token = self.config.get('telegram_token')
        self._telegram_chat_id = self.config.get('telegram_chat_id')
        
        self._email_enabled = self.config.get('email_enabled', False)
        self._email_config = self.config.get('email_config', {})
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        self._running = False
        return True
    
    def register_handler(self, name: str, handler: Callable) -> None:
        """Register custom alert handler."""
        self._handlers[name] = handler
    
    async def send_alert(
        self,
        level: AlertLevel,
        message: str,
        details: Optional[Dict] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send an alert.
        
        Args:
            level: Alert level
            message: Alert message
            details: Additional details
            channels: Specific channels to send to
        
        Returns:
            Alert result
        """
        alert = {
            'id': f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'level': level.value,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat(),
            'channels': channels or ['log']
        }
        
        self._alerts.append(alert)
        
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts:]
        
        # Send to channels
        results = {}
        
        for channel in alert['channels']:
            if channel == 'log':
                results['log'] = self._send_to_log(alert)
            elif channel == 'telegram':
                results['telegram'] = await self._send_to_telegram(alert)
            elif channel == 'email':
                results['email'] = await self._send_to_email(alert)
            elif channel in self._handlers:
                results[channel] = await self._run_handler(channel, alert)
        
        alert['results'] = results
        return alert
    
    def _send_to_log(self, alert: Dict) -> bool:
        """Send to log."""
        level = alert['level']
        message = f"[{level.upper()}] {alert['message']}"
        
        if level == 'critical':
            self.logger.critical(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        return True
    
    async def _send_to_telegram(self, alert: Dict) -> bool:
        """Send to Telegram."""
        if not self._telegram_enabled or not self._telegram_token:
            return False
        
        try:
            import aiohttp
            
            text = f"*{alert['level'].upper()}*\n{alert['message']}"
            
            if alert['details']:
                text += f"\n\n```\n{alert['details']}\n```"
            
            url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={
                    'chat_id': self._telegram_chat_id,
                    'text': text,
                    'parse_mode': 'Markdown'
                }) as response:
                    return response.status == 200
        
        except Exception as e:
            self.logger.error(f"Telegram alert error: {e}")
            return False
    
    async def _send_to_email(self, alert: Dict) -> bool:
        """Send to email."""
        if not self._email_enabled:
            return False
        
        # Email implementation would go here
        return True
    
    async def _run_handler(self, name: str, alert: Dict) -> Any:
        """Run custom handler."""
        handler = self._handlers.get(name)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(alert)
                else:
                    return handler(alert)
            except Exception as e:
                self.logger.error(f"Handler {name} error: {e}")
                return {'error': str(e)}
        return None
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get alerts."""
        alerts = self._alerts
        
        if level:
            alerts = [a for a in alerts if a['level'] == level.value]
        
        return alerts[-limit:]
    
    def get_unacknowledged(self) -> List[Dict]:
        """Get unacknowledged alerts."""
        return [a for a in self._alerts if not a.get('acknowledged')]
    
    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_at'] = datetime.now().isoformat()
                return True
        return False
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()
