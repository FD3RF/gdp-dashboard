"""
Base classes for the AI Quant Trading System.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import threading


class Singleton(type):
    """Singleton metaclass for ensuring single instance."""
    
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseModule(ABC):
    """
    Base class for all modules in the system.
    Provides common functionality for initialization, logging, and lifecycle management.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        self._initialized = False
        self._running = False
        self._start_time: Optional[datetime] = None
        self._stop_event = asyncio.Event()
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the module. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the module. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the module. Must be implemented by subclasses."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if module is initialized."""
        return self._initialized
    
    @property
    def is_running(self) -> bool:
        """Check if module is running."""
        return self._running
    
    @property
    def uptime(self) -> Optional[float]:
        """Get module uptime in seconds."""
        if self._start_time is None:
            return None
        return (datetime.now() - self._start_time).total_seconds()
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status information."""
        return {
            'name': self.name,
            'initialized': self._initialized,
            'running': self._running,
            'uptime': self.uptime,
            'config': self.config
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the module."""
        return {
            'healthy': self._initialized and self._running,
            'name': self.name,
            'timestamp': datetime.now().isoformat()
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', running={self._running})>"


class AsyncContextManager:
    """Async context manager mixin for modules."""
    
    async def __aenter__(self):
        await self.initialize()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False


class StateMachine:
    """Simple state machine for managing module states."""
    
    STATES = ['created', 'initialized', 'running', 'paused', 'stopped', 'error']
    
    def __init__(self, initial_state: str = 'created'):
        self._state = initial_state
        self._transitions = {
            'created': ['initialized'],
            'initialized': ['running', 'stopped'],
            'running': ['paused', 'stopped', 'error'],
            'paused': ['running', 'stopped'],
            'stopped': ['initialized'],
            'error': ['initialized', 'stopped']
        }
    
    @property
    def state(self) -> str:
        return self._state
    
    def can_transition_to(self, new_state: str) -> bool:
        return new_state in self._transitions.get(self._state, [])
    
    def transition_to(self, new_state: str) -> bool:
        if self.can_transition_to(new_state):
            self._state = new_state
            return True
        return False
