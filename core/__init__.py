"""
Core module for AI Quant Trading System.
Provides base classes, exceptions, utilities, memory, and state management.
"""

from .base import BaseModule, Singleton
from .exceptions import (
    QuantSystemException,
    DataException,
    StrategyException,
    RiskException,
    ExecutionException,
    AgentException,
    ConfigurationException
)
from .constants import (
    TimeFrame,
    OrderSide,
    OrderType,
    PositionSide,
    StrategyStatus,
    RiskLevel
)
from .utils import (
    setup_logger,
    async_retry,
    timing_decorator,
    validate_symbol,
    calculate_pnl
)

# New modules for improved architecture
from .memory import MemoryStore, MemoryEntry, MemoryType
from .project_state import ProjectStateManager, Task, TaskStatus, ProjectPhase
from .security import (
    SecurityConfig,
    safe_json_parse,
    sanitize_for_log,
    generate_secure_token,
    validate_file_path,
    mask_api_key,
    safe_execute
)

__all__ = [
    # Base
    'BaseModule',
    'Singleton',
    # Exceptions
    'QuantSystemException',
    'DataException',
    'StrategyException',
    'RiskException',
    'ExecutionException',
    'AgentException',
    'ConfigurationException',
    # Constants
    'TimeFrame',
    'OrderSide',
    'OrderType',
    'PositionSide',
    'StrategyStatus',
    'RiskLevel',
    # Utils
    'setup_logger',
    'async_retry',
    'timing_decorator',
    'validate_symbol',
    'calculate_pnl',
    # Memory (v2)
    'MemoryStore',
    'MemoryEntry',
    'MemoryType',
    # Project State (v2)
    'ProjectStateManager',
    'Task',
    'TaskStatus',
    'ProjectPhase'
]
