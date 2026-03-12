"""
Custom exceptions for the AI Quant Trading System.
"""

from typing import Optional, Dict, Any


class QuantSystemException(Exception):
    """Base exception for all quant system errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or 'UNKNOWN_ERROR'
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }
    
    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class DataException(QuantSystemException):
    """Exception for data-related errors."""
    
    def __init__(self, message: str, source: Optional[str] = None, **kwargs):
        kwargs.setdefault('error_code', 'DATA_ERROR')
        if source:
            kwargs['details'] = {**kwargs.get('details', {}), 'source': source}
        super().__init__(message, **kwargs)


class DataSourceException(DataException):
    """Exception for data source connection errors."""
    
    def __init__(self, message: str, source: str, **kwargs):
        kwargs['error_code'] = 'DATA_SOURCE_ERROR'
        super().__init__(message, source=source, **kwargs)


class DataValidationException(DataException):
    """Exception for data validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        kwargs['error_code'] = 'DATA_VALIDATION_ERROR'
        if field:
            kwargs['details'] = {**kwargs.get('details', {}), 'field': field}
        super().__init__(message, **kwargs)


class StrategyException(QuantSystemException):
    """Exception for strategy-related errors."""
    
    def __init__(self, message: str, strategy_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('error_code', 'STRATEGY_ERROR')
        if strategy_name:
            kwargs['details'] = {**kwargs.get('details', {}), 'strategy': strategy_name}
        super().__init__(message, **kwargs)


class StrategyNotFoundException(StrategyException):
    """Exception when strategy is not found."""
    
    def __init__(self, strategy_name: str, **kwargs):
        kwargs['error_code'] = 'STRATEGY_NOT_FOUND'
        super().__init__(f"Strategy '{strategy_name}' not found", strategy_name, **kwargs)


class RiskException(QuantSystemException):
    """Exception for risk management errors."""
    
    def __init__(self, message: str, risk_type: Optional[str] = None, **kwargs):
        kwargs.setdefault('error_code', 'RISK_ERROR')
        if risk_type:
            kwargs['details'] = {**kwargs.get('details', {}), 'risk_type': risk_type}
        super().__init__(message, **kwargs)


class RiskLimitExceededException(RiskException):
    """Exception when risk limits are exceeded."""
    
    def __init__(self, message: str, limit_type: str, current_value: float, limit_value: float, **kwargs):
        kwargs['error_code'] = 'RISK_LIMIT_EXCEEDED'
        kwargs['details'] = {
            **kwargs.get('details', {}),
            'limit_type': limit_type,
            'current_value': current_value,
            'limit_value': limit_value
        }
        super().__init__(message, risk_type=limit_type, **kwargs)


class ExecutionException(QuantSystemException):
    """Exception for trade execution errors."""
    
    def __init__(self, message: str, order_id: Optional[str] = None, **kwargs):
        kwargs.setdefault('error_code', 'EXECUTION_ERROR')
        if order_id:
            kwargs['details'] = {**kwargs.get('details', {}), 'order_id': order_id}
        super().__init__(message, **kwargs)


class OrderRejectedError(ExecutionException):
    """Exception when an order is rejected."""
    
    def __init__(self, message: str, order_id: str, reason: str, **kwargs):
        kwargs['error_code'] = 'ORDER_REJECTED'
        kwargs['details'] = {**kwargs.get('details', {}), 'reason': reason}
        super().__init__(message, order_id, **kwargs)


class AgentException(QuantSystemException):
    """Exception for AI agent errors."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('error_code', 'AGENT_ERROR')
        if agent_name:
            kwargs['details'] = {**kwargs.get('details', {}), 'agent': agent_name}
        super().__init__(message, **kwargs)


class AgentTimeoutException(AgentException):
    """Exception when agent operation times out."""
    
    def __init__(self, agent_name: str, timeout: float, **kwargs):
        kwargs['error_code'] = 'AGENT_TIMEOUT'
        kwargs['details'] = {**kwargs.get('details', {}), 'timeout': timeout}
        super().__init__(f"Agent '{agent_name}' timed out after {timeout}s", agent_name, **kwargs)


class ModelException(AgentException):
    """Exception for AI model errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        kwargs['error_code'] = 'MODEL_ERROR'
        if model_name:
            kwargs['details'] = {**kwargs.get('details', {}), 'model': model_name}
        super().__init__(message, **kwargs)


class ConfigurationException(QuantSystemException):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        kwargs.setdefault('error_code', 'CONFIG_ERROR')
        if config_key:
            kwargs['details'] = {**kwargs.get('details', {}), 'config_key': config_key}
        super().__init__(message, **kwargs)


class BacktestException(QuantSystemException):
    """Exception for backtesting errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'BACKTEST_ERROR')
        super().__init__(message, **kwargs)


class ConnectionException(QuantSystemException):
    """Exception for connection errors."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, **kwargs):
        kwargs.setdefault('error_code', 'CONNECTION_ERROR')
        if endpoint:
            kwargs['details'] = {**kwargs.get('details', {}), 'endpoint': endpoint}
        super().__init__(message, **kwargs)


class RateLimitException(QuantSystemException):
    """Exception for rate limit errors."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        kwargs['error_code'] = 'RATE_LIMIT_ERROR'
        if retry_after:
            kwargs['details'] = {**kwargs.get('details', {}), 'retry_after': retry_after}
        super().__init__(message, **kwargs)
