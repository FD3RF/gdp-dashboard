"""
Settings and configuration management for the AI Quant Trading System.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


class DatabaseSettings(BaseModel):
    """Database configuration settings."""
    host: str = Field(default='localhost', description='Database host')
    port: int = Field(default=5432, description='Database port')
    name: str = Field(default='quant_system', description='Database name')
    user: str = Field(default='quant_user', description='Database user')
    password: str = Field(default='', description='Database password')
    pool_size: int = Field(default=10, description='Connection pool size')
    echo: bool = Field(default=False, description='Echo SQL queries')
    
    @property
    def url(self) -> str:
        """Get database connection URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        """Get synchronous database connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseModel):
    """Redis configuration settings."""
    host: str = Field(default='localhost', description='Redis host')
    port: int = Field(default=6379, description='Redis port')
    db: int = Field(default=0, description='Redis database number')
    password: Optional[str] = Field(default=None, description='Redis password')
    pool_size: int = Field(default=10, description='Connection pool size')
    
    @property
    def url(self) -> str:
        """Get Redis connection URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class OllamaSettings(BaseModel):
    """Ollama AI model configuration settings."""
    host: str = Field(default='localhost', description='Ollama host')
    port: int = Field(default=11434, description='Ollama port')
    model: str = Field(default='qwen2.5-coder:latest', description='Model name')
    timeout: int = Field(default=300, description='Request timeout in seconds')
    temperature: float = Field(default=0.7, description='Model temperature')
    max_tokens: int = Field(default=4096, description='Maximum tokens')
    
    @property
    def base_url(self) -> str:
        """Get Ollama base URL."""
        return f"http://{self.host}:{self.port}"


class RiskSettings(BaseModel):
    """Risk management configuration settings."""
    max_position_size_pct: float = Field(default=0.02, description='Max position size as % of portfolio')
    max_portfolio_leverage: float = Field(default=3.0, description='Maximum portfolio leverage')
    max_drawdown_pct: float = Field(default=0.15, description='Maximum drawdown percentage')
    max_daily_loss_pct: float = Field(default=0.05, description='Maximum daily loss percentage')
    max_single_trade_risk_pct: float = Field(default=0.01, description='Max single trade risk %')
    max_correlated_positions: int = Field(default=3, description='Max correlated positions')
    var_confidence_level: float = Field(default=0.95, description='VaR confidence level')
    
    @field_validator('max_position_size_pct', 'max_drawdown_pct', 'max_daily_loss_pct')
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError('Percentage must be between 0 and 1')
        return v


class ExecutionSettings(BaseModel):
    """Trade execution configuration settings."""
    default_slippage: float = Field(default=0.0005, description='Default slippage tolerance')
    default_timeout: int = Field(default=30, description='Default order timeout in seconds')
    max_retries: int = Field(default=3, description='Maximum retry attempts')
    retry_delay: float = Field(default=1.0, description='Retry delay in seconds')
    smart_routing_enabled: bool = Field(default=True, description='Enable smart order routing')
    twap_default_duration: int = Field(default=300, description='Default TWAP duration in seconds')
    vwap_default_duration: int = Field(default=300, description='Default VWAP duration in seconds')


class ExchangeSettings(BaseModel):
    """Exchange configuration settings."""
    name: str = Field(..., description='Exchange name')
    api_key: str = Field(default='', description='API key')
    api_secret: str = Field(default='', description='API secret')
    testnet: bool = Field(default=True, description='Use testnet')
    rate_limit: int = Field(default=1200, description='Rate limit (requests per minute)')
    timeout: int = Field(default=30, description='Request timeout')
    
    class Config:
        extra = 'allow'


class DataSettings(BaseModel):
    """Data collection configuration settings."""
    default_exchange: str = Field(default='binance', description='Default data exchange')
    default_timeframe: str = Field(default='1h', description='Default timeframe')
    cache_enabled: bool = Field(default=True, description='Enable data caching')
    cache_ttl: int = Field(default=3600, description='Cache TTL in seconds')
    max_workers: int = Field(default=4, description='Maximum worker threads')
    
    # Supported symbols
    symbols: List[str] = Field(
        default=['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'],
        description='Trading symbols to collect'
    )


class BacktestSettings(BaseModel):
    """Backtesting configuration settings."""
    initial_capital: float = Field(default=100000.0, description='Initial capital')
    commission_rate: float = Field(default=0.001, description='Commission rate')
    slippage_model: str = Field(default='percentage', description='Slippage model')
    default_slippage: float = Field(default=0.0005, description='Default slippage')
    benchmark_symbol: str = Field(default='BTC/USDT', description='Benchmark symbol')
    
    class Config:
        extra = 'allow'


class MonitorSettings(BaseModel):
    """Monitoring configuration settings."""
    enabled: bool = Field(default=True, description='Enable monitoring')
    log_level: str = Field(default='INFO', description='Logging level')
    log_file: Optional[str] = Field(default=None, description='Log file path')
    metrics_port: int = Field(default=9090, description='Metrics server port')
    health_check_interval: int = Field(default=60, description='Health check interval')
    
    # Alert settings
    telegram_enabled: bool = Field(default=False, description='Enable Telegram alerts')
    telegram_token: Optional[str] = Field(default=None, description='Telegram bot token')
    telegram_chat_id: Optional[str] = Field(default=None, description='Telegram chat ID')
    
    email_enabled: bool = Field(default=False, description='Enable email alerts')
    smtp_host: Optional[str] = Field(default=None, description='SMTP host')
    smtp_port: int = Field(default=587, description='SMTP port')
    smtp_user: Optional[str] = Field(default=None, description='SMTP user')
    smtp_password: Optional[str] = Field(default=None, description='SMTP password')


class StrategySettings(BaseModel):
    """Strategy configuration settings."""
    default_strategy: str = Field(default='trend_following', description='Default strategy')
    max_strategies: int = Field(default=10, description='Maximum concurrent strategies')
    auto_start: bool = Field(default=True, description='Auto-start strategies')
    
    class Config:
        extra = 'allow'


class Settings(BaseSettings):
    """
    Main settings class for the AI Quant Trading System.
    Loads configuration from environment variables and config files.
    """
    
    # System settings
    system_name: str = Field(default='AI Quant Trading System')
    system_version: str = Field(default='1.0.0')
    environment: str = Field(default='development')
    debug: bool = Field(default=False)
    
    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    monitor: MonitorSettings = Field(default_factory=MonitorSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    
    # Exchange configurations
    exchanges: Dict[str, ExchangeSettings] = Field(default_factory=dict)
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        env_nested_delimiter = '__'
        extra = 'ignore'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Settings':
        """Load settings from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'Settings':
        """Load settings from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save settings to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def to_json(self, path: str) -> None:
        """Save settings to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == 'development'


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure single instance.
    """
    return Settings()


# Global settings instance for easy import
settings = get_settings()


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load configuration from file or environment.
    
    Args:
        config_path: Optional path to config file (YAML or JSON)
    
    Returns:
        Settings instance
    """
    if config_path:
        path = Path(config_path)
        if path.suffix in ('.yaml', '.yml'):
            return Settings.from_yaml(config_path)
        elif path.suffix == '.json':
            return Settings.from_json(config_path)
    
    return Settings()


# Global settings instance for easy import
settings = get_settings()
