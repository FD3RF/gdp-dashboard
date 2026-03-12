"""
Logging System for comprehensive logging and log management.
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from queue import Queue
from core.base import BaseModule


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line: int
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'logger': self.logger_name,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'line': self.line,
            'extra': self.extra
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line=record.lineno,
            extra=getattr(record, 'extra', {})
        )
        return entry.to_json()


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class LogHandler:
    """Base log handler interface."""
    
    def handle(self, entry: LogEntry) -> None:
        raise NotImplementedError


class CallbackLogHandler(LogHandler):
    """Log handler that calls a callback function."""
    
    def __init__(self, callback: Callable[[LogEntry], None]):
        self.callback = callback
    
    def handle(self, entry: LogEntry) -> None:
        try:
            self.callback(entry)
        except Exception:
            pass


class LoggingSystem(BaseModule):
    """
    Comprehensive logging system with multiple handlers and structured output.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('logging_system', config)
        self._log_handlers: List[LogHandler] = []
        self._loggers: Dict[str, logging.Logger] = {}
        self._log_queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self._log_level = LogLevel[self.config.get('level', 'INFO').upper()]
        self._log_dir = Path(self.config.get('log_dir', 'logs'))
        self._max_file_size = self.config.get('max_file_size', 10 * 1024 * 1024)  # 10MB
        self._backup_count = self.config.get('backup_count', 5)
        self._console_output = self.config.get('console_output', True)
        self._json_format = self.config.get('json_format', False)
    
    async def initialize(self) -> bool:
        """Initialize logging system."""
        self.logger.info("Initializing logging system...")
        
        # Create log directory
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        self._configure_root_logger()
        
        # Start worker thread for custom handlers
        self._start_worker()
        
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start logging system."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop logging system."""
        self._stop_event.set()
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        
        self._running = False
        return True
    
    def _configure_root_logger(self) -> None:
        """Configure root logger with handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self._log_level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self._console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._log_level.value)
            
            if self._json_format:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(ColoredFormatter(
                    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
                ))
            
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = self._log_dir / 'quant_system.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self._max_file_size,
            backupCount=self._backup_count
        )
        file_handler.setLevel(self._log_level.value)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_file = self._log_dir / 'errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=self._max_file_size,
            backupCount=self._backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)
    
    def _start_worker(self) -> None:
        """Start worker thread for custom handlers."""
        def worker():
            while not self._stop_event.is_set():
                try:
                    entry = self._log_queue.get(timeout=1)
                    for handler in self._log_handlers:
                        handler.handle(entry)
                except Exception:
                    pass
        
        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger by name."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self._log_level.value)
            self._loggers[name] = logger
        return self._loggers[name]
    
    def add_handler(self, handler: LogHandler) -> None:
        """Add a custom log handler."""
        self._log_handlers.append(handler)
    
    def remove_handler(self, handler: LogHandler) -> None:
        """Remove a log handler."""
        if handler in self._log_handlers:
            self._log_handlers.remove(handler)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = 'quant',
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a message."""
        logger = self.get_logger(logger_name)
        logger.log(level.value, message, extra={'extra': extra or {}})
    
    def debug(self, message: str, logger_name: str = 'quant', **kwargs) -> None:
        self.log(LogLevel.DEBUG, message, logger_name, kwargs)
    
    def info(self, message: str, logger_name: str = 'quant', **kwargs) -> None:
        self.log(LogLevel.INFO, message, logger_name, kwargs)
    
    def warning(self, message: str, logger_name: str = 'quant', **kwargs) -> None:
        self.log(LogLevel.WARNING, message, logger_name, kwargs)
    
    def error(self, message: str, logger_name: str = 'quant', **kwargs) -> None:
        self.log(LogLevel.ERROR, message, logger_name, kwargs)
    
    def critical(self, message: str, logger_name: str = 'quant', **kwargs) -> None:
        self.log(LogLevel.CRITICAL, message, logger_name, kwargs)
    
    def set_level(self, level: LogLevel) -> None:
        """Set global log level."""
        self._log_level = level
        logging.getLogger().setLevel(level.value)
        for logger in self._loggers.values():
            logger.setLevel(level.value)
    
    def get_recent_logs(self, count: int = 100, level: Optional[LogLevel] = None) -> List[str]:
        """Get recent log entries from log file."""
        log_file = self._log_dir / 'quant_system.log'
        if not log_file.exists():
            return []
        
        entries = []
        with open(log_file, 'r') as f:
            lines = f.readlines()[-count:]
            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    if level is None or entry.get('level') == level.name:
                        entries.append(line.strip())
                except json.JSONDecodeError:
                    entries.append(line.strip())
        
        return entries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        log_file = self._log_dir / 'quant_system.log'
        file_size = log_file.stat().st_size if log_file.exists() else 0
        
        return {
            'log_level': self._log_level.name,
            'log_dir': str(self._log_dir),
            'log_file_size': file_size,
            'logger_count': len(self._loggers),
            'custom_handlers': len(self._log_handlers)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'healthy': self._initialized,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'stats': self.get_stats()
        }
