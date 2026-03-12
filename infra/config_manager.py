"""
Configuration Manager for dynamic configuration management.
"""

import asyncio
import json
import logging
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
import hashlib
from core.base import BaseModule
from core.exceptions import ConfigurationException


@dataclass
class ConfigEntry:
    """Configuration entry with metadata."""
    key: str
    value: Any
    scope: str = 'global'
    version: int = 1
    updated_at: datetime = field(default_factory=datetime.now)
    updated_by: str = 'system'
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigManager(BaseModule):
    """
    Dynamic configuration management with hot-reload support.
    Supports file-based, environment-based, and runtime configuration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('config_manager', config)
        self._config: Dict[str, ConfigEntry] = {}
        self._watchers: Dict[str, List[Callable]] = {}
        self._file_paths: Dict[str, Path] = {}
        self._env_prefix = self.config.get('env_prefix', 'QUANT_')
        self._auto_reload = self.config.get('auto_reload', True)
        self._reload_task: Optional[asyncio.Task] = None
        self._file_hashes: Dict[str, str] = {}
    
    async def initialize(self) -> bool:
        """Initialize configuration manager."""
        self.logger.info("Initializing config manager...")
        
        # Load from environment
        self._load_from_env()
        
        # Load from default config file if exists
        config_file = self.config.get('config_file')
        if config_file:
            await self.load_from_file(config_file)
        
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start configuration manager."""
        self._running = True
        self._start_time = datetime.now()
        
        if self._auto_reload:
            self._reload_task = asyncio.create_task(self._watch_files())
        
        return True
    
    async def stop(self) -> bool:
        """Stop configuration manager."""
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
        
        self._running = False
        return True
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                config_key = key[len(self._env_prefix):].lower()
                self.set(config_key, value, scope='env')
    
    def _compute_file_hash(self, path: Path) -> str:
        """Compute file hash for change detection."""
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    async def _watch_files(self) -> None:
        """Watch configuration files for changes."""
        while self._running:
            try:
                for name, path in self._file_paths.items():
                    if path.exists():
                        current_hash = self._compute_file_hash(path)
                        if current_hash != self._file_hashes.get(name):
                            self.logger.info(f"Configuration file changed: {path}")
                            await self.load_from_file(str(path), name)
                            self._file_hashes[name] = current_hash
                
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error watching files: {e}")
                await asyncio.sleep(5)
    
    async def load_from_file(self, file_path: str, name: str = 'default') -> bool:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file (YAML or JSON)
            name: Configuration name
        
        Returns:
            True if successful
        """
        path = Path(file_path)
        if not path.exists():
            raise ConfigurationException(f"Config file not found: {file_path}")
        
        try:
            with open(path, 'r') as f:
                if path.suffix in ('.yaml', '.yml'):
                    data = yaml.safe_load(f)
                elif path.suffix == '.json':
                    data = json.load(f)
                else:
                    raise ConfigurationException(f"Unsupported config format: {path.suffix}")
            
            # Flatten nested config
            flat_config = self._flatten_dict(data)
            
            # Update configuration
            for key, value in flat_config.items():
                self.set(key, value, scope=name)
            
            self._file_paths[name] = path
            self._file_hashes[name] = self._compute_file_hash(path)
            
            self.logger.info(f"Loaded configuration from: {file_path}")
            return True
            
        except Exception as e:
            raise ConfigurationException(f"Failed to load config: {e}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _unflatten_dict(self, d: Dict, sep: str = '.') -> Dict:
        """Unflatten dictionary."""
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            d_ref = result
            for part in parts[:-1]:
                if part not in d_ref:
                    d_ref[part] = {}
                d_ref = d_ref[part]
            d_ref[parts[-1]] = value
        return result
    
    def get(
        self,
        key: str,
        default: Any = None,
        scope: Optional[str] = None
    ) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if not found
            scope: Optional scope to search in
        
        Returns:
            Configuration value
        """
        if scope:
            entry = self._config.get(f"{scope}.{key}")
            if entry:
                return entry.value
        
        entry = self._config.get(key)
        if entry:
            return entry.value
        
        return default
    
    def set(
        self,
        key: str,
        value: Any,
        scope: str = 'runtime',
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            scope: Configuration scope
            metadata: Optional metadata
        """
        old_value = self.get(key)
        
        entry = ConfigEntry(
            key=key,
            value=value,
            scope=scope,
            metadata=metadata or {}
        )
        
        self._config[key] = entry
        
        # Notify watchers
        self._notify_watchers(key, old_value, value)
        
        self.logger.debug(f"Set config: {key} = {value}")
    
    def delete(self, key: str) -> bool:
        """Delete configuration key."""
        if key in self._config:
            old_value = self._config[key].value
            del self._config[key]
            self._notify_watchers(key, old_value, None)
            return True
        return False
    
    def watch(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """
        Register a callback for configuration changes.
        
        Args:
            key: Configuration key to watch (use '*' for all)
            callback: Callback function(key, old_value, new_value)
        """
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)
    
    def unwatch(self, key: str, callback: Callable) -> bool:
        """Remove a configuration watcher."""
        if key in self._watchers and callback in self._watchers[key]:
            self._watchers[key].remove(callback)
            return True
        return False
    
    def _notify_watchers(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify watchers of configuration change."""
        # Notify specific key watchers
        if key in self._watchers:
            for callback in self._watchers[key]:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    self.logger.error(f"Config watcher error: {e}")
        
        # Notify wildcard watchers
        if '*' in self._watchers:
            for callback in self._watchers['*']:
                try:
                    callback(key, old_value, new_value)
                except Exception as e:
                    self.logger.error(f"Config watcher error: {e}")
    
    def get_all(self, scope: Optional[str] = None) -> Dict[str, Any]:
        """Get all configuration values."""
        if scope:
            return {
                k: v.value for k, v in self._config.items()
                if v.scope == scope
            }
        return {k: v.value for k, v in self._config.items()}
    
    def get_scopes(self) -> Set[str]:
        """Get all configuration scopes."""
        return {entry.scope for entry in self._config.values()}
    
    def save_to_file(self, file_path: str, scope: Optional[str] = None) -> bool:
        """Save configuration to file."""
        path = Path(file_path)
        config_data = self.get_all(scope)
        
        try:
            with open(path, 'w') as f:
                if path.suffix in ('.yaml', '.yml'):
                    yaml.dump(config_data, f, default_flow_style=False)
                elif path.suffix == '.json':
                    json.dump(config_data, f, indent=2)
                else:
                    raise ConfigurationException(f"Unsupported format: {path.suffix}")
            
            self.logger.info(f"Saved configuration to: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get configuration status."""
        return {
            **super().get_status(),
            'config_count': len(self._config),
            'scopes': list(self.get_scopes()),
            'watched_keys': list(self._watchers.keys()),
            'file_paths': {k: str(v) for k, v in self._file_paths.items()}
        }
