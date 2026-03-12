# infrastructure/cache.py
"""
模块 2: Redis缓存
==================
毫秒级存取实时 Tick 与订单簿快照
"""

import json
import time
from typing import Any, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis缓存管理器
    
    功能：
    - 毫秒级数据存取
    - 支持TTL过期
    - 连接池管理
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self._client = None
        self._enabled = False
        self._local_cache: Dict[str, Any] = {}  # 本地缓存降级
        self._timestamps: Dict[str, float] = {}
        
    def connect(self) -> bool:
        """连接Redis"""
        try:
            import redis
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_timeout=1,
                socket_connect_timeout=1
            )
            self._client.ping()
            self._enabled = True
            logger.info(f"✓ Redis连接成功: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.warning(f"Redis连接失败，使用本地缓存: {e}")
            self._enabled = False
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            if self._enabled and self._client:
                value = self._client.get(key)
                if value:
                    return json.loads(value)
            else:
                # 本地缓存
                if key in self._local_cache:
                    # 检查TTL
                    if key in self._timestamps:
                        if time.time() - self._timestamps[key] < 60:  # 60秒TTL
                            return self._local_cache[key]
                return None
        except Exception as e:
            logger.debug(f"缓存读取失败: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: int = 60) -> bool:
        """设置缓存"""
        try:
            if self._enabled and self._client:
                self._client.setex(key, ttl, json.dumps(value, default=str))
            else:
                # 本地缓存
                self._local_cache[key] = value
                self._timestamps[key] = time.time()
            return True
        except Exception as e:
            logger.debug(f"缓存写入失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            if self._enabled and self._client:
                self._client.delete(key)
            else:
                self._local_cache.pop(key, None)
                self._timestamps.pop(key, None)
            return True
        except Exception as e:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "enabled": self._enabled,
            "local_cache_size": len(self._local_cache),
            "backend": "redis" if self._enabled else "local"
        }


# 全局缓存实例
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """获取全局缓存实例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
        _cache_instance.connect()
    return _cache_instance


# 装饰器：缓存结果
def cached(ttl: int = 60, key_prefix: str = ""):
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_cache()
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
