# infrastructure/__init__.py
"""
基础设施层模块
Layer 1: 系统基础设施层
"""

from .cache import RedisCache
from .message_queue import KafkaProducer

__all__ = ['RedisCache', 'KafkaProducer']
