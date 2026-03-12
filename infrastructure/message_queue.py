# infrastructure/message_queue.py
"""
模块 3: Kafka消息队列
=====================
承载数据吞吐洪峰，确保行情不丢包、不延迟
"""

import json
import time
import queue
from typing import Any, Optional, Callable, Dict
from datetime import datetime
import logging
import threading

logger = logging.getLogger(__name__)


class KafkaProducer:
    """
    Kafka消息生产者
    
    功能：
    - 高吞吐消息发送
    - 异步批量发送
    - 本地队列降级
    """
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self._producer = None
        self._enabled = False
        self._local_queue = queue.Queue(maxsize=10000)
        self._callbacks: Dict[str, list] = {}
        
    def connect(self) -> bool:
        """连接Kafka"""
        try:
            from kafka import KafkaProducer as KP
            self._producer = KP(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                acks='all',
                retries=3,
                max_block_ms=1000
            )
            self._enabled = True
            logger.info(f"✓ Kafka连接成功: {self.bootstrap_servers}")
            return True
        except ImportError:
            logger.warning("kafka-python未安装，使用本地队列")
            self._enabled = False
            return False
        except Exception as e:
            logger.warning(f"Kafka连接失败，使用本地队列: {e}")
            self._enabled = False
            return False
    
    def send(self, topic: str, value: Any, key: Optional[str] = None) -> bool:
        """发送消息"""
        try:
            message = {
                "topic": topic,
                "key": key,
                "value": value,
                "timestamp": datetime.now().isoformat()
            }
            
            if self._enabled and self._producer:
                future = self._producer.send(topic, value=value, key=key)
                future.get(timeout=1)
            else:
                # 本地队列
                self._local_queue.put(message, block=False)
            
            # 触发回调
            self._trigger_callback(topic, value)
            
            return True
        except Exception as e:
            logger.debug(f"消息发送失败: {e}")
            return False
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        """订阅主题回调"""
        if topic not in self._callbacks:
            self._callbacks[topic] = []
        self._callbacks[topic].append(callback)
    
    def _trigger_callback(self, topic: str, value: Any) -> None:
        """触发回调"""
        if topic in self._callbacks:
            for callback in self._callbacks[topic]:
                try:
                    callback(value)
                except Exception as e:
                    logger.debug(f"回调执行失败: {e}")
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        return self._local_queue.qsize()


class DataStream:
    """
    数据流管理器
    
    功能：
    - 统一数据流管理
    - 支持多数据源
    - 数据分发
    """
    
    def __init__(self):
        self._queues: Dict[str, queue.Queue] = {}
        self._running = False
        self._threads: list = []
    
    def create_queue(self, name: str, maxsize: int = 1000) -> queue.Queue:
        """创建数据队列"""
        if name not in self._queues:
            self._queues[name] = queue.Queue(maxsize=maxsize)
        return self._queues[name]
    
    def get_queue(self, name: str) -> Optional[queue.Queue]:
        """获取队列"""
        return self._queues.get(name)
    
    def put(self, queue_name: str, data: Any, block: bool = False) -> bool:
        """放入数据"""
        q = self._queues.get(queue_name)
        if q:
            try:
                q.put(data, block=block)
                return True
            except queue.Full:
                return False
        return False
    
    def get(self, queue_name: str, block: bool = False, timeout: float = 1.0) -> Optional[Any]:
        """获取数据"""
        q = self._queues.get(queue_name)
        if q:
            try:
                return q.get(block=block, timeout=timeout)
            except queue.Empty:
                return None
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计"""
        return {name: q.qsize() for name, q in self._queues.items()}


# 全局数据流实例
_data_stream: Optional[DataStream] = None


def get_data_stream() -> DataStream:
    """获取全局数据流实例"""
    global _data_stream
    if _data_stream is None:
        _data_stream = DataStream()
    return _data_stream
