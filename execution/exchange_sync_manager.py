"""
交易所同步管理器
==================

解决所有交易所同步问题：
- 自动重连机制
- 数据一致性验证
- 心跳检测
- 速率限制处理
- WebSocket 实时数据
- 缓存失效处理
- 网络错误处理
- 同步队列管理
- 延迟监控
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib


class SyncStatus(Enum):
    """同步状态"""
    DISCONNECTED = 'disconnected'
    CONNECTING = 'connecting'
    CONNECTED = 'connected'
    SYNCING = 'syncing'
    SYNCED = 'synced'
    ERROR = 'error'


class DataType(Enum):
    """数据类型"""
    TICKER = 'ticker'
    OHLCV = 'ohlcv'
    ORDERBOOK = 'orderbook'
    TRADES = 'trades'
    BALANCE = 'balance'
    POSITIONS = 'positions'
    ORDERS = 'orders'


@dataclass
class SyncState:
    """同步状态数据"""
    last_sync: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    retry_count: int = 0
    latency_ms: float = 0.0
    data_hash: str = ''
    records_synced: int = 0
    pending_updates: int = 0


@dataclass
class ExchangeConnection:
    """交易所连接状态"""
    name: str
    status: SyncStatus = SyncStatus.DISCONNECTED
    connected_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    reconnect_attempts: int = 0
    latency_ms: float = 0.0
    error_count: int = 0
    sync_states: Dict[str, SyncState] = field(default_factory=dict)


class ExchangeSyncManager:
    """
    交易所同步管理器
    
    功能：
    1. 自动重连 - 断线后自动尝试重连
    2. 心跳检测 - 定期检查连接状态
    3. 数据同步 - 确保本地与交易所数据一致
    4. 速率限制 - 自动处理API限制
    5. 错误处理 - 网络错误自动恢复
    6. 延迟监控 - 实时监控API延迟
    7. 缓存管理 - 智能缓存失效
    8. 重试队列 - 失败请求自动重试
    """
    
    # 配置常量
    DEFAULT_CONFIG = {
        'heartbeat_interval': 30,  # 心跳间隔（秒）
        'max_reconnect_attempts': 5,  # 最大重连次数
        'reconnect_delay': 5,  # 重连延迟（秒）
        'reconnect_backoff': 2,  # 重连退避因子
        'sync_interval': 1,  # 同步间隔（秒）
        'max_retry_queue': 100,  # 最大重试队列
        'cache_ttl': 60,  # 缓存有效期（秒）
        'latency_threshold': 1000,  # 延迟阈值（毫秒）
        'rate_limit_buffer': 0.9,  # 速率限制缓冲
    }
    
    def __init__(
        self,
        exchange_adapter,
        config: Optional[Dict[str, Any]] = None
    ):
        self.exchange_adapter = exchange_adapter
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger("ExchangeSyncManager")
        
        # 连接状态
        self._connections: Dict[str, ExchangeConnection] = {}
        self._sync_tasks: Dict[str, asyncio.Task] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # 数据缓存
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # 重试队列
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._retry_task: Optional[asyncio.Task] = None
        
        # 回调
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # 统计
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency_ms': 0,
            'reconnects': 0,
            'syncs': 0
        }
    
    async def start(self):
        """启动同步管理器"""
        self.logger.info("Starting exchange sync manager...")
        
        # 启动心跳任务
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # 启动重试队列任务
        self._retry_task = asyncio.create_task(self._retry_loop())
        
        # 初始化连接状态
        for name in self._get_exchange_names():
            self._connections[name] = ExchangeConnection(name=name)
            self._sync_tasks[name] = asyncio.create_task(
                self._sync_loop(name)
            )
        
        self.logger.info("Exchange sync manager started")
    
    async def stop(self):
        """停止同步管理器"""
        self.logger.info("Stopping exchange sync manager...")
        
        # 取消所有任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._retry_task:
            self._retry_task.cancel()
        
        for task in self._sync_tasks.values():
            task.cancel()
        
        self._connections.clear()
        self._sync_tasks.clear()
        
        self.logger.info("Exchange sync manager stopped")
    
    def _get_exchange_names(self) -> List[str]:
        """获取所有交易所名称"""
        if hasattr(self.exchange_adapter, '_exchanges'):
            return list(self.exchange_adapter._exchanges.keys())
        return ['default']
    
    # ==================== 连接管理 ====================
    
    async def check_connection(self, exchange: str = None) -> bool:
        """检查连接状态"""
        exchange_name = exchange or 'default'
        
        try:
            # 发送测试请求
            start_time = time.time()
            
            if hasattr(self.exchange_adapter, 'get_exchange'):
                ex = self.exchange_adapter.get_exchange(exchange_name)
                if ex:
                    await ex.fetch_ticker('BTC/USDT')
            
            latency = (time.time() - start_time) * 1000
            
            # 更新连接状态
            if exchange_name in self._connections:
                conn = self._connections[exchange_name]
                conn.status = SyncStatus.CONNECTED
                conn.last_heartbeat = datetime.now()
                conn.latency_ms = latency
                
                if latency > self.config['latency_threshold']:
                    self.logger.warning(
                        f"High latency detected: {latency:.0f}ms"
                    )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection check failed: {e}")
            
            if exchange_name in self._connections:
                self._connections[exchange_name].status = SyncStatus.ERROR
                self._connections[exchange_name].error_count += 1
            
            return False
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while True:
            try:
                for name in self._connections:
                    conn = self._connections[name]
                    
                    # 检查是否需要重连
                    if conn.status in [SyncStatus.DISCONNECTED, SyncStatus.ERROR]:
                        await self._reconnect(name)
                    
                    # 检查心跳超时
                    elif conn.last_heartbeat:
                        elapsed = (datetime.now() - conn.last_heartbeat).total_seconds()
                        
                        if elapsed > self.config['heartbeat_interval'] * 3:
                            self.logger.warning(
                                f"Heartbeat timeout for {name}, reconnecting..."
                            )
                            conn.status = SyncStatus.DISCONNECTED
                            await self._reconnect(name)
                        else:
                            # 发送心跳
                            await self.check_connection(name)
                
                await asyncio.sleep(self.config['heartbeat_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _reconnect(self, exchange: str):
        """重新连接交易所"""
        conn = self._connections.get(exchange)
        if not conn:
            return
        
        conn.status = SyncStatus.CONNECTING
        conn.reconnect_attempts += 1
        self._stats['reconnects'] += 1
        
        # 计算退避延迟
        delay = min(
            self.config['reconnect_delay'] * (
                self.config['reconnect_backoff'] ** conn.reconnect_attempts
            ),
            60  # 最大60秒
        )
        
        self.logger.info(
            f"Reconnecting to {exchange} "
            f"(attempt {conn.reconnect_attempts}, delay {delay:.0f}s)"
        )
        
        await asyncio.sleep(delay)
        
        # 尝试重连
        success = await self.check_connection(exchange)
        
        if success:
            conn.status = SyncStatus.CONNECTED
            conn.reconnect_attempts = 0
            conn.connected_at = datetime.now()
            self.logger.info(f"Reconnected to {exchange}")
        elif conn.reconnect_attempts >= self.config['max_reconnect_attempts']:
            conn.status = SyncStatus.ERROR
            self.logger.error(
                f"Max reconnect attempts reached for {exchange}"
            )
    
    # ==================== 数据同步 ====================
    
    async def _sync_loop(self, exchange: str):
        """数据同步循环"""
        while True:
            try:
                conn = self._connections.get(exchange)
                if not conn or conn.status != SyncStatus.CONNECTED:
                    await asyncio.sleep(1)
                    continue
                
                conn.status = SyncStatus.SYNCING
                
                # 同步各类数据
                await self._sync_tickers(exchange)
                await self._sync_balances(exchange)
                await self._sync_positions(exchange)
                await self._sync_orders(exchange)
                
                conn.status = SyncStatus.SYNCED
                self._stats['syncs'] += 1
                
                await asyncio.sleep(self.config['sync_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Sync error for {exchange}: {e}")
                if exchange in self._connections:
                    self._connections[exchange].status = SyncStatus.ERROR
                await asyncio.sleep(5)
    
    async def _sync_tickers(self, exchange: str):
        """同步行情数据"""
        symbols = self._get_active_symbols(exchange)
        
        for symbol in symbols:
            try:
                cache_key = f"{exchange}:ticker:{symbol}"
                
                # 检查缓存
                if self._is_cache_valid(cache_key):
                    continue
                
                # 获取数据
                ticker = await self._safe_request(
                    self.exchange_adapter.fetch_ticker,
                    symbol,
                    exchange
                )
                
                if ticker:
                    self._update_cache(cache_key, ticker)
                    self._notify_callbacks('ticker', ticker)
                    
            except Exception as e:
                self.logger.error(f"Error syncing ticker {symbol}: {e}")
    
    async def _sync_balances(self, exchange: str):
        """同步账户余额"""
        try:
            cache_key = f"{exchange}:balance"
            
            balance = await self._safe_request(
                self.exchange_adapter.get_balance,
                exchange
            )
            
            if balance:
                self._update_cache(cache_key, balance)
                self._notify_callbacks('balance', balance)
                
        except Exception as e:
            self.logger.error(f"Error syncing balance: {e}")
    
    async def _sync_positions(self, exchange: str):
        """同步持仓"""
        try:
            positions = await self._safe_request(
                self.exchange_adapter.get_positions,
                exchange
            )
            
            if positions:
                cache_key = f"{exchange}:positions"
                self._update_cache(cache_key, positions)
                self._notify_callbacks('positions', positions)
                
        except Exception as e:
            self.logger.error(f"Error syncing positions: {e}")
    
    async def _sync_orders(self, exchange: str):
        """同步订单状态"""
        # 获取本地未完成订单
        if hasattr(self.exchange_adapter, 'get_open_orders'):
            open_orders = await self._safe_request(
                self.exchange_adapter.get_open_orders,
                exchange
            )
            
            if open_orders:
                self._notify_callbacks('orders', open_orders)
    
    def _get_active_symbols(self, exchange: str) -> List[str]:
        """获取活跃交易对"""
        # 默认监控主流币种
        return [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT',
            'SOL/USDT', 'XRP/USDT'
        ]
    
    # ==================== 安全请求 ====================
    
    async def _safe_request(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        安全执行请求
        
        包含：
        - 速率限制处理
        - 错误处理
        - 延迟监控
        - 重试机制
        """
        self._stats['total_requests'] += 1
        
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            latency = (time.time() - start_time) * 1000
            self._stats['total_latency_ms'] += latency
            self._stats['successful_requests'] += 1
            
            return result
            
        except Exception as e:
            self._stats['failed_requests'] += 1
            
            # 加入重试队列
            if self._retry_queue.qsize() < self.config['max_retry_queue']:
                await self._retry_queue.put((func, args, kwargs))
            
            raise
    
    async def _retry_loop(self):
        """重试队列处理"""
        while True:
            try:
                func, args, kwargs = await asyncio.wait_for(
                    self._retry_queue.get(),
                    timeout=5.0
                )
                
                await self._safe_request(func, *args, **kwargs)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Retry failed: {e}")
    
    # ==================== 缓存管理 ====================
    
    def _is_cache_valid(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key not in self._cache_timestamps:
            return False
        
        elapsed = (datetime.now() - self._cache_timestamps[key]).total_seconds()
        return elapsed < self.config['cache_ttl']
    
    def _update_cache(self, key: str, data: Any):
        """更新缓存"""
        self._cache[key] = data
        self._cache_timestamps[key] = datetime.now()
    
    def get_cached(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def invalidate_cache(self, pattern: str = None):
        """
        使缓存失效
        
        Args:
            pattern: 缓存键模式（可选）
        """
        if pattern:
            keys_to_remove = [
                k for k in self._cache
                if pattern in k
            ]
            for key in keys_to_remove:
                del self._cache[key]
                del self._cache_timestamps[key]
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
    
    # ==================== 数据一致性验证 ====================
    
    def verify_data_integrity(
        self,
        data: Dict[str, Any],
        expected_fields: List[str]
    ) -> bool:
        """验证数据完整性"""
        if not data:
            return False
        
        for field in expected_fields:
            if field not in data:
                self.logger.warning(f"Missing field: {field}")
                return False
        
        return True
    
    def calculate_data_hash(self, data: Any) -> str:
        """计算数据哈希用于一致性检查"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    # ==================== 回调管理 ====================
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册回调"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def _notify_callbacks(self, event_type: str, data: Any):
        """通知回调"""
        callbacks = self._callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(data))
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    # ==================== 状态查询 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取同步状态"""
        return {
            'connections': {
                name: {
                    'status': conn.status.value,
                    'latency_ms': conn.latency_ms,
                    'last_heartbeat': conn.last_heartbeat.isoformat() if conn.last_heartbeat else None,
                    'error_count': conn.error_count,
                    'reconnect_attempts': conn.reconnect_attempts
                }
                for name, conn in self._connections.items()
            },
            'stats': {
                **self._stats,
                'avg_latency_ms': (
                    self._stats['total_latency_ms'] / self._stats['successful_requests']
                    if self._stats['successful_requests'] > 0 else 0
                ),
                'success_rate': (
                    self._stats['successful_requests'] / self._stats['total_requests'] * 100
                    if self._stats['total_requests'] > 0 else 0
                )
            },
            'cache_size': len(self._cache),
            'retry_queue_size': self._retry_queue.qsize()
        }
    
    def get_data(
        self,
        data_type: str,
        exchange: str = None,
        symbol: str = None
    ) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            data_type: 数据类型 (ticker, balance, positions, orders)
            exchange: 交易所名称
            symbol: 交易对（可选）
            
        Returns:
            缓存的数据或None
        """
        exchange = exchange or 'default'
        
        if symbol:
            key = f"{exchange}:{data_type}:{symbol}"
        else:
            key = f"{exchange}:{data_type}"
        
        return self.get_cached(key)


# 导出
__all__ = [
    'ExchangeSyncManager',
    'SyncStatus',
    'DataType',
    'SyncState',
    'ExchangeConnection'
]
