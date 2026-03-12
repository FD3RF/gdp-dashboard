"""
OKX 交易所同步管理器
====================

专门针对 OKX 的同步优化：
- 自动重连（处理 OKX 特有的断连问题）
- 数据一致性验证
- 心跳检测
- 速率限制智能处理
- 订单状态同步
- 余额对账
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

from execution.okx_adapter import OKXAdapter
from data.data_consistency import DataConsistencyValidator


class OKXSyncStatus(Enum):
    """OKX 同步状态"""
    DISCONNECTED = 'disconnected'
    CONNECTING = 'connecting'
    CONNECTED = 'connected'
    AUTHENTICATED = 'authenticated'
    SYNCING = 'syncing'
    SYNCED = 'synced'
    ERROR = 'error'
    RATE_LIMITED = 'rate_limited'


@dataclass
class OKXSyncState:
    """OKX 同步状态数据"""
    last_sync: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    retry_count: int = 0
    latency_ms: float = 0.0
    data_hash: str = ''
    records_synced: int = 0
    pending_updates: int = 0


@dataclass
class OKXConnectionState:
    """OKX 连接状态"""
    status: OKXSyncStatus = OKXSyncStatus.DISCONNECTED
    connected_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    last_ping: Optional[datetime] = None
    reconnect_attempts: int = 0
    latency_ms: float = 0.0
    error_count: int = 0
    rate_limit_reset: Optional[datetime] = None
    api_calls_remaining: int = 0


class OKXSyncManager:
    """
    OKX 交易所同步管理器
    
    特性：
    1. 智能重连 - OKX 特有的断连处理
    2. 速率限制 - OKX 速率限制自动处理
    3. 数据一致性 - 多重验证机制
    4. 订单同步 - 实时订单状态追踪
    5. 余额对账 - 定期余额验证
    6. 错误恢复 - 自动错误处理和恢复
    """
    
    # OKX 特定配置
    OKX_CONFIG = {
        'heartbeat_interval': 25,  # OKX 要求 30 秒内 ping
        'max_reconnect_attempts': 10,
        'reconnect_delay': 5,
        'reconnect_backoff': 1.5,  # 更平缓的退避
        'sync_interval': 2,
        'balance_sync_interval': 10,  # 余额同步间隔
        'order_sync_interval': 1,  # 订单同步间隔
        'cache_ttl': 30,
        'latency_threshold': 500,  # OKX 通常延迟较低
        'rate_limit_buffer': 0.8,  # 更保守的速率限制
    }
    
    # OKX 数据类型
    SYNC_TYPES = [
        'ticker', 'orderbook', 'klines', 'balance',
        'positions', 'orders', 'trades', 'funding_rate'
    ]
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = {**self.OKX_CONFIG, **(config or {})}
        self.logger = logging.getLogger("OKXSyncManager")
        
        # OKX 适配器
        self._adapter = OKXAdapter({
            'api_key': api_key,
            'api_secret': api_secret,
            'passphrase': passphrase,
            'testnet': config.get('testnet', False) if config else False,
            'account_type': config.get('account_type', 'spot') if config else 'spot',
        })
        
        # 连接状态
        self._connection = OKXConnectionState()
        
        # 同步状态
        self._sync_states: Dict[str, OKXSyncState] = {
            sync_type: OKXSyncState() for sync_type in self.SYNC_TYPES
        }
        
        # 数据缓存
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # 订单追踪
        self._local_orders: Dict[str, Dict] = {}  # 本地订单缓存
        self._pending_orders: Dict[str, Dict] = {}  # 待确认订单
        
        # 余额追踪
        self._last_balance: Dict[str, float] = {}
        self._balance_checksum: str = ''
        
        # 后台任务
        self._tasks: Dict[str, asyncio.Task] = {}
        
        # 回调
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # 数据一致性验证器
        self._validator = DataConsistencyValidator()
        
        # 统计
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency_ms': 0,
            'reconnects': 0,
            'syncs': 0,
            'errors_by_type': {},
        }
    
    async def start(self) -> bool:
        """启动同步管理器"""
        self.logger.info("Starting OKX sync manager...")
        
        # 初始化适配器
        if not await self._adapter.initialize():
            self.logger.error("Failed to initialize OKX adapter")
            return False
        
        await self._adapter.start()
        
        # 更新连接状态
        self._connection.status = OKXSyncStatus.CONNECTED
        self._connection.connected_at = datetime.now()
        
        # 启动后台任务
        self._tasks['heartbeat'] = asyncio.create_task(self._heartbeat_loop())
        self._tasks['balance_sync'] = asyncio.create_task(self._balance_sync_loop())
        self._tasks['order_sync'] = asyncio.create_task(self._order_sync_loop())
        self._tasks['data_sync'] = asyncio.create_task(self._data_sync_loop())
        self._tasks['health_check'] = asyncio.create_task(self._health_check_loop())
        
        self.logger.info("OKX sync manager started successfully")
        return True
    
    async def stop(self) -> bool:
        """停止同步管理器"""
        self.logger.info("Stopping OKX sync manager...")
        
        # 取消所有任务
        for name, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # 停止适配器
        await self._adapter.stop()
        
        self._connection.status = OKXSyncStatus.DISCONNECTED
        self.logger.info("OKX sync manager stopped")
        return True
    
    # ==================== 心跳检测 ====================
    
    async def _heartbeat_loop(self):
        """心跳检测循环"""
        while True:
            try:
                if self._connection.status in [
                    OKXSyncStatus.CONNECTED, 
                    OKXSyncStatus.SYNCED
                ]:
                    # 发送心跳（通过获取服务器时间）
                    start_time = time.time()
                    
                    try:
                        # 获取行情作为心跳
                        ticker = await self._adapter.get_ticker('BTC/USDT')
                        latency = (time.time() - start_time) * 1000
                        
                        self._connection.last_heartbeat = datetime.now()
                        self._connection.latency_ms = latency
                        
                        # 更新缓存
                        self._update_cache('ticker:BTC/USDT', ticker)
                        
                        # 检查延迟
                        if latency > self.config['latency_threshold']:
                            self.logger.warning(f"High latency: {latency:.0f}ms")
                        
                    except Exception as e:
                        self.logger.error(f"Heartbeat failed: {e}")
                        self._connection.error_count += 1
                        
                        # 检查是否需要重连
                        if self._connection.error_count >= 3:
                            await self._reconnect()
                
                await asyncio.sleep(self.config['heartbeat_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)
    
    async def _reconnect(self):
        """重连 OKX"""
        self._connection.status = OKXSyncStatus.CONNECTING
        self._connection.reconnect_attempts += 1
        self._stats['reconnects'] += 1
        
        # 计算退避延迟
        delay = min(
            self.config['reconnect_delay'] * (
                self.config['reconnect_backoff'] ** self._connection.reconnect_attempts
            ),
            60  # 最大 60 秒
        )
        
        self.logger.info(
            f"Reconnecting to OKX "
            f"(attempt {self._connection.reconnect_attempts}, delay {delay:.0f}s)"
        )
        
        await asyncio.sleep(delay)
        
        # 重新初始化适配器
        try:
            await self._adapter.stop()
            await asyncio.sleep(1)
            
            if await self._adapter.initialize():
                await self._adapter.start()
                
                self._connection.status = OKXSyncStatus.CONNECTED
                self._connection.error_count = 0
                self._connection.reconnect_attempts = 0
                self._connection.connected_at = datetime.now()
                
                self.logger.info("OKX reconnected successfully")
            else:
                self._connection.status = OKXSyncStatus.ERROR
                
                if self._connection.reconnect_attempts >= self.config['max_reconnect_attempts']:
                    self.logger.error("Max reconnect attempts reached")
                    
        except Exception as e:
            self.logger.error(f"Reconnect failed: {e}")
            self._connection.status = OKXSyncStatus.ERROR
    
    # ==================== 余额同步 ====================
    
    async def _balance_sync_loop(self):
        """余额同步循环"""
        while True:
            try:
                if self._connection.status not in [
                    OKXSyncStatus.CONNECTED, 
                    OKXSyncStatus.SYNCED
                ]:
                    await asyncio.sleep(5)
                    continue
                
                # 获取余额
                balance = await self._adapter.get_balance()
                
                if balance:
                    # 验证余额一致性
                    if self._last_balance:
                        result = self._validator.validate_balance_consistency(
                            self._last_balance,
                            balance['total'],
                            tolerance=0.0001  # 0.01% 容差
                        )
                        
                        if not result.is_valid:
                            self.logger.warning(
                                f"Balance inconsistency detected: {result.errors}"
                            )
                            # 通知回调
                            self._notify_callbacks('balance_mismatch', {
                                'local': self._last_balance,
                                'exchange': balance['total'],
                                'errors': result.errors
                            })
                    
                    # 更新本地余额
                    self._last_balance = balance['total']
                    self._balance_checksum = self._calculate_checksum(balance['total'])
                    
                    # 更新缓存
                    self._update_cache('balance', balance)
                    
                    # 更新同步状态
                    self._sync_states['balance'].last_sync = datetime.now()
                    self._sync_states['balance'].last_success = datetime.now()
                    self._sync_states['balance'].data_hash = self._balance_checksum
                    
                    # 通知回调
                    self._notify_callbacks('balance', balance)
                
                await asyncio.sleep(self.config['balance_sync_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Balance sync error: {e}")
                self._sync_states['balance'].last_error = str(e)
                await asyncio.sleep(5)
    
    # ==================== 订单同步 ====================
    
    async def _order_sync_loop(self):
        """订单同步循环"""
        while True:
            try:
                if self._connection.status not in [
                    OKXSyncStatus.CONNECTED, 
                    OKXSyncStatus.SYNCED
                ]:
                    await asyncio.sleep(5)
                    continue
                
                # 获取未完成订单
                open_orders = await self._adapter.get_open_orders()
                
                # 同步本地订单状态
                for order in open_orders:
                    order_id = order['order_id']
                    
                    if order_id in self._local_orders:
                        local_order = self._local_orders[order_id]
                        
                        # 验证订单一致性
                        result = self._validator.validate_order_consistency(
                            local_order,
                            order
                        )
                        
                        if not result.is_valid:
                            self.logger.warning(
                                f"Order inconsistency: {order_id}, errors: {result.errors}"
                            )
                            # 更新本地订单
                            self._local_orders[order_id] = order
                    
                    # 从待确认列表移除
                    if order_id in self._pending_orders:
                        del self._pending_orders[order_id]
                
                # 检查已完成的订单
                for order_id, order in list(self._local_orders.items()):
                    if order_id not in [o['order_id'] for o in open_orders]:
                        # 订单已完成或取消
                        try:
                            final_order = await self._adapter.get_order(
                                order_id, 
                                order['symbol']
                            )
                            
                            # 更新状态
                            self._local_orders[order_id] = final_order
                            
                            # 通知回调
                            self._notify_callbacks('order_completed', final_order)
                            
                        except Exception as e:
                            self.logger.error(f"Error fetching final order {order_id}: {e}")
                
                # 更新缓存
                self._update_cache('open_orders', open_orders)
                
                # 更新同步状态
                self._sync_states['orders'].last_sync = datetime.now()
                
                await asyncio.sleep(self.config['order_sync_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Order sync error: {e}")
                self._sync_states['orders'].last_error = str(e)
                await asyncio.sleep(5)
    
    # ==================== 数据同步 ====================
    
    async def _data_sync_loop(self):
        """数据同步循环"""
        while True:
            try:
                if self._connection.status not in [
                    OKXSyncStatus.CONNECTED, 
                    OKXSyncStatus.SYNCED
                ]:
                    await asyncio.sleep(5)
                    continue
                
                self._connection.status = OKXSyncStatus.SYNCING
                
                # 同步主要交易对数据
                symbols = self._get_active_symbols()
                
                for symbol in symbols:
                    try:
                        # 行情
                        ticker = await self._adapter.get_ticker(symbol)
                        self._update_cache(f'ticker:{symbol}', ticker)
                        
                        # 订单簿
                        orderbook = await self._adapter.get_orderbook(symbol, limit=5)
                        self._update_cache(f'orderbook:{symbol}', orderbook)
                        
                    except Exception as e:
                        self.logger.error(f"Error syncing data for {symbol}: {e}")
                
                # 合约持仓
                if self._adapter._account_type in ['swap', 'futures']:
                    positions = await self._adapter.get_positions()
                    self._update_cache('positions', positions)
                    self._notify_callbacks('positions', positions)
                
                self._connection.status = OKXSyncStatus.SYNCED
                self._stats['syncs'] += 1
                
                # 更新同步状态
                self._sync_states['ticker'].last_sync = datetime.now()
                
                await asyncio.sleep(self.config['sync_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Data sync error: {e}")
                self._connection.status = OKXSyncStatus.ERROR
                await asyncio.sleep(5)
    
    # ==================== 健康检查 ====================
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                # 检查连接状态
                if self._connection.last_heartbeat:
                    elapsed = (datetime.now() - self._connection.last_heartbeat).total_seconds()
                    
                    if elapsed > self.config['heartbeat_interval'] * 3:
                        self.logger.warning("Heartbeat timeout, triggering reconnect")
                        await self._reconnect()
                
                # 检查同步状态
                for sync_type, state in self._sync_states.items():
                    if state.last_sync:
                        elapsed = (datetime.now() - state.last_sync).total_seconds()
                        
                        # 如果某种类型数据超过预期时间未同步
                        expected_interval = self.config.get(
                            f'{sync_type}_interval', 
                            self.config['sync_interval']
                        )
                        
                        if elapsed > expected_interval * 5:
                            self.logger.warning(
                                f"Sync timeout for {sync_type}, last sync: {elapsed:.0f}s ago"
                            )
                
                # 检查错误率
                if self._stats['total_requests'] > 100:
                    error_rate = (
                        self._stats['failed_requests'] / 
                        self._stats['total_requests']
                    )
                    
                    if error_rate > 0.1:  # 10% 错误率
                        self.logger.warning(f"High error rate: {error_rate*100:.1f}%")
                
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    # ==================== 工具方法 ====================
    
    def _get_active_symbols(self) -> List[str]:
        """获取活跃交易对"""
        # 默认监控主流币种
        return [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOGE/USDT', 'ADA/USDT'
        ]
    
    def _update_cache(self, key: str, data: Any):
        """更新缓存"""
        self._cache[key] = data
        self._cache_timestamps[key] = datetime.now()
    
    def get_cached(self, key: str, max_age: Optional[int] = None) -> Optional[Any]:
        """获取缓存数据"""
        if key not in self._cache:
            return None
        
        if max_age is None:
            max_age = self.config['cache_ttl']
        
        timestamp = self._cache_timestamps.get(key)
        if timestamp:
            age = (datetime.now() - timestamp).total_seconds()
            if age > max_age:
                return None
        
        return self._cache.get(key)
    
    def _calculate_checksum(self, data: Dict) -> str:
        """计算数据校验和"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
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
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册回调"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    # ==================== 公共接口 ====================
    
    async def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """提交订单"""
        try:
            order = await self._adapter.create_order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            # 添加到本地追踪
            self._local_orders[order['order_id']] = order
            self._pending_orders[order['order_id']] = order
            
            return order
            
        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """取消订单"""
        try:
            result = await self._adapter.cancel_order(order_id, symbol)
            
            # 更新本地状态
            if order_id in self._local_orders:
                self._local_orders[order_id]['status'] = 'cancelled'
            
            return result
            
        except Exception as e:
            self.logger.error(f"Order cancellation failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """获取同步状态"""
        return {
            'connection': {
                'status': self._connection.status.value,
                'connected_at': self._connection.connected_at.isoformat() if self._connection.connected_at else None,
                'last_heartbeat': self._connection.last_heartbeat.isoformat() if self._connection.last_heartbeat else None,
                'latency_ms': self._connection.latency_ms,
                'error_count': self._connection.error_count,
                'reconnect_attempts': self._connection.reconnect_attempts,
            },
            'sync_states': {
                name: {
                    'last_sync': state.last_sync.isoformat() if state.last_sync else None,
                    'last_error': state.last_error,
                    'retry_count': state.retry_count,
                }
                for name, state in self._sync_states.items()
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
                ),
            },
            'cache_size': len(self._cache),
            'tracked_orders': len(self._local_orders),
            'pending_orders': len(self._pending_orders),
        }
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """获取行情"""
        return self.get_cached(f'ticker:{symbol}', max_age=10)
    
    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """获取订单簿"""
        return self.get_cached(f'orderbook:{symbol}', max_age=5)
    
    def get_balance(self) -> Optional[Dict]:
        """获取余额"""
        return self.get_cached('balance', max_age=60)
    
    def get_positions(self) -> Optional[List[Dict]]:
        """获取持仓"""
        return self.get_cached('positions', max_age=30)


# 导出
__all__ = ['OKXSyncManager', 'OKXSyncStatus', 'OKXSyncState', 'OKXConnectionState']
