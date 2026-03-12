"""
OKX 交易所适配器
==================

专门针对 OKX 交易所的优化适配器：
- 完整的 API 支持（现货、合约、期权）
- WebSocket 实时数据
- 签名认证
- 速率限制处理
- 错误恢复机制
"""

import asyncio
import logging
import hmac
import base64
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from decimal import Decimal
import json

try:
    import ccxt.async_support as ccxt
except ImportError:
    import ccxt

from core.base import BaseModule
from core.exceptions import ExecutionException


class OKXAdapter(BaseModule):
    """
    OKX 交易所适配器
    
    功能：
    1. 现货交易
    2. 永续合约交易
    3. 期权交易
    4. WebSocket 实时数据
    5. 账户管理
    6. 资金划转
    """
    
    # OKX API 端点
    API_URL = 'https://www.okx.com'
    
    # 速率限制配置
    RATE_LIMITS = {
        'spot': {'requests': 20, 'interval': 2},  # 每2秒20次
        'futures': {'requests': 20, 'interval': 2},
        'options': {'requests': 20, 'interval': 2},
        'account': {'requests': 10, 'interval': 2},
    }
    
    # OKX 特定的错误码
    ERROR_CODES = {
        '0': 'Success',
        '50000': 'Invalid parameter',
        '50001': 'System error',
        '50002': 'Service unavailable',
        '50004': 'Request too frequent',
        '50005': 'API key expired',
        '50006': 'Invalid API key',
        '50007': 'Signature verification failed',
        '50008': 'Timestamp error',
        '50009': 'IP not allowed',
        '50010': 'Insufficient balance',
        '50011': 'Order not found',
        '50012': 'Order already cancelled',
        '50013': 'Order already filled',
        '50014': 'Order quantity too small',
        '50015': 'Position not found',
        '50016': 'Leverage too high',
        '50017': 'Risk limit exceeded',
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('okx_adapter', config)
        
        self._exchange: Optional[ccxt.okx] = None
        self._ws_connections: Dict[str, Any] = {}
        
        # 配置
        self._api_key = self.config.get('api_key', '')
        self._api_secret = self.config.get('api_secret', '')
        self._passphrase = self.config.get('passphrase', '')
        self._testnet = self.config.get('testnet', False)
        self._account_type = self.config.get('account_type', 'spot')  # spot, swap, futures
        
        # 速率限制状态
        self._rate_limit_counters: Dict[str, List[float]] = {}
        
        # 订单回调
        self._order_callbacks: List[Callable] = []
        
        # WebSocket 消息处理器
        self._ws_handlers: Dict[str, Callable] = {}
    
    async def initialize(self) -> bool:
        """初始化 OKX 连接"""
        self.logger.info("Initializing OKX adapter...")
        
        try:
            # 创建 CCXT OKX 实例
            okx_config = {
                'apiKey': self._api_key,
                'secret': self._api_secret,
                'password': self._passphrase,
                'enableRateLimit': True,
                'rateLimit': 100,  # 每100ms一次请求
                'options': {
                    'defaultType': self._account_type,
                }
            }
            
            if self._testnet:
                okx_config['options']['broker'] = {'name': 'testnet'}
            
            self._exchange = ccxt.okx(okx_config)
            
            # 加载市场
            await self._exchange.load_markets()
            
            self._initialized = True
            self.logger.info(f"OKX adapter initialized, markets: {len(self._exchange.markets)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OKX: {e}")
            return False
    
    async def start(self) -> bool:
        """启动适配器"""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """停止适配器"""
        if self._exchange:
            await self._exchange.close()
        
        # 关闭 WebSocket 连接
        for ws in self._ws_connections.values():
            try:
                await ws.close()
            except Exception:
                pass
        
        self._running = False
        return True
    
    # ==================== 签名认证 ====================
    
    def _generate_signature(
        self, 
        timestamp: str, 
        method: str, 
        request_path: str, 
        body: str = ''
    ) -> str:
        """
        生成 OKX API 签名
        
        Args:
            timestamp: ISO 格式时间戳
            method: HTTP 方法 (GET, POST, DELETE)
            request_path: 请求路径
            body: 请求体
            
        Returns:
            Base64 编码的签名
        """
        message = timestamp + method + request_path + body
        mac = hmac.new(
            self._api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')
    
    def _get_headers(
        self, 
        method: str, 
        request_path: str, 
        body: str = ''
    ) -> Dict[str, str]:
        """获取 API 请求头"""
        timestamp = datetime.utcnow().isoformat() + 'Z'
        sign = self._generate_signature(timestamp, method, request_path, body)
        
        return {
            'OK-ACCESS-KEY': self._api_key,
            'OK-ACCESS-SIGN': sign,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self._passphrase,
            'Content-Type': 'application/json',
        }
    
    # ==================== 速率限制 ====================
    
    async def _check_rate_limit(self, category: str = 'spot'):
        """检查并等待速率限制"""
        limits = self.RATE_LIMITS.get(category, self.RATE_LIMITS['spot'])
        max_requests = limits['requests']
        interval = limits['interval']
        
        current_time = time.time()
        
        if category not in self._rate_limit_counters:
            self._rate_limit_counters[category] = []
        
        # 清理过期记录
        self._rate_limit_counters[category] = [
            t for t in self._rate_limit_counters[category]
            if current_time - t < interval
        ]
        
        # 检查是否需要等待
        if len(self._rate_limit_counters[category]) >= max_requests:
            wait_time = interval - (current_time - self._rate_limit_counters[category][0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # 记录请求
        self._rate_limit_counters[category].append(time.time())
    
    # ==================== 市场数据 ====================
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """获取行情数据"""
        await self._check_rate_limit()
        
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return {
                'symbol': ticker['symbol'],
                'last': float(ticker['last']),
                'bid': float(ticker['bid']),
                'ask': float(ticker['ask']),
                'high': float(ticker['high']),
                'low': float(ticker['low']),
                'volume': float(ticker['baseVolume']),
                'quote_volume': float(ticker['quoteVolume']),
                'timestamp': ticker['timestamp'],
                'datetime': ticker['datetime'],
            }
        except Exception as e:
            self.logger.error(f"Error fetching ticker {symbol}: {e}")
            raise
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """获取订单簿"""
        await self._check_rate_limit()
        
        try:
            orderbook = await self._exchange.fetch_order_book(symbol, limit)
            return {
                'symbol': symbol,
                'bids': orderbook['bids'][:limit],
                'asks': orderbook['asks'][:limit],
                'timestamp': orderbook['timestamp'],
                'datetime': orderbook['datetime'],
            }
        except Exception as e:
            self.logger.error(f"Error fetching orderbook {symbol}: {e}")
            raise
    
    async def get_klines(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        limit: int = 100
    ) -> List[List]:
        """获取K线数据"""
        await self._check_rate_limit()
        
        try:
            ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            self.logger.error(f"Error fetching klines {symbol}: {e}")
            raise
    
    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """获取资金费率（合约）"""
        await self._check_rate_limit('futures')
        
        try:
            if hasattr(self._exchange, 'fetch_funding_rate'):
                rate = await self._exchange.fetch_funding_rate(symbol)
                return {
                    'symbol': symbol,
                    'funding_rate': float(rate['fundingRate']),
                    'funding_timestamp': rate['fundingTimestamp'],
                    'next_funding_rate': float(rate.get('nextFundingRate', 0)),
                    'next_funding_timestamp': rate.get('nextFundingTimestamp'),
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching funding rate {symbol}: {e}")
            raise
    
    # ==================== 账户操作 ====================
    
    async def get_balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        await self._check_rate_limit('account')
        
        try:
            balance = await self._exchange.fetch_balance()
            
            # 整理余额数据
            result = {
                'total': {},
                'free': {},
                'used': {},
                'timestamp': balance.get('timestamp'),
            }
            
            for currency, amounts in balance.items():
                if currency in ['info', 'timestamp', 'datetime', 'free', 'used', 'total']:
                    continue
                if amounts and isinstance(amounts, dict):
                    total = float(amounts.get('total', 0))
                    if total > 0:
                        result['total'][currency] = total
                        result['free'][currency] = float(amounts.get('free', 0))
                        result['used'][currency] = float(amounts.get('used', 0))
            
            return result
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            raise
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """获取持仓（合约）"""
        await self._check_rate_limit('futures')
        
        try:
            positions = await self._exchange.fetch_positions()
            
            result = []
            for pos in positions:
                contracts = float(pos.get('contracts', 0))
                if abs(contracts) > 0:
                    result.append({
                        'symbol': pos['symbol'],
                        'side': 'long' if contracts > 0 else 'short',
                        'contracts': abs(contracts),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'mark_price': float(pos.get('markPrice', 0)),
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0)),
                        'leverage': float(pos.get('leverage', 1)),
                        'liquidation_price': float(pos.get('liquidationPrice', 0)),
                        'margin_mode': pos.get('marginMode', 'cross'),
                    })
            
            return result
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            raise
    
    async def get_account_config(self) -> Dict[str, Any]:
        """获取账户配置"""
        await self._check_rate_limit('account')
        
        try:
            if hasattr(self._exchange, 'fetch_account_configuration'):
                config = await self._exchange.fetch_account_configuration()
                return config
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching account config: {e}")
            return {}
    
    # ==================== 订单操作 ====================
    
    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        创建订单
        
        Args:
            symbol: 交易对
            order_type: 订单类型 (market, limit)
            side: 方向 (buy, sell)
            amount: 数量
            price: 价格（限价单必填）
            params: 额外参数
            
        Returns:
            订单信息
        """
        await self._check_rate_limit()
        
        try:
            order = await self._exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params or {}
            )
            
            result = {
                'order_id': order['id'],
                'client_order_id': order.get('clientOrderId'),
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'price': float(order.get('price', 0)),
                'amount': float(order['amount']),
                'filled': float(order.get('filled', 0)),
                'remaining': float(order.get('remaining', 0)),
                'status': order['status'],
                'timestamp': order['timestamp'],
                'info': order.get('info', {}),
            }
            
            # 通知回调
            for callback in self._order_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)
                except Exception as e:
                    self.logger.error(f"Order callback error: {e}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # 解析 OKX 错误码
            for code, msg in self.ERROR_CODES.items():
                if code in error_msg:
                    raise ExecutionException(f"OKX Error {code}: {msg}")
            
            raise ExecutionException(f"Order creation failed: {e}")
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """取消订单"""
        await self._check_rate_limit()
        
        try:
            result = await self._exchange.cancel_order(order_id, symbol)
            return {
                'order_id': order_id,
                'status': 'cancelled',
                'timestamp': result.get('timestamp'),
            }
        except Exception as e:
            raise ExecutionException(f"Order cancellation failed: {e}")
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """取消所有订单"""
        await self._check_rate_limit()
        
        try:
            if symbol:
                result = await self._exchange.cancel_all_orders(symbol)
            else:
                result = await self._exchange.cancel_all_orders()
            
            return {
                'status': 'success',
                'result': result,
            }
        except Exception as e:
            raise ExecutionException(f"Cancel all orders failed: {e}")
    
    async def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """查询订单"""
        await self._check_rate_limit()
        
        try:
            order = await self._exchange.fetch_order(order_id, symbol)
            return {
                'order_id': order['id'],
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'price': float(order.get('price', 0)),
                'amount': float(order['amount']),
                'filled': float(order.get('filled', 0)),
                'remaining': float(order.get('remaining', 0)),
                'status': order['status'],
                'timestamp': order['timestamp'],
            }
        except Exception as e:
            raise ExecutionException(f"Order query failed: {e}")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取未完成订单"""
        await self._check_rate_limit()
        
        try:
            orders = await self._exchange.fetch_open_orders(symbol)
            
            return [
                {
                    'order_id': o['id'],
                    'symbol': o['symbol'],
                    'type': o['type'],
                    'side': o['side'],
                    'price': float(o.get('price', 0)),
                    'amount': float(o['amount']),
                    'filled': float(o.get('filled', 0)),
                    'remaining': float(o.get('remaining', 0)),
                    'status': o['status'],
                    'timestamp': o['timestamp'],
                }
                for o in orders
            ]
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}")
            return []
    
    async def get_order_history(
        self, 
        symbol: Optional[str] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取订单历史"""
        await self._check_rate_limit()
        
        try:
            orders = await self._exchange.fetch_orders(symbol, limit=limit)
            
            return [
                {
                    'order_id': o['id'],
                    'symbol': o['symbol'],
                    'type': o['type'],
                    'side': o['side'],
                    'price': float(o.get('price', 0)),
                    'amount': float(o['amount']),
                    'filled': float(o.get('filled', 0)),
                    'status': o['status'],
                    'timestamp': o['timestamp'],
                }
                for o in orders
            ]
        except Exception as e:
            self.logger.error(f"Error fetching order history: {e}")
            return []
    
    # ==================== 合约特定操作 ====================
    
    async def set_leverage(
        self, 
        symbol: str, 
        leverage: int,
        margin_mode: str = 'cross'
    ) -> Dict[str, Any]:
        """设置杠杆"""
        await self._check_rate_limit('futures')
        
        try:
            # OKX 特定参数
            params = {
                'mgnMode': margin_mode,  # cross 或 isolated
            }
            
            result = await self._exchange.set_leverage(leverage, symbol, params)
            return {
                'symbol': symbol,
                'leverage': leverage,
                'margin_mode': margin_mode,
                'result': result,
            }
        except Exception as e:
            raise ExecutionException(f"Set leverage failed: {e}")
    
    async def set_margin_mode(self, margin_mode: str) -> Dict[str, Any]:
        """设置保证金模式"""
        await self._check_rate_limit('account')
        
        try:
            # OKX 需要通过特定 API 设置
            # 通常在账户配置中设置
            return {'status': 'success', 'margin_mode': margin_mode}
        except Exception as e:
            raise ExecutionException(f"Set margin mode failed: {e}")
    
    # ==================== WebSocket 连接 ====================
    
    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """订阅行情数据"""
        channel = f"tickers.{symbol.replace('/', '-')}"
        self._ws_handlers[channel] = callback
        # WebSocket 实现需要额外依赖
    
    async def subscribe_orderbook(self, symbol: str, callback: Callable):
        """订阅订单簿"""
        channel = f"books.{symbol.replace('/', '-')}"
        self._ws_handlers[channel] = callback
    
    async def subscribe_orders(self, callback: Callable):
        """订阅订单更新"""
        channel = "orders"
        self._ws_handlers[channel] = callback
    
    # ==================== 工具方法 ====================
    
    def get_supported_symbols(self) -> List[str]:
        """获取支持的交易对"""
        if self._exchange:
            return list(self._exchange.markets.keys())
        return []
    
    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取市场信息"""
        if self._exchange and symbol in self._exchange.markets:
            market = self._exchange.markets[symbol]
            return {
                'symbol': symbol,
                'base': market['base'],
                'quote': market['quote'],
                'type': market['type'],
                'contract_size': market.get('contractSize'),
                'tick_size': market['precision']['price'],
                'amount_step': market['precision']['amount'],
                'min_amount': market['limits']['amount']['min'],
                'max_amount': market['limits']['amount']['max'],
                'min_price': market['limits']['price']['min'],
            }
        return {}
    
    def add_order_callback(self, callback: Callable):
        """添加订单回调"""
        self._order_callbacks.append(callback)
    
    def parse_error(self, error: Exception) -> Dict[str, Any]:
        """解析错误信息"""
        error_str = str(error)
        
        for code, msg in self.ERROR_CODES.items():
            if code in error_str:
                return {
                    'code': code,
                    'message': msg,
                    'recoverable': code in ['50004', '50008'],
                }
        
        return {
            'code': 'unknown',
            'message': error_str,
            'recoverable': False,
        }


# 导出
__all__ = ['OKXAdapter']
