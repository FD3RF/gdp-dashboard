"""
Exchange Adapter for unified exchange interface.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import ccxt.async_support as ccxt
from core.base import BaseModule
from core.exceptions import ExecutionException


class ExchangeAdapter(BaseModule):
    """
    Unified interface for multiple exchanges.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('exchange_adapter', config)
        
        self._exchanges: Dict[str, ccxt.Exchange] = {}
        self._default_exchange = self.config.get('default_exchange', 'binance')
        self._testnet = self.config.get('testnet', True)
    
    async def initialize(self) -> bool:
        """Initialize exchange connections."""
        self.logger.info("Initializing exchange adapter...")
        
        exchanges_config = self.config.get('exchanges', {})
        
        if not exchanges_config:
            # Add default Binance
            exchanges_config['binance'] = {
                'apiKey': self.config.get('binance_api_key', ''),
                'secret': self.config.get('binance_api_secret', ''),
                'enableRateLimit': True,
                'options': {'defaultType': 'future' if self._testnet else 'spot'}
            }
        
        for name, ex_config in exchanges_config.items():
            try:
                exchange_class = getattr(ccxt, name)
                self._exchanges[name] = exchange_class(ex_config)
                await self._exchanges[name].load_markets()
                self.logger.info(f"Connected to exchange: {name}")
            except Exception as e:
                self.logger.error(f"Failed to connect to {name}: {e}")
        
        self._initialized = len(self._exchanges) > 0
        return self._initialized
    
    async def start(self) -> bool:
        self._running = True
        return True
    
    async def stop(self) -> bool:
        for name, exchange in self._exchanges.items():
            try:
                await exchange.close()
            except Exception as e:
                self.logger.error(f"Error closing {name}: {e}")
        self._running = False
        return True
    
    def get_exchange(self, name: Optional[str] = None) -> Optional[ccxt.Exchange]:
        """Get exchange instance."""
        name = name or self._default_exchange
        return self._exchanges.get(name)
    
    async def submit_order(self, order: Dict) -> Dict[str, Any]:
        """Submit order to exchange."""
        exchange_name = order.get('exchange', self._default_exchange)
        exchange = self._exchanges.get(exchange_name)
        
        if not exchange:
            raise ExecutionException(f"Exchange not found: {exchange_name}")
        
        try:
            result = await exchange.create_order(
                symbol=order['symbol'],
                type=order['type'],
                side=order['side'],
                amount=order['quantity'],
                price=order.get('price')
            )
            
            return {
                'exchange_order_id': result.get('id'),
                'status': 'open',
                'filled_quantity': result.get('filled', 0),
                'filled_price': result.get('average'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            raise ExecutionException(f"Order submission failed: {e}")
    
    async def cancel_order(self, order_id: str, symbol: str, exchange: Optional[str] = None) -> Dict:
        """Cancel an order."""
        exchange_name = exchange or self._default_exchange
        ex = self._exchanges.get(exchange_name)
        
        if not ex:
            return {'error': 'Exchange not found'}
        
        try:
            result = await ex.cancel_order(order_id, symbol)
            return {'status': 'cancelled', 'result': result}
        except Exception as e:
            return {'error': str(e)}
    
    async def get_order_status(self, order_id: str, symbol: str, exchange: Optional[str] = None) -> Dict:
        """Get order status."""
        exchange_name = exchange or self._default_exchange
        ex = self._exchanges.get(exchange_name)
        
        if not ex:
            return {'error': 'Exchange not found'}
        
        try:
            result = await ex.fetch_order(order_id, symbol)
            return {
                'exchange_order_id': result.get('id'),
                'status': result.get('status'),
                'filled_quantity': result.get('filled', 0),
                'filled_price': result.get('average'),
                'remaining': result.get('remaining', 0)
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def get_balance(self, exchange: Optional[str] = None) -> Dict:
        """Get account balance."""
        exchange_name = exchange or self._default_exchange
        ex = self._exchanges.get(exchange_name)
        
        if not ex:
            return {}
        
        try:
            balance = await ex.fetch_balance()
            return {
                'total': balance.get('total', {}),
                'free': balance.get('free', {}),
                'used': balance.get('used', {})
            }
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {}
    
    async def get_positions(self, exchange: Optional[str] = None) -> List[Dict]:
        """Get open positions (for futures)."""
        exchange_name = exchange or self._default_exchange
        ex = self._exchanges.get(exchange_name)
        
        if not ex:
            return []
        
        try:
            positions = await ex.fetch_positions()
            return [p for p in positions if float(p.get('contracts', 0)) != 0]
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return []
    
    def get_supported_symbols(self, exchange: Optional[str] = None) -> List[str]:
        """Get supported trading symbols."""
        exchange_name = exchange or self._default_exchange
        ex = self._exchanges.get(exchange_name)
        
        if ex:
            return list(ex.markets.keys())
        return []
