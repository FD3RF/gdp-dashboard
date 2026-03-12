"""
On-chain Data Collector for blockchain data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import aiohttp
import pandas as pd
from core.base import BaseModule
from core.exceptions import DataSourceException


class OnChainDataCollector(BaseModule):
    """
    Collects on-chain data from various blockchain analytics providers.
    Supports Glassnode, Blockchain.com, Etherscan APIs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('onchain_data_collector', config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        # API keys
        self._glassnode_api_key = self.config.get('glassnode_api_key')
        self._etherscan_api_key = self.config.get('etherscan_api_key')
    
    async def initialize(self) -> bool:
        """Initialize the on-chain data collector."""
        self.logger.info("Initializing on-chain data collector...")
        
        timeout = aiohttp.ClientTimeout(total=60)
        self._session = aiohttp.ClientSession(timeout=timeout)
        
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the collector."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the collector."""
        if self._session:
            await self._session.close()
        self._running = False
        return True
    
    async def fetch_btc_network_stats(self) -> Dict[str, Any]:
        """
        Fetch Bitcoin network statistics.
        
        Returns:
            Dictionary with network stats
        """
        try:
            # Fetch from blockchain.com API
            url = "https://api.blockchain.info/stats"
            
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'timestamp': datetime.now(),
                        'btc_price': data.get('market_price_usd'),
                        'hashrate': data.get('hash_rate'),
                        'difficulty': data.get('difficulty'),
                        'block_height': data.get('n_blocks_total'),
                        'total_btc': data.get('totalbc'),
                        'transactions_24h': data.get('n_tx'),
                        'avg_block_size': data.get('avg_block_size'),
                        'mempool_size': data.get('n_btc_mined'),
                    }
        except Exception as e:
            self.logger.error(f"Error fetching BTC stats: {e}")
        
        return {}
    
    async def fetch_eth_network_stats(self) -> Dict[str, Any]:
        """
        Fetch Ethereum network statistics.
        
        Returns:
            Dictionary with network stats
        """
        if not self._etherscan_api_key:
            return {}
        
        try:
            url = f"https://api.etherscan.io/api?module=stats&action=ethprice&apikey={self._etherscan_api_key}"
            
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == '1':
                        result = data.get('result', {})
                        return {
                            'timestamp': datetime.now(),
                            'eth_price': float(result.get('ethusd', 0)),
                            'eth_btc': float(result.get('ethbtc', 0)),
                        }
        except Exception as e:
            self.logger.error(f"Error fetching ETH stats: {e}")
        
        return {}
    
    async def fetch_exchange_flows(
        self,
        asset: str = 'BTC',
        exchange: str = 'all'
    ) -> Dict[str, Any]:
        """
        Fetch exchange inflow/outflow data.
        
        Args:
            asset: Asset symbol (BTC, ETH)
            exchange: Exchange name or 'all'
        
        Returns:
            Exchange flow data
        """
        # Requires Glassnode API
        if not self._glassnode_api_key:
            return self._estimate_exchange_flows(asset)
        
        try:
            url = f"https://api.glassnode.com/v1/metrics/exchanges/netflow"
            params = {
                'api_key': self._glassnode_api_key,
                'a': asset.lower(),
                'e': exchange
            }
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'asset': asset,
                        'exchange': exchange,
                        'data': data,
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            self.logger.error(f"Error fetching exchange flows: {e}")
        
        return {}
    
    def _estimate_exchange_flows(self, asset: str) -> Dict[str, Any]:
        """Estimate exchange flows when API not available."""
        import numpy as np
        
        # Generate simulated data for demonstration
        return {
            'asset': asset,
            'exchange': 'all',
            'net_flow': np.random.uniform(-1000, 1000),
            'estimated': True,
            'timestamp': datetime.now()
        }
    
    async def fetch_whale_transactions(
        self,
        asset: str = 'BTC',
        min_value: float = 1000000
    ) -> List[Dict[str, Any]]:
        """
        Fetch large whale transactions.
        
        Args:
            asset: Asset symbol
            min_value: Minimum transaction value in USD
        
        Returns:
            List of large transactions
        """
        # This would integrate with whale alert APIs
        # For now, return empty list
        return []
    
    async def fetch_active_addresses(
        self,
        asset: str = 'BTC'
    ) -> Dict[str, Any]:
        """
        Fetch active address count.
        
        Args:
            asset: Asset symbol
        
        Returns:
            Active address data
        """
        if not self._glassnode_api_key:
            return {'asset': asset, 'estimated': True, 'count': 0}
        
        try:
            url = "https://api.glassnode.com/v1/metrics/addresses/active_count"
            params = {
                'api_key': self._glassnode_api_key,
                'a': asset.lower()
            }
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        latest = data[-1]
                        return {
                            'asset': asset,
                            'active_addresses': latest.get('v'),
                            'timestamp': datetime.now()
                        }
        except Exception as e:
            self.logger.error(f"Error fetching active addresses: {e}")
        
        return {'asset': asset, 'active_addresses': 0}
    
    async def fetch_nvt_ratio(self, asset: str = 'BTC') -> Dict[str, Any]:
        """
        Fetch Network Value to Transactions ratio.
        
        Args:
            asset: Asset symbol
        
        Returns:
            NVT ratio data
        """
        # Requires premium API access
        return {'asset': asset, 'nvt_ratio': None, 'estimated': True}
    
    async def fetch_mvrv_ratio(self, asset: str = 'BTC') -> Dict[str, Any]:
        """
        Fetch Market Value to Realized Value ratio.
        
        Args:
            asset: Asset symbol
        
        Returns:
            MVRV ratio data
        """
        # Requires premium API access
        return {'asset': asset, 'mvrv_ratio': None, 'estimated': True}
    
    async def get_onchain_metrics(self, asset: str = 'BTC') -> Dict[str, Any]:
        """
        Get comprehensive on-chain metrics for an asset.
        
        Args:
            asset: Asset symbol
        
        Returns:
            Dictionary with all metrics
        """
        results = await asyncio.gather(
            self.fetch_btc_network_stats() if asset == 'BTC' else self.fetch_eth_network_stats(),
            self.fetch_exchange_flows(asset),
            self.fetch_active_addresses(asset),
            return_exceptions=True
        )
        
        metrics = {'asset': asset, 'timestamp': datetime.now()}
        
        for result in results:
            if isinstance(result, dict):
                metrics.update(result)
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'healthy': self._initialized,
            'name': self.name,
            'timestamp': datetime.now().isoformat()
        }
