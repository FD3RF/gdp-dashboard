"""
Macro Data Collector for macroeconomic indicators.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import aiohttp
from dataclasses import dataclass, field
from core.base import BaseModule


@dataclass
class MacroIndicator:
    """Represents a macroeconomic indicator."""
    name: str
    value: float
    unit: str
    country: str
    timestamp: datetime
    previous_value: Optional[float] = None
    forecast: Optional[float] = None
    importance: str = 'medium'  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'country': self.country,
            'timestamp': self.timestamp.isoformat(),
            'previous': self.previous_value,
            'forecast': self.forecast,
            'importance': self.importance
        }


class MacroDataCollector(BaseModule):
    """
    Collects macroeconomic data and indicators.
    Supports FRED, FRED API, and economic calendars.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('macro_data_collector', config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._indicators: Dict[str, MacroIndicator] = {}
        
        # API keys
        self._fred_api_key = self.config.get('fred_api_key')
        
        # Tracked indicators
        self._tracked_indicators = self.config.get('indicators', [
            'CPIAUCSL',  # Consumer Price Index
            'UNRATE',     # Unemployment Rate
            'GDP',        # Gross Domestic Product
            'FEDFUNDS',   # Federal Funds Rate
            'DGS10',      # 10-Year Treasury Yield
            'DEXUSEU',    # US/Euro Exchange Rate
        ])
    
    async def initialize(self) -> bool:
        """Initialize the macro data collector."""
        self.logger.info("Initializing macro data collector...")
        
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
    
    async def fetch_fred_series(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch data series from FRED API.
        
        Args:
            series_id: FRED series ID
            observation_start: Start date (YYYY-MM-DD)
            observation_end: End date (YYYY-MM-DD)
        
        Returns:
            Series data
        """
        if not self._fred_api_key:
            return {'error': 'FRED API key not configured'}
        
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self._fred_api_key,
                'file_type': 'json'
            }
            
            if observation_start:
                params['observation_start'] = observation_start
            if observation_end:
                params['observation_end'] = observation_end
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    observations = data.get('observations', [])
                    
                    if observations:
                        latest = observations[-1]
                        previous = observations[-2] if len(observations) > 1 else None
                        
                        indicator = MacroIndicator(
                            name=series_id,
                            value=float(latest['value']) if latest['value'] != '.' else 0,
                            unit='index',
                            country='US',
                            timestamp=datetime.strptime(latest['date'], '%Y-%m-%d'),
                            previous_value=float(previous['value']) if previous and previous['value'] != '.' else None
                        )
                        
                        self._indicators[series_id] = indicator
                        
                        return {
                            'series_id': series_id,
                            'observations': observations,
                            'latest': indicator.to_dict()
                        }
        
        except Exception as e:
            self.logger.error(f"Error fetching FRED series {series_id}: {e}")
        
        return {'series_id': series_id, 'error': 'Failed to fetch'}
    
    async def fetch_economic_calendar(
        self,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Fetch economic calendar events.
        
        Args:
            days: Number of days to fetch
        
        Returns:
            List of calendar events
        """
        # This would typically integrate with a financial calendar API
        # For now, return simulated data
        events = []
        
        event_templates = [
            {'name': 'FOMC Meeting', 'importance': 'high'},
            {'name': 'Non-Farm Payrolls', 'importance': 'high'},
            {'name': 'CPI Release', 'importance': 'high'},
            {'name': 'GDP Release', 'importance': 'medium'},
            {'name': 'Unemployment Rate', 'importance': 'medium'},
            {'name': 'Retail Sales', 'importance': 'medium'},
        ]
        
        import random
        
        for i, template in enumerate(event_templates):
            event_date = datetime.now() + timedelta(days=random.randint(0, days))
            events.append({
                'name': template['name'],
                'date': event_date.strftime('%Y-%m-%d'),
                'importance': template['importance'],
                'country': 'US',
                'forecast': round(random.uniform(0.1, 5.0), 2),
                'previous': round(random.uniform(0.1, 5.0), 2)
            })
        
        return events
    
    async def fetch_all_indicators(self) -> Dict[str, MacroIndicator]:
        """
        Fetch all tracked indicators.
        
        Returns:
            Dictionary of indicators
        """
        if not self._fred_api_key:
            # Return simulated data
            return self._get_simulated_indicators()
        
        tasks = [
            self.fetch_fred_series(series_id)
            for series_id in self._tracked_indicators
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return self._indicators
    
    def _get_simulated_indicators(self) -> Dict[str, MacroIndicator]:
        """Get simulated indicator data when API not available."""
        import random
        
        simulated = {
            'CPIAUCSL': MacroIndicator(
                name='Consumer Price Index',
                value=300.0 + random.uniform(-5, 5),
                unit='index',
                country='US',
                timestamp=datetime.now(),
                importance='high'
            ),
            'UNRATE': MacroIndicator(
                name='Unemployment Rate',
                value=round(3.5 + random.uniform(-0.5, 0.5), 1),
                unit='percent',
                country='US',
                timestamp=datetime.now(),
                importance='high'
            ),
            'FEDFUNDS': MacroIndicator(
                name='Federal Funds Rate',
                value=round(5.25 + random.uniform(-0.25, 0.25), 2),
                unit='percent',
                country='US',
                timestamp=datetime.now(),
                importance='high'
            ),
            'DGS10': MacroIndicator(
                name='10-Year Treasury Yield',
                value=round(4.5 + random.uniform(-0.5, 0.5), 2),
                unit='percent',
                country='US',
                timestamp=datetime.now(),
                importance='high'
            )
        }
        
        self._indicators.update(simulated)
        return simulated
    
    def get_indicator(self, name: str) -> Optional[MacroIndicator]:
        """Get a specific indicator."""
        return self._indicators.get(name)
    
    def get_all_indicators(self) -> Dict[str, MacroIndicator]:
        """Get all cached indicators."""
        return self._indicators.copy()
    
    def analyze_macro_environment(self) -> Dict[str, Any]:
        """
        Analyze current macro environment.
        
        Returns:
            Macro analysis summary
        """
        if not self._indicators:
            return {'status': 'no_data'}
        
        analysis = {
            'timestamp': datetime.now(),
            'indicators': {},
            'summary': {}
        }
        
        # Analyze each indicator
        for name, indicator in self._indicators.items():
            trend = 'stable'
            if indicator.previous_value:
                change = indicator.value - indicator.previous_value
                if change > 0:
                    trend = 'increasing'
                elif change < 0:
                    trend = 'decreasing'
            
            analysis['indicators'][name] = {
                **indicator.to_dict(),
                'trend': trend
            }
        
        # Generate summary
        fed_funds = self._indicators.get('FEDFUNDS')
        if fed_funds:
            analysis['summary']['interest_rate_environment'] = (
                'tightening' if fed_funds.value > 3 else 'loose'
            )
        
        unrate = self._indicators.get('UNRATE')
        if unrate:
            analysis['summary']['employment_health'] = (
                'strong' if unrate.value < 4.5 else 'weak'
            )
        
        cpi = self._indicators.get('CPIAUCSL')
        if cpi:
            analysis['summary']['inflation_trend'] = (
                'rising' if cpi.value > (cpi.previous_value or cpi.value) else 'falling'
            )
        
        return analysis
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'healthy': self._initialized,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'indicators_count': len(self._indicators)
        }
