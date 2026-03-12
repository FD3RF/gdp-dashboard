"""
Research Agent for market research and analysis.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from agents.base_agent import BaseAgent, AgentTask


class ResearchAgent(BaseAgent):
    """
    Research Agent responsible for:
    - Market research and analysis
    - Data gathering and synthesis
    - Generating research reports
    """
    
    def __init__(
        self,
        name: str = 'research',
        config: Optional[Dict[str, Any]] = None,
        model_manager = None,
        vector_memory = None,
        data_collectors = None
    ):
        super().__init__(name, config, model_manager, vector_memory)
        self._data_collectors = data_collectors or {}
        self._research_cache: Dict[str, Dict] = {}
    
    def register_data_collector(self, name: str, collector) -> None:
        """Register a data collector."""
        self._data_collectors[name] = collector
    
    def _get_default_system_prompt(self) -> str:
        return """You are the Research Agent in a quantitative trading system.
Your role is to:
1. Conduct thorough market research
2. Analyze market conditions and trends
3. Identify trading opportunities
4. Synthesize information from multiple sources
5. Generate actionable research reports

Always provide data-driven insights with specific metrics and evidence."""

    async def process_task(self, task: AgentTask) -> Any:
        """Process a research task."""
        if task.type == 'market_analysis':
            return await self._market_analysis(task)
        elif task.type == 'opportunity_scan':
            return await self._opportunity_scan(task)
        elif task.type == 'trend_analysis':
            return await self._trend_analysis(task)
        elif task.type == 'generate_report':
            return await self._generate_report(task)
        elif task.type == 'analyze_sentiment':
            return await self._analyze_sentiment(task)
        else:
            raise ValueError(f"Unknown task type: {task.type}")
    
    async def _market_analysis(self, task: AgentTask) -> Dict[str, Any]:
        """Perform comprehensive market analysis."""
        symbol = task.parameters.get('symbol', 'BTC/USDT')
        timeframe = task.parameters.get('timeframe', '1d')
        
        analysis = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'data': {},
            'insights': []
        }
        
        # Gather data from collectors
        market_collector = self._data_collectors.get('market')
        if market_collector:
            try:
                ohlcv = await market_collector.fetch_ohlcv(symbol, timeframe, limit=100)
                analysis['data']['ohlcv'] = {
                    'rows': len(ohlcv),
                    'start': str(ohlcv.index[0]) if len(ohlcv) > 0 else None,
                    'end': str(ohlcv.index[-1]) if len(ohlcv) > 0 else None,
                    'latest_close': float(ohlcv['close'].iloc[-1]) if len(ohlcv) > 0 else None
                }
            except Exception as e:
                self.logger.error(f"Error fetching OHLCV: {e}")
        
        # Generate insights using LLM
        prompt = f"""Analyze this market data and provide insights:

Symbol: {symbol}
Timeframe: {timeframe}
Data Summary: {json.dumps(analysis['data'], indent=2)}

Provide:
1. Current market condition (trending/ranging)
2. Key support and resistance levels
3. Notable patterns or signals
4. Risk factors
5. Recommended actions

Respond with JSON."""

        try:
            response = await self.generate_response(prompt)
            json_str = response[response.find('{'):response.rfind('}')+1]
            insights = json.loads(json_str)
            analysis['insights'] = insights
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            analysis['insights'] = {'error': str(e)}
        
        return analysis
    
    async def _opportunity_scan(self, task: AgentTask) -> Dict[str, Any]:
        """Scan for trading opportunities."""
        symbols = task.parameters.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        criteria = task.parameters.get('criteria', 'momentum')
        
        opportunities = []
        
        market_collector = self._data_collectors.get('market')
        if not market_collector:
            return {'opportunities': [], 'error': 'No market data collector'}
        
        for symbol in symbols:
            try:
                ohlcv = await market_collector.fetch_ohlcv(symbol, '1h', limit=100)
                
                if len(ohlcv) < 20:
                    continue
                
                # Calculate basic metrics
                returns = ohlcv['close'].pct_change()
                volatility = returns.std() * (252 * 24) ** 0.5  # Annualized
                momentum = (ohlcv['close'].iloc[-1] / ohlcv['close'].iloc[-20] - 1) * 100
                
                opportunities.append({
                    'symbol': symbol,
                    'volatility': volatility,
                    'momentum': momentum,
                    'score': abs(momentum) * volatility  # Simple scoring
                })
                
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'criteria': criteria,
            'opportunities': opportunities[:10]
        }
    
    async def _trend_analysis(self, task: AgentTask) -> Dict[str, Any]:
        """Analyze market trends."""
        symbol = task.parameters.get('symbol', 'BTC/USDT')
        lookback = task.parameters.get('lookback_days', 30)
        
        market_collector = self._data_collectors.get('market')
        if not market_collector:
            return {'error': 'No market data collector'}
        
        try:
            ohlcv = await market_collector.fetch_historical_data(
                symbol=symbol,
                timeframe='1d',
                start_date=datetime.now() - timedelta(days=lookback)
            )
            
            if ohlcv.empty:
                return {'error': 'No data available'}
            
            # Calculate trend metrics
            sma_20 = ohlcv['close'].rolling(20).mean()
            sma_50 = ohlcv['close'].rolling(50).mean()
            
            current_price = ohlcv['close'].iloc[-1]
            trend_direction = 'up' if current_price > sma_20.iloc[-1] else 'down'
            trend_strength = abs((current_price - sma_20.iloc[-1]) / sma_20.iloc[-1]) * 100
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': float(current_price),
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'sma_20': float(sma_20.iloc[-1]),
                'sma_50': float(sma_50.iloc[-1]) if len(ohlcv) >= 50 else None,
                'price_range_52w': {
                    'high': float(ohlcv['high'].max()),
                    'low': float(ohlcv['low'].min())
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _analyze_sentiment(self, task: AgentTask) -> Dict[str, Any]:
        """Analyze market sentiment."""
        keyword = task.parameters.get('keyword', 'bitcoin')
        
        sentiment_collector = self._data_collectors.get('sentiment')
        news_collector = self._data_collectors.get('news')
        
        sentiment_data = {
            'keyword': keyword,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get social sentiment
        if sentiment_collector:
            try:
                sentiment = await sentiment_collector.fetch_all_sentiment()
                sentiment_data['social'] = sentiment
            except Exception as e:
                sentiment_data['social_error'] = str(e)
        
        # Get news
        if news_collector:
            try:
                articles = news_collector.search_articles(keyword)
                sentiment_data['news_count'] = len(articles)
            except Exception as e:
                sentiment_data['news_error'] = str(e)
        
        return sentiment_data
    
    async def _generate_report(self, task: AgentTask) -> Dict[str, Any]:
        """Generate a research report."""
        topic = task.parameters.get('topic', 'market overview')
        data = task.parameters.get('data', {})
        
        prompt = f"""Generate a comprehensive research report on:

Topic: {topic}

Data: {json.dumps(data, indent=2, default=str)}

Structure the report with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Risk Assessment
5. Recommendations

Provide actionable insights for trading decisions."""

        report = await self.generate_response(prompt)
        
        return {
            'topic': topic,
            'report': report,
            'generated_at': datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get research agent status."""
        return {
            **super().get_status(),
            'data_collectors': list(self._data_collectors.keys()),
            'cached_research': len(self._research_cache)
        }
