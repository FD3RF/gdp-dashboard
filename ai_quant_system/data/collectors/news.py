"""
News Collector for market news and analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import aiohttp
import feedparser
from dataclasses import dataclass, field
from core.base import BaseModule
from core.exceptions import DataSourceException


@dataclass
class NewsArticle:
    """Represents a news article."""
    title: str
    source: str
    url: str
    published_at: datetime
    summary: str = ""
    content: str = ""
    categories: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'source': self.source,
            'url': self.url,
            'published_at': self.published_at.isoformat(),
            'summary': self.summary,
            'sentiment': self.sentiment,
            'relevance_score': self.relevance_score
        }


class NewsCollector(BaseModule):
    """
    Collects news data from various sources.
    Supports RSS feeds, CryptoCompare, CryptoPanic APIs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('news_collector', config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._articles: List[NewsArticle] = []
        self._max_articles = self.config.get('max_articles', 1000)
        
        # API keys
        self._cryptocompare_api_key = self.config.get('cryptocompare_api_key')
        self._cryptopanic_api_key = self.config.get('cryptopanic_api_key')
        
        # RSS feeds
        self._rss_feeds = self.config.get('rss_feeds', [
            'https://cointelegraph.com/rss',
            'https://cryptonews-api.com/api/v1/feed',
            'https://www.coindesk.com/arc/outboundfeeds/rss/'
        ])
    
    async def initialize(self) -> bool:
        """Initialize the news collector."""
        self.logger.info("Initializing news collector...")
        
        timeout = aiohttp.ClientTimeout(total=30)
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
    
    async def fetch_rss_feed(self, feed_url: str) -> List[NewsArticle]:
        """
        Fetch articles from RSS feed.
        
        Args:
            feed_url: RSS feed URL
        
        Returns:
            List of NewsArticle
        """
        articles = []
        
        try:
            async with self._session.get(feed_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries:
                        published = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                        
                        article = NewsArticle(
                            title=entry.get('title', ''),
                            source=feed.feed.get('title', 'Unknown'),
                            url=entry.get('link', ''),
                            published_at=published,
                            summary=entry.get('summary', ''),
                            categories=entry.get('tags', [])
                        )
                        articles.append(article)
        
        except Exception as e:
            self.logger.error(f"Error fetching RSS feed {feed_url}: {e}")
        
        return articles
    
    async def fetch_cryptocompare_news(
        self,
        categories: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Fetch news from CryptoCompare API.
        
        Args:
            categories: News categories
            limit: Number of articles
        
        Returns:
            List of NewsArticle
        """
        if not self._cryptocompare_api_key:
            return []
        
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            params = {
                'api_key': self._cryptocompare_api_key,
                'limit': limit
            }
            
            if categories:
                params['categories'] = ','.join(categories)
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = []
                    
                    for item in data.get('Data', []):
                        article = NewsArticle(
                            title=item.get('title', ''),
                            source=item.get('source', 'Unknown'),
                            url=item.get('url', ''),
                            published_at=datetime.fromtimestamp(item.get('published_on', 0)),
                            summary=item.get('body', ''),
                            categories=item.get('categories', '').split(',')
                        )
                        articles.append(article)
                    
                    return articles
        
        except Exception as e:
            self.logger.error(f"Error fetching CryptoCompare news: {e}")
        
        return []
    
    async def fetch_all_news(self, limit: int = 100) -> List[NewsArticle]:
        """
        Fetch news from all configured sources.
        
        Args:
            limit: Maximum articles to return
        
        Returns:
            List of NewsArticle
        """
        tasks = []
        
        # RSS feeds
        for feed_url in self._rss_feeds:
            tasks.append(self.fetch_rss_feed(feed_url))
        
        # CryptoCompare
        tasks.append(self.fetch_cryptocompare_news(limit=limit))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        
        # Sort by date and deduplicate
        seen_urls = set()
        unique_articles = []
        
        for article in sorted(all_articles, key=lambda x: x.published_at, reverse=True):
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        # Cache
        self._articles = unique_articles[:self._max_articles]
        
        return unique_articles[:limit]
    
    def get_cached_articles(self, limit: int = 50) -> List[NewsArticle]:
        """Get cached articles."""
        return self._articles[:limit]
    
    def search_articles(self, query: str) -> List[NewsArticle]:
        """
        Search cached articles by query.
        
        Args:
            query: Search query
        
        Returns:
            Matching articles
        """
        query_lower = query.lower()
        
        matches = []
        for article in self._articles:
            if (query_lower in article.title.lower() or
                query_lower in article.summary.lower()):
                matches.append(article)
        
        return matches
    
    def get_recent_articles(
        self,
        hours: int = 24,
        sources: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        """
        Get recent articles within specified hours.
        
        Args:
            hours: Number of hours to look back
            sources: Filter by sources
        
        Returns:
            Recent articles
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = []
        for article in self._articles:
            if article.published_at >= cutoff:
                if sources is None or article.source in sources:
                    recent.append(article)
        
        return sorted(recent, key=lambda x: x.published_at, reverse=True)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'healthy': self._initialized,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'cached_articles': len(self._articles)
        }
