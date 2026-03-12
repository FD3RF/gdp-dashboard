"""
Social Sentiment Collector for social media sentiment analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import aiohttp
import re
from dataclasses import dataclass, field
from core.base import BaseModule


@dataclass
class SocialPost:
    """Represents a social media post."""
    id: str
    platform: str
    author: str
    content: str
    timestamp: datetime
    likes: int = 0
    shares: int = 0
    comments: int = 0
    sentiment: float = 0.0
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'platform': self.platform,
            'author': self.author,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'likes': self.likes,
            'shares': self.shares,
            'sentiment': self.sentiment
        }


class SentimentAnalyzer:
    """Simple sentiment analyzer using keyword matching."""
    
    POSITIVE_WORDS = {
        'bullish', 'moon', 'buy', 'long', 'pump', 'gain', 'profit',
        'uptrend', 'rally', 'breakout', 'support', 'accumulate',
        'positive', 'growth', 'surge', 'soar', 'rocket'
    }
    
    NEGATIVE_WORDS = {
        'bearish', 'crash', 'sell', 'short', 'dump', 'loss',
        'downtrend', 'breakdown', 'resistance', 'fear', 'panic',
        'negative', 'decline', 'drop', 'plunge', 'collapse'
    }
    
    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment analysis results
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        positive_count = len(words & cls.POSITIVE_WORDS)
        negative_count = len(words & cls.NEGATIVE_WORDS)
        
        total = positive_count + negative_count
        if total == 0:
            sentiment = 0.0
        else:
            sentiment = (positive_count - negative_count) / total
        
        return {
            'sentiment': sentiment,
            'positive_words': list(words & cls.POSITIVE_WORDS),
            'negative_words': list(words & cls.NEGATIVE_WORDS)
        }


class SocialSentimentCollector(BaseModule):
    """
    Collects social media sentiment data.
    Supports Twitter, Reddit, and custom sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('social_sentiment_collector', config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._posts: List[SocialPost] = []
        self._sentiment_history: Dict[str, List[float]] = {}
        self._max_posts = self.config.get('max_posts', 10000)
        
        # Keywords to track
        self._tracked_keywords = self.config.get('keywords', [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto',
            'trading', 'market', 'bull', 'bear'
        ])
        
        # API credentials
        self._twitter_bearer_token = self.config.get('twitter_bearer_token')
        self._reddit_client_id = self.config.get('reddit_client_id')
        self._reddit_client_secret = self.config.get('reddit_client_secret')
    
    async def initialize(self) -> bool:
        """Initialize the social sentiment collector."""
        self.logger.info("Initializing social sentiment collector...")
        
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
    
    async def fetch_twitter_posts(
        self,
        query: str,
        limit: int = 100
    ) -> List[SocialPost]:
        """
        Fetch posts from Twitter/X.
        
        Args:
            query: Search query
            limit: Maximum posts
        
        Returns:
            List of SocialPost
        """
        if not self._twitter_bearer_token:
            return []
        
        posts = []
        
        try:
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {self._twitter_bearer_token}'
            }
            params = {
                'query': query,
                'max_results': min(limit, 100),
                'tweet.fields': 'created_at,public_metrics'
            }
            
            async with self._session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for tweet in data.get('data', []):
                        metrics = tweet.get('public_metrics', {})
                        
                        post = SocialPost(
                            id=tweet['id'],
                            platform='twitter',
                            author=tweet.get('author_id', 'unknown'),
                            content=tweet['text'],
                            timestamp=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                            likes=metrics.get('like_count', 0),
                            shares=metrics.get('retweet_count', 0),
                            comments=metrics.get('reply_count', 0)
                        )
                        
                        # Analyze sentiment
                        sentiment_result = SentimentAnalyzer.analyze(post.content)
                        post.sentiment = sentiment_result['sentiment']
                        
                        posts.append(post)
        
        except Exception as e:
            self.logger.error(f"Error fetching Twitter posts: {e}")
        
        return posts
    
    async def fetch_reddit_posts(
        self,
        subreddit: str = 'cryptocurrency',
        limit: int = 100
    ) -> List[SocialPost]:
        """
        Fetch posts from Reddit.
        
        Args:
            subreddit: Subreddit name
            limit: Maximum posts
        
        Returns:
            List of SocialPost
        """
        posts = []
        
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            headers = {'User-Agent': 'AIQuantSystem/1.0'}
            params = {'limit': limit}
            
            async with self._session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for child in data.get('data', {}).get('children', []):
                        post_data = child.get('data', {})
                        
                        post = SocialPost(
                            id=post_data.get('id', ''),
                            platform='reddit',
                            author=post_data.get('author', 'unknown'),
                            content=post_data.get('title', '') + ' ' + post_data.get('selftext', ''),
                            timestamp=datetime.fromtimestamp(post_data.get('created_utc', 0)),
                            likes=post_data.get('ups', 0),
                            shares=post_data.get('num_crossposts', 0),
                            comments=post_data.get('num_comments', 0)
                        )
                        
                        # Analyze sentiment
                        sentiment_result = SentimentAnalyzer.analyze(post.content)
                        post.sentiment = sentiment_result['sentiment']
                        
                        posts.append(post)
        
        except Exception as e:
            self.logger.error(f"Error fetching Reddit posts: {e}")
        
        return posts
    
    async def fetch_all_sentiment(self) -> Dict[str, Any]:
        """
        Fetch and aggregate sentiment from all sources.
        
        Returns:
            Aggregated sentiment data
        """
        tasks = [
            self.fetch_reddit_posts('cryptocurrency', 50),
            self.fetch_reddit_posts('Bitcoin', 50),
        ]
        
        if self._twitter_bearer_token:
            tasks.append(self.fetch_twitter_posts('bitcoin OR crypto', 50))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_posts = []
        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)
        
        # Update cache
        self._posts = (all_posts + self._posts)[:self._max_posts]
        
        # Calculate aggregate sentiment
        if all_posts:
            avg_sentiment = sum(p.sentiment for p in all_posts) / len(all_posts)
            positive_count = sum(1 for p in all_posts if p.sentiment > 0.2)
            negative_count = sum(1 for p in all_posts if p.sentiment < -0.2)
            neutral_count = len(all_posts) - positive_count - negative_count
        else:
            avg_sentiment = 0
            positive_count = negative_count = neutral_count = 0
        
        # Track sentiment history
        timestamp_key = datetime.now().strftime('%Y-%m-%d %H:%M')
        if 'aggregate' not in self._sentiment_history:
            self._sentiment_history['aggregate'] = []
        self._sentiment_history['aggregate'].append(avg_sentiment)
        
        return {
            'timestamp': datetime.now(),
            'average_sentiment': avg_sentiment,
            'total_posts': len(all_posts),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_label': self._get_sentiment_label(avg_sentiment)
        }
    
    def _get_sentiment_label(self, sentiment: float) -> str:
        """Get sentiment label from score."""
        if sentiment > 0.3:
            return 'very_bullish'
        elif sentiment > 0.1:
            return 'bullish'
        elif sentiment < -0.3:
            return 'very_bearish'
        elif sentiment < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_sentiment_trend(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get sentiment trend over time.
        
        Args:
            hours: Hours to analyze
        
        Returns:
            Trend data
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_posts = [p for p in self._posts if p.timestamp >= cutoff]
        
        if not recent_posts:
            return {'trend': 'stable', 'change': 0}
        
        # Group by hour
        hourly_sentiment = {}
        for post in recent_posts:
            hour_key = post.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_sentiment:
                hourly_sentiment[hour_key] = []
            hourly_sentiment[hour_key].append(post.sentiment)
        
        # Calculate hourly averages
        hourly_avg = {
            ts: sum(s) / len(s)
            for ts, s in sorted(hourly_sentiment.items())
        }
        
        if len(hourly_avg) < 2:
            return {'trend': 'stable', 'change': 0}
        
        # Calculate trend
        values = list(hourly_avg.values())
        change = values[-1] - values[0]
        
        if change > 0.2:
            trend = 'improving'
        elif change < -0.2:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change': change,
            'hourly_sentiment': {k.isoformat(): v for k, v in hourly_avg.items()}
        }
    
    def get_posts_by_keyword(self, keyword: str) -> List[SocialPost]:
        """Get posts containing a specific keyword."""
        keyword_lower = keyword.lower()
        return [
            p for p in self._posts
            if keyword_lower in p.content.lower()
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            'healthy': self._initialized,
            'name': self.name,
            'timestamp': datetime.now().isoformat(),
            'cached_posts': len(self._posts)
        }
