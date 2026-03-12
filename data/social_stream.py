"""
社交情绪数据流模块 (Layer 2-4)
抓取 Twitter/Reddit/Telegram 情绪数据并量化
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from decimal import Decimal
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SocialPost:
    """社交媒体帖子数据结构"""
    platform: str  # twitter, reddit, telegram
    content: str
    timestamp: datetime
    author: str
    engagement: int  # 点赞/转发/评论数
    sentiment_score: float = 0.0  # -1 到 1
    keywords: List[str] = field(default_factory=list)


class SentimentAnalyzer:
    """
    NLP 情绪分析器
    使用关键词匹配 + 简单规则实现（生产环境可替换为 Transformers 模型）
    """
    
    # 看涨关键词
    BULLISH_KEYWORDS = {
        "bullish": 0.8, "moon": 0.6, "pump": 0.5, "buy": 0.4,
        "long": 0.4, "hold": 0.3, "hodl": 0.3, "breakout": 0.5,
        "support": 0.3, " accumulation": 0.4, "底": 0.5, "涨": 0.6,
        "突破": 0.5, "抄底": 0.5, "看涨": 0.6, "牛市": 0.7,
        "买入": 0.5, "加仓": 0.4, "持有": 0.3,
    }
    
    # 看跌关键词
    BEARISH_KEYWORDS = {
        "bearish": 0.8, "dump": 0.7, "sell": 0.5, "crash": 0.8,
        "short": 0.5, "rejection": 0.4, "breakdown": 0.6, "resistance": 0.3,
        "顶": 0.5, "跌": 0.6, "破位": 0.6, "看跌": 0.6, "熊市": 0.7,
        "卖出": 0.5, "减仓": 0.3, "止损": 0.4, "割肉": 0.5,
    }
    
    # 加密货币关键词
    CRYPTO_KEYWORDS = ["eth", "ethereum", "btc", "bitcoin", "defi", "altcoin", "crypto"]
    
    def analyze(self, text: str) -> float:
        """
        分析文本情绪
        返回: -1 (极度看跌) 到 1 (极度看涨)
        """
        text_lower = text.lower()
        
        bullish_score = 0.0
        bearish_score = 0.0
        
        # 计算看涨分数
        for keyword, weight in self.BULLISH_KEYWORDS.items():
            if keyword in text_lower:
                bullish_score += weight
        
        # 计算看跌分数
        for keyword, weight in self.BEARISH_KEYWORDS.items():
            if keyword in text_lower:
                bearish_score += weight
        
        # 归一化到 [-1, 1]
        total = bullish_score + bearish_score
        if total == 0:
            return 0.0
        
        sentiment = (bullish_score - bearish_score) / max(total, 1)
        return max(-1, min(1, sentiment))
    
    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        text_lower = text.lower()
        found = []
        
        for kw in self.CRYPTO_KEYWORDS:
            if kw in text_lower:
                found.append(kw)
        
        # 提取价格提及
        price_pattern = r'\$[\d,]+(?:\.\d+)?k?'
        prices = re.findall(price_pattern, text)
        found.extend(prices[:3])
        
        return found[:5]


class SocialStreamCollector:
    """
    社交数据流收集器
    支持多平台数据聚合
    """
    
    def __init__(
        self,
        twitter_bearer_token: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        telegram_api_id: Optional[str] = None,
        use_simulation: bool = True
    ):
        self.twitter_token = twitter_bearer_token
        self.reddit_client_id = reddit_client_id
        self.telegram_api_id = telegram_api_id
        self.use_simulation = use_simulation
        
        self.analyzer = SentimentAnalyzer()
        
        # 缓存
        self.post_cache: List[SocialPost] = []
        self.sentiment_history: List[Dict] = []
        
        # 统计
        self.stats = {
            "total_posts": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
        }
    
    async def collect_twitter(self, keywords: List[str] = None) -> List[SocialPost]:
        """收集 Twitter 数据"""
        if self.use_simulation:
            return self._simulate_twitter_posts()
        
        if not self.twitter_token:
            return self._simulate_twitter_posts()
        
        # TODO: 实现 Twitter API v2 调用
        return self._simulate_twitter_posts()
    
    async def collect_reddit(self, subreddits: List[str] = None) -> List[SocialPost]:
        """收集 Reddit 数据"""
        if self.use_simulation:
            return self._simulate_reddit_posts()
        
        # TODO: 实现 Reddit API 调用
        return self._simulate_reddit_posts()
    
    async def collect_telegram(self, channels: List[str] = None) -> List[SocialPost]:
        """收集 Telegram 数据"""
        if self.use_simulation:
            return self._simulate_telegram_posts()
        
        # TODO: 实现 Telegram API 调用
        return self._simulate_telegram_posts()
    
    def _simulate_twitter_posts(self) -> List[SocialPost]:
        """模拟 Twitter 数据"""
        import random
        
        templates = [
            ("ETH looking bullish! Breaking resistance at $2100 🚀", "crypto_trader", 150, 0.6),
            ("Bearish divergence on ETH 4H, expecting retest of $2000", "analyst_pro", 89, -0.5),
            ("Accumulation phase continues. Long term holders not selling 💎", "whale_watch", 234, 0.4),
            ("Massive sell wall at $2070, be careful", "orderflow_bot", 67, -0.4),
            ("ETH following BTC perfectly. Next target $2200", "chart_master", 156, 0.5),
            ("Warning: whale alert! 500 ETH moved to Binance", "whale_alert", 445, -0.3),
            ("DeFi TVL increasing, bullish for ETH", "defi_research", 78, 0.4),
            ("Market looking weak, potential dump incoming", "bear_signals", 123, -0.6),
        ]
        
        posts = []
        now = datetime.now()
        
        for content, author, engagement, base_sentiment in templates:
            # 添加随机波动
            sentiment = base_sentiment + random.uniform(-0.1, 0.1)
            sentiment = max(-1, min(1, sentiment))
            
            post = SocialPost(
                platform="twitter",
                content=content,
                timestamp=now - timedelta(minutes=random.randint(1, 120)),
                author=author,
                engagement=engagement,
                sentiment_score=sentiment,
                keywords=self.analyzer.extract_keywords(content)
            )
            posts.append(post)
        
        return posts
    
    def _simulate_reddit_posts(self) -> List[SocialPost]:
        """模拟 Reddit 数据"""
        import random
        
        templates = [
            ("Daily ETH Discussion - What's your strategy?", "AutoModerator", 234, 0.0),
            ("TA Analysis: Why ETH might test $2050 before next leg up", "ta_expert", 89, 0.3),
            ("Sold half my stack, market feels toppy", "crypto_skeptic", 156, -0.4),
            ("Just bought the dip! ETH to the moon!", "diamond_hands", 345, 0.7),
            ("Funding rate is getting extreme, brace for volatility", "derivatives_trader", 67, -0.2),
            ("ETH staking rewards looking good, long term play", "eth_staker", 123, 0.4),
        ]
        
        posts = []
        now = datetime.now()
        
        for content, author, engagement, sentiment in templates:
            post = SocialPost(
                platform="reddit",
                content=content,
                timestamp=now - timedelta(minutes=random.randint(5, 240)),
                author=author,
                engagement=engagement,
                sentiment_score=sentiment + random.uniform(-0.1, 0.1),
                keywords=self.analyzer.extract_keywords(content)
            )
            posts.append(post)
        
        return posts
    
    def _simulate_telegram_posts(self) -> List[SocialPost]:
        """模拟 Telegram 数据"""
        import random
        
        templates = [
            ("🚨 ALERT: Large buy order detected on Binance", "signal_bot", 567, 0.5),
            ("ETH/USDT: Short setup with tight stop", "trading_signals", 234, -0.3),
            ("Whale accumulation continues, 3rd day in a row", "whale_tracker", 345, 0.6),
            ("Resistance at $2070 holding, might see rejection", "technical_analysis", 123, -0.2),
            ("Breaking: ETH ETF news incoming?", "crypto_news", 890, 0.4),
        ]
        
        posts = []
        now = datetime.now()
        
        for content, author, engagement, sentiment in templates:
            post = SocialPost(
                platform="telegram",
                content=content,
                timestamp=now - timedelta(minutes=random.randint(1, 60)),
                author=author,
                engagement=engagement,
                sentiment_score=sentiment + random.uniform(-0.1, 0.1),
                keywords=self.analyzer.extract_keywords(content)
            )
            posts.append(post)
        
        return posts
    
    def calculate_social_sentiment_score(
        self,
        posts: List[SocialPost],
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        计算综合社交情绪分数
        
        Returns:
            score: 0-100 分数 (50 = 中性)
            sentiment: bullish/bearish/neutral
            confidence: 置信度
        """
        if not posts:
            return {
                "score": 50,
                "sentiment": "neutral",
                "confidence": 0,
                "bullish_ratio": 0.5,
                "bearish_ratio": 0.5,
            }
        
        # 加权平均情绪
        total_weight = 0
        weighted_sentiment = 0.0
        
        for post in posts:
            # 使用互动量作为权重
            weight = 1 + (post.engagement / 100)
            weighted_sentiment += post.sentiment_score * weight
            total_weight += weight
        
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # 转换为 0-100 分数
        score = 50 + (avg_sentiment * 50)
        score = max(0, min(100, score))
        
        # 统计多空比例
        bullish = sum(1 for p in posts if p.sentiment_score > 0.1)
        bearish = sum(1 for p in posts if p.sentiment_score < -0.1)
        total = len(posts)
        
        bullish_ratio = bullish / total if total > 0 else 0.5
        bearish_ratio = bearish / total if total > 0 else 0.5
        
        # 判断情绪方向
        if avg_sentiment > 0.15:
            sentiment = "bullish"
        elif avg_sentiment < -0.15:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        # 置信度基于数据量和一致性
        confidence = min(1.0, total / 50) * (1 - abs(0.5 - bullish_ratio) * 0.5)
        
        return {
            "score": round(score, 1),
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "bullish_ratio": round(bullish_ratio, 3),
            "bearish_ratio": round(bearish_ratio, 3),
            "avg_sentiment": round(avg_sentiment, 3),
            "sample_size": total,
        }
    
    async def collect_all(self) -> List[SocialPost]:
        """收集所有平台数据"""
        twitter_posts = await self.collect_twitter()
        reddit_posts = await self.collect_reddit()
        telegram_posts = await self.collect_telegram()
        
        all_posts = twitter_posts + reddit_posts + telegram_posts
        
        # 更新缓存
        self.post_cache = all_posts[-100:]  # 保持最近100条
        self.stats["total_posts"] += len(all_posts)
        
        # 更新统计
        for post in all_posts:
            if post.sentiment_score > 0.1:
                self.stats["bullish_count"] += 1
            elif post.sentiment_score < -0.1:
                self.stats["bearish_count"] += 1
            else:
                self.stats["neutral_count"] += 1
        
        return all_posts
    
    def get_social_sentiment(self) -> Dict[str, Any]:
        """
        获取社交情绪综合报告
        """
        # 收集数据
        posts = asyncio.run(self.collect_all())
        
        # 计算情绪分数
        sentiment_result = self.calculate_social_sentiment_score(posts)
        
        # 按平台分组统计
        by_platform = {}
        for platform in ["twitter", "reddit", "telegram"]:
            platform_posts = [p for p in posts if p.platform == platform]
            if platform_posts:
                by_platform[platform] = self.calculate_social_sentiment_score(platform_posts)
        
        # 热门话题
        all_keywords = []
        for post in posts:
            all_keywords.extend(post.keywords)
        
        from collections import Counter
        keyword_counts = Counter(all_keywords).most_common(5)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_sentiment": sentiment_result,
            "by_platform": by_platform,
            "trending_keywords": keyword_counts,
            "stats": self.stats,
            "recent_posts": [
                {
                    "platform": p.platform,
                    "author": p.author,
                    "content": p.content[:100] + "..." if len(p.content) > 100 else p.content,
                    "sentiment": round(p.sentiment_score, 2),
                    "engagement": p.engagement,
                }
                for p in sorted(posts, key=lambda x: x.engagement, reverse=True)[:5]
            ]
        }


# 便捷函数
def social_sentiment_score() -> Dict[str, Any]:
    """
    获取社交情绪分数（同步接口）
    
    Returns:
        社交情绪综合报告
    """
    collector = SocialStreamCollector(use_simulation=True)
    return collector.get_social_sentiment()
