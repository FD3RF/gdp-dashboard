# agent/perception.py
"""
第2层：全息感知层
==================

功能：将杂乱的市场数据编码为 AI 能理解的 256 维"脑电波"
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class PerceptionEncoder(nn.Module):
    """
    全息感知编码器
    
    将多源异构数据压缩为 256 维向量：
    - 微观结构（订单簿）
    - 价格序列（动量、波动率）
    - 链上行为（鲸鱼活动）
    - 情绪光谱（资金费率、市场情绪）
    """
    
    def __init__(self, input_dim: int = 50, output_dim: int = 256):
        super(PerceptionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 多层 Transformer 编码器
        self.embedding = nn.Linear(input_dim, 512)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        提取原始特征
        
        Args:
            market_data: 市场数据字典
            
        Returns:
            特征向量
        """
        features = []
        
        # 1. 微观结构特征 (10维)
        ob_imbalance = market_data.get('orderbook_imbalance', 0.5)
        bid_volume = market_data.get('bid_volume', 0)
        ask_volume = market_data.get('ask_volume', 0)
        spread_pct = market_data.get('spread_pct', 0)
        trade_flow = market_data.get('trade_flow', 0)
        
        features.extend([
            ob_imbalance,
            np.log1p(bid_volume),
            np.log1p(ask_volume),
            spread_pct,
            trade_flow,
            bid_volume / (ask_volume + 1e-8),  # 买卖比
            market_data.get('large_order_ratio', 0),
            market_data.get('small_order_ratio', 0),
            market_data.get('buy_pressure', 0),
            market_data.get('sell_pressure', 0),
        ])
        
        # 2. 价格序列特征 (15维)
        features.extend([
            market_data.get('momentum_5m', 0),
            market_data.get('momentum_15m', 0),
            market_data.get('momentum_1h', 0),
            market_data.get('volatility_5m', 0),
            market_data.get('volatility_15m', 0),
            market_data.get('volatility_1h', 0),
            market_data.get('rsi_14', 50) / 100,  # 归一化
            market_data.get('macd', 0),
            market_data.get('macd_signal', 0),
            market_data.get('macd_hist', 0),
            market_data.get('bb_position', 0.5),  # 布林带位置
            market_data.get('atr_ratio', 0),
            market_data.get('price_change_5m', 0),
            market_data.get('price_change_15m', 0),
            market_data.get('price_change_1h', 0),
        ])
        
        # 3. 链上行为特征 (10维)
        features.extend([
            market_data.get('whale_activity_score', 0),
            market_data.get('exchange_inflow', 0),
            market_data.get('exchange_outflow', 0),
            market_data.get('holder_change_24h', 0),
            market_data.get('active_addresses', 0),
            market_data.get('transaction_volume', 0),
            market_data.get('nvt_ratio', 0),
            market_data.get('mvrv_ratio', 0),
            market_data.get('sopr', 0),
            market_data.get('puell_multiple', 0),
        ])
        
        # 4. 情绪光谱特征 (10维)
        features.extend([
            market_data.get('funding_rate', 0) * 1000,  # 放大
            market_data.get('sentiment_score', 0.5),
            market_data.get('fear_greed_index', 50) / 100,
            market_data.get('long_short_ratio', 1) - 1,
            market_data.get('open_interest_change', 0),
            market_data.get('liquidation_long', 0),
            market_data.get('liquidation_short', 0),
            market_data.get('social_volume', 0),
            market_data.get('news_sentiment', 0.5),
            market_data.get('twitter_sentiment', 0.5),
        ])
        
        # 5. 时间特征 (5维)
        now = market_data.get('timestamp', None)
        if now:
            import datetime
            dt = datetime.datetime.fromtimestamp(now / 1000) if isinstance(now, (int, float)) else now
            hour_sin = np.sin(2 * np.pi * dt.hour / 24)
            hour_cos = np.cos(2 * np.pi * dt.hour / 24)
            day_sin = np.sin(2 * np.pi * dt.weekday() / 7)
            day_cos = np.cos(2 * np.pi * dt.weekday() / 7)
            is_weekend = 1.0 if dt.weekday() >= 5 else 0.0
        else:
            hour_sin, hour_cos, day_sin, day_cos, is_weekend = 0, 0, 0, 0, 0
        
        features.extend([hour_sin, hour_cos, day_sin, day_cos, is_weekend])
        
        return np.array(features, dtype=np.float32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            256维状态向量 [batch_size, output_dim]
        """
        # Embedding
        x = self.embedding(x)  # [batch, 512]
        
        # Transformer
        x = x.unsqueeze(1)  # [batch, 1, 512]
        x = self.transformer(x)  # [batch, 1, 512]
        x = x.squeeze(1)  # [batch, 512]
        
        # 输出
        return self.fc_out(x)  # [batch, 256]
    
    def encode(self, market_data: Dict[str, Any]) -> torch.Tensor:
        """
        将多源数据融合为 256 维状态向量
        
        Args:
            market_data: 市场数据字典
            
        Returns:
            256维状态向量
        """
        # 提取特征
        raw_features = self.extract_features(market_data)
        
        # 归一化
        mean = np.mean(raw_features)
        std = np.std(raw_features) + 1e-8
        normalized = (raw_features - mean) / std
        
        # 填充至 input_dim 维
        padded = np.zeros(self.input_dim, dtype=np.float32)
        padded[:len(normalized)] = normalized[:self.input_dim]
        
        # 转换为张量
        state_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0)
        
        # 神经编码
        with torch.no_grad():
            state_vector = self.forward(state_tensor)
        
        return state_vector.squeeze(0)


class MarketDataCollector:
    """市场数据收集器"""
    
    def __init__(self):
        self.history = []
        self.max_history = 1000
    
    def update(self, data: Dict[str, Any]):
        """更新数据"""
        self.history.append(data)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """获取最新数据"""
        return self.history[-1] if self.history else None
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.history:
            return {}
        
        # 计算动量、波动率等
        return {
            'data_count': len(self.history),
            'last_update': self.history[-1].get('timestamp', 0),
        }


# 导出
__all__ = ['PerceptionEncoder', 'MarketDataCollector']
