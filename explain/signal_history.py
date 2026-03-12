# explain/signal_history.py
"""
模块 43: 历史信号效能追踪
=========================
展示过去N次信号的胜率与盈亏比，验证AI可靠度
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class SignalRecord:
    """信号记录"""
    timestamp: datetime
    symbol: str
    signal: str  # LONG/SHORT/HOLD
    price: float
    confidence: float
    stop_loss: float
    take_profit: float
    
    # 结果 (事后填充)
    result: Optional[str] = None  # WIN/LOSS/PENDING
    pnl_percent: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None


@dataclass
class PerformanceStats:
    """性能统计"""
    total_signals: int
    completed_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_rr_ratio: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int


class SignalHistoryTracker:
    """
    信号历史追踪器
    
    功能：
    - 记录所有历史信号
    - 计算胜率和盈亏比
    - 追踪信号可靠度
    """
    
    def __init__(self, max_history: int = 100, storage_path: str = ".signal_history.json"):
        """
        Args:
            max_history: 最大历史记录数
            storage_path: 存储路径
        """
        self.max_history = max_history
        self.storage_path = storage_path
        self._history: List[SignalRecord] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """加载历史记录"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for item in data[-self.max_history:]:
                        record = SignalRecord(
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            symbol=item['symbol'],
                            signal=item['signal'],
                            price=item['price'],
                            confidence=item['confidence'],
                            stop_loss=item.get('stop_loss', 0),
                            take_profit=item.get('take_profit', 0),
                            result=item.get('result'),
                            pnl_percent=item.get('pnl_percent'),
                            exit_price=item.get('exit_price'),
                            exit_time=datetime.fromisoformat(item['exit_time']) if item.get('exit_time') else None
                        )
                        self._history.append(record)
            except Exception:
                pass
    
    def _save_history(self) -> None:
        """保存历史记录"""
        try:
            data = []
            for record in self._history[-self.max_history:]:
                data.append({
                    'timestamp': record.timestamp.isoformat(),
                    'symbol': record.symbol,
                    'signal': record.signal,
                    'price': record.price,
                    'confidence': record.confidence,
                    'stop_loss': record.stop_loss,
                    'take_profit': record.take_profit,
                    'result': record.result,
                    'pnl_percent': record.pnl_percent,
                    'exit_price': record.exit_price,
                    'exit_time': record.exit_time.isoformat() if record.exit_time else None
                })
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def record_signal(self, 
                      symbol: str,
                      signal: str,
                      price: float,
                      confidence: float,
                      stop_loss: float = 0,
                      take_profit: float = 0) -> None:
        """
        记录新信号
        
        Args:
            symbol: 交易对
            signal: 信号类型
            price: 价格
            confidence: 信心指数
            stop_loss: 止损价
            take_profit: 止盈价
        """
        record = SignalRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            signal=signal,
            price=price,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self._history.append(record)
        
        # 保持最大长度
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        
        self._save_history()
    
    def update_result(self, 
                      symbol: str,
                      entry_price: float,
                      exit_price: float,
                      result: str) -> None:
        """
        更新信号结果
        
        Args:
            symbol: 交易对
            entry_price: 入场价
            exit_price: 出场价
            result: 结果 (WIN/LOSS)
        """
        # 找到最近的匹配记录
        for record in reversed(self._history):
            if (record.symbol == symbol and 
                abs(record.price - entry_price) < entry_price * 0.01 and
                record.result is None):
                
                record.result = result
                record.exit_price = exit_price
                record.exit_time = datetime.now()
                
                # 计算盈亏
                if record.signal == "LONG":
                    record.pnl_percent = (exit_price - entry_price) / entry_price * 100
                elif record.signal == "SHORT":
                    record.pnl_percent = (entry_price - exit_price) / entry_price * 100
                else:
                    record.pnl_percent = 0
                
                break
        
        self._save_history()
    
    def get_stats(self, symbol: Optional[str] = None, limit: int = 100) -> PerformanceStats:
        """
        获取性能统计
        
        Args:
            symbol: 交易对 (None表示所有)
            limit: 限制记录数
            
        Returns:
            PerformanceStats
        """
        # 过滤记录
        records = self._history[-limit:]
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        
        # 已完成的交易
        completed = [r for r in records if r.result is not None]
        
        wins = [r for r in completed if r.result == "WIN"]
        losses = [r for r in completed if r.result == "LOSS"]
        
        total = len(completed)
        win_count = len(wins)
        loss_count = len(losses)
        
        win_rate = win_count / total if total > 0 else 0
        
        avg_win = np.mean([r.pnl_percent for r in wins]) if wins else 0
        avg_loss = np.mean([abs(r.pnl_percent) for r in losses]) if losses else 0
        
        profit_factor = (avg_win * win_count) / (avg_loss * loss_count + 1e-8)
        
        # 盈亏比
        avg_rr = avg_win / (avg_loss + 1e-8)
        
        # 最佳/最差交易
        all_pnl = [r.pnl_percent for r in completed if r.pnl_percent]
        best = max(all_pnl) if all_pnl else 0
        worst = min(all_pnl) if all_pnl else 0
        
        # 连胜/连亏
        consecutive_wins = 0
        consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for r in completed:
            if r.result == "WIN":
                current_wins += 1
                current_losses = 0
                consecutive_wins = max(consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)
        
        return PerformanceStats(
            total_signals=len(records),
            completed_signals=total,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_rr_ratio=avg_rr,
            best_trade=best,
            worst_trade=worst,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses
        )
    
    def get_reliability_score(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        获取可靠度评分
        
        Args:
            symbol: 交易对
            
        Returns:
            可靠度信息
        """
        stats = self.get_stats(symbol)
        
        # 计算可靠度评分 (0-100)
        if stats.completed_signals < 10:
            reliability = 50  # 样本不足
            reliability_level = "数据不足"
        else:
            # 胜率权重 40%
            win_score = stats.win_rate * 40
            
            # 盈亏比权重 30%
            rr_score = min(stats.avg_rr_ratio, 3) / 3 * 30
            
            # 盈利因子权重 30%
            pf_score = min(stats.profit_factor, 3) / 3 * 30
            
            reliability = win_score + rr_score + pf_score
            
            if reliability >= 70:
                reliability_level = "高可靠度"
            elif reliability >= 50:
                reliability_level = "中等可靠度"
            else:
                reliability_level = "低可靠度"
        
        return {
            'score': reliability,
            'level': reliability_level,
            'win_rate': stats.win_rate * 100,
            'profit_factor': stats.profit_factor,
            'avg_rr_ratio': stats.avg_rr_ratio,
            'sample_size': stats.completed_signals,
            'stats': {
                'total': stats.total_signals,
                'completed': stats.completed_signals,
                'wins': stats.wins,
                'losses': stats.losses
            }
        }


# 全局追踪器
_tracker: Optional[SignalHistoryTracker] = None


def get_tracker() -> SignalHistoryTracker:
    """获取全局追踪器"""
    global _tracker
    if _tracker is None:
        _tracker = SignalHistoryTracker()
    return _tracker


def record_signal(symbol: str, signal: str, price: float, 
                  confidence: float, stop_loss: float = 0, take_profit: float = 0) -> None:
    """快速记录信号"""
    get_tracker().record_signal(symbol, signal, price, confidence, stop_loss, take_profit)


def get_signal_reliability(symbol: str = None) -> Dict[str, Any]:
    """快速获取信号可靠度"""
    return get_tracker().get_reliability_score(symbol)
