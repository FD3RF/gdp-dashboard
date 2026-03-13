# explain/signal_history.py
"""
模块 43: 历史信号效能追踪 + 三重标签法
=====================================
使用 Triple Barrier Method 正确评估信号成功/失败
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignalRecord:
    """信号记录 - 三重标签法"""
    # 基本信息
    signal_id: str
    timestamp: datetime
    symbol: str
    signal: str  # LONG/SHORT/HOLD
    entry_price: float
    confidence: float
    
    # 三重边界
    take_profit: float  # 上边界 - 止盈
    stop_loss: float    # 下边界 - 止损
    max_hold_minutes: int = 60  # 时间边界 - 最大持仓时间
    
    # 评估结果
    result: Optional[str] = None  # WIN/LOSS/NEUTRAL/PENDING
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "TP"/"SL"/"TIME"/"MANUAL"
    
    # 盈亏
    pnl_percent: Optional[float] = None
    pnl_amount: Optional[float] = None
    r_multiple: Optional[float] = None  # R倍数 (盈利/R风险)
    
    # 额外信息
    market_context: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class TripleBarrierResult:
    """三重标签评估结果"""
    touched: bool
    touch_type: str  # "TP", "SL", "TIME", "NONE"
    exit_price: float
    exit_time: datetime
    pnl_percent: float
    result: str  # "WIN", "LOSS", "NEUTRAL"


@dataclass
class PerformanceStats:
    """性能统计"""
    total_signals: int
    completed_signals: int
    pending_signals: int
    wins: int
    losses: int
    neutrals: int
    win_rate: float
    profit_factor: float
    expectancy: float  # 期望值
    avg_r_multiple: float  # 平均R倍数
    avg_win_percent: float
    avg_loss_percent: float
    avg_hold_minutes: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    sharpe_ratio: float


class TripleBarrierMethod:
    """
    三重标签法 (Triple Barrier Method)
    
    正确的信号评估方式：
    1. 上边界 (TP) - 止盈触发 → WIN
    2. 下边界 (SL) - 止损触发 → LOSS
    3. 时间边界 - 超时未触发 → NEUTRAL
    
    而不是简单看下一根K线的涨跌
    """
    
    def __init__(self):
        self.price_history: Dict[str, List[Dict]] = {}  # symbol -> [{time, price}]
    
    def add_price_point(self, symbol: str, price: float, timestamp: datetime) -> None:
        """添加价格点用于评估"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # 保留最近1000个点
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def evaluate_signal(
        self,
        signal: SignalRecord,
        price_data: List[Dict]
    ) -> TripleBarrierResult:
        """
        评估信号 - 三重标签法核心逻辑
        
        Args:
            signal: 信号记录
            price_data: 价格数据 [{price, timestamp}, ...]
        
        Returns:
            TripleBarrierResult
        """
        entry_time = signal.timestamp
        entry_price = signal.entry_price
        tp = signal.take_profit
        sl = signal.stop_loss
        max_time = entry_time + timedelta(minutes=signal.max_hold_minutes)
        
        is_long = signal.signal == "LONG"
        
        # 遍历价格数据
        for point in price_data:
            point_time = point['timestamp']
            point_price = point['price']
            
            # 跳过入场前的数据
            if point_time <= entry_time:
                continue
            
            # 检查时间边界
            if point_time > max_time:
                # 超时，使用最后价格
                last_price = price_data[-1]['price'] if price_data else entry_price
                pnl = self._calculate_pnl(is_long, entry_price, last_price)
                return TripleBarrierResult(
                    touched=True,
                    touch_type="TIME",
                    exit_price=last_price,
                    exit_time=max_time,
                    pnl_percent=pnl,
                    result="NEUTRAL"
                )
            
            # 检查止盈 (LONG: price >= TP, SHORT: price <= TP)
            if is_long:
                if tp > 0 and point_price >= tp:
                    pnl = self._calculate_pnl(is_long, entry_price, tp)
                    return TripleBarrierResult(
                        touched=True,
                        touch_type="TP",
                        exit_price=tp,
                        exit_time=point_time,
                        pnl_percent=pnl,
                        result="WIN"
                    )
                if sl > 0 and point_price <= sl:
                    pnl = self._calculate_pnl(is_long, entry_price, sl)
                    return TripleBarrierResult(
                        touched=True,
                        touch_type="SL",
                        exit_price=sl,
                        exit_time=point_time,
                        pnl_percent=pnl,
                        result="LOSS"
                    )
            else:  # SHORT
                if tp > 0 and point_price <= tp:
                    pnl = self._calculate_pnl(is_long, entry_price, tp)
                    return TripleBarrierResult(
                        touched=True,
                        touch_type="TP",
                        exit_price=tp,
                        exit_time=point_time,
                        pnl_percent=pnl,
                        result="WIN"
                    )
                if sl > 0 and point_price >= sl:
                    pnl = self._calculate_pnl(is_long, entry_price, sl)
                    return TripleBarrierResult(
                        touched=True,
                        touch_type="SL",
                        exit_price=sl,
                        exit_time=point_time,
                        pnl_percent=pnl,
                        result="LOSS"
                    )
        
        # 未触发任何边界，仍在持仓中
        last_price = price_data[-1]['price'] if price_data else entry_price
        last_time = price_data[-1]['timestamp'] if price_data else entry_time
        pnl = self._calculate_pnl(is_long, entry_price, last_price)
        
        return TripleBarrierResult(
            touched=False,
            touch_type="NONE",
            exit_price=last_price,
            exit_time=last_time,
            pnl_percent=pnl,
            result="PENDING"
        )
    
    def _calculate_pnl(self, is_long: bool, entry: float, exit_price: float) -> float:
        """计算盈亏百分比"""
        if entry == 0:
            return 0
        if is_long:
            return (exit_price - entry) / entry * 100
        else:
            return (entry - exit_price) / entry * 100
    
    def calculate_r_multiple(
        self,
        signal: SignalRecord,
        pnl_percent: float
    ) -> float:
        """
        计算R倍数
        
        R = 实际盈亏 / 风险金额
        
        例如：
        - 入场价 100, 止损 98, 风险 = 2%
        - 实际盈利 4%, R = 4/2 = 2R
        """
        if signal.stop_loss == 0 or signal.entry_price == 0:
            return 0
        
        risk_percent = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
        
        if risk_percent == 0:
            return 0
        
        return pnl_percent / risk_percent


class SignalHistoryTracker:
    """
    信号历史追踪器 - 使用三重标签法
    
    核心改进：
    1. 不再简单看下一根K线涨跌
    2. 使用 TP/SL/时间边界 正确评估
    3. 计算 R倍数 和期望值
    """
    
    def __init__(
        self,
        max_history: int = 500,
        storage_path: str = ".signal_history.json",
        default_hold_minutes: int = 60
    ):
        self.max_history = max_history
        self.storage_path = storage_path
        self.default_hold_minutes = default_hold_minutes
        
        self._history: List[SignalRecord] = []
        self._triple_barrier = TripleBarrierMethod()
        self._signal_counter = 0
        
        self._load_history()
    
    def _load_history(self) -> None:
        """加载历史记录"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                # 兼容数组格式和对象格式
                if isinstance(data, list):
                    items = data
                else:
                    items = data.get("signals", data.get("history", []))
                
                for item in items:
                    try:
                        record = SignalRecord(
                            signal_id=item.get('signal_id', f"sig_{len(self._history)}"),
                            timestamp=datetime.fromisoformat(item['timestamp']) if isinstance(item.get('timestamp'), str) else item.get('timestamp', datetime.now()),
                            symbol=item.get('symbol', 'ETH/USDT'),
                            signal=item.get('signal', 'HOLD'),
                            entry_price=float(item.get('price', item.get('entry_price', 0))),
                            confidence=float(item.get('confidence', 0)),
                            take_profit=float(item.get('take_profit', 0)),
                            stop_loss=float(item.get('stop_loss', 0)),
                            max_hold_minutes=int(item.get('max_hold_minutes', 60)),
                            result=item.get('result'),
                            exit_price=float(item.get('exit_price')) if item.get('exit_price') else None,
                            exit_time=datetime.fromisoformat(item['exit_time']) if item.get('exit_time') else None,
                            exit_reason=item.get('exit_reason'),
                            pnl_percent=float(item.get('pnl_percent')) if item.get('pnl_percent') else None,
                            pnl_amount=float(item.get('pnl_amount')) if item.get('pnl_amount') else None,
                            r_multiple=float(item.get('r_multiple')) if item.get('r_multiple') else None,
                            market_context=item.get('market_context', {}),
                        )
                        self._history.append(record)
                    except Exception as e:
                        logger.debug(f"Skip invalid record: {e}")
                        continue
                
                self._signal_counter = len(self._history)
                logger.info(f"Loaded {len(self._history)} signal records")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
    
    def _save_history(self) -> None:
        """保存历史记录"""
        try:
            data = {
                "signals": [
                    {
                        'signal_id': r.signal_id,
                        'timestamp': r.timestamp.isoformat(),
                        'symbol': r.symbol,
                        'signal': r.signal,
                        'entry_price': r.entry_price,
                        'confidence': r.confidence,
                        'take_profit': r.take_profit,
                        'stop_loss': r.stop_loss,
                        'max_hold_minutes': r.max_hold_minutes,
                        'result': r.result,
                        'exit_price': r.exit_price,
                        'exit_time': r.exit_time.isoformat() if r.exit_time else None,
                        'exit_reason': r.exit_reason,
                        'pnl_percent': r.pnl_percent,
                        'pnl_amount': r.pnl_amount,
                        'r_multiple': r.r_multiple,
                        'market_context': r.market_context,
                    }
                    for r in self._history[-self.max_history:]
                ],
                "last_updated": datetime.now().isoformat(),
                "stats": self._calculate_summary_stats()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")
    
    def record_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        confidence: float,
        stop_loss: float = 0,
        take_profit: float = 0,
        max_hold_minutes: int = None,
        market_context: Dict = None
    ) -> SignalRecord:
        """
        记录新信号
        
        Args:
            symbol: 交易对
            signal: 信号类型 (LONG/SHORT/HOLD)
            price: 入场价格
            confidence: 信心指数
            stop_loss: 止损价格
            take_profit: 止盈价格
            max_hold_minutes: 最大持仓时间（分钟）
            market_context: 市场环境上下文
        
        Returns:
            SignalRecord
        """
        self._signal_counter += 1
        
        # 自动计算 TP/SL 如果未提供
        if signal in ["LONG", "SHORT"] and (stop_loss == 0 or take_profit == 0):
            sl, tp = self._auto_calculate_tp_sl(signal, price)
            if stop_loss == 0:
                stop_loss = sl
            if take_profit == 0:
                take_profit = tp
        
        record = SignalRecord(
            signal_id=f"sig_{self._signal_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            symbol=symbol,
            signal=signal,
            entry_price=price,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_minutes=max_hold_minutes or self.default_hold_minutes,
            market_context=market_context or {},
        )
        
        self._history.append(record)
        
        # 保持最大长度
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        
        self._save_history()
        
        logger.info(f"Recorded signal: {signal} @ {price}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
        
        return record
    
    def _auto_calculate_tp_sl(
        self,
        signal: str,
        price: float,
        risk_percent: float = 0.01,
        reward_ratio: float = 1.5
    ) -> Tuple[float, float]:
        """
        自动计算止损止盈
        
        Args:
            signal: 信号类型
            price: 入场价格
            risk_percent: 风险百分比 (默认1%)
            reward_ratio: 盈亏比 (默认1.5)
        
        Returns:
            (stop_loss, take_profit)
        """
        risk_amount = price * risk_percent
        reward_amount = risk_amount * reward_ratio
        
        if signal == "LONG":
            stop_loss = price - risk_amount
            take_profit = price + reward_amount
        else:  # SHORT
            stop_loss = price + risk_amount
            take_profit = price - reward_amount
        
        return stop_loss, take_profit
    
    def update_with_price_data(
        self,
        symbol: str,
        price_data: List[Dict]
    ) -> List[SignalRecord]:
        """
        使用价格数据更新所有待评估信号
        
        Args:
            symbol: 交易对
            price_data: 价格数据 [{price, timestamp}, ...]
        
        Returns:
            更新的信号列表
        """
        updated = []
        
        for record in self._history:
            if record.symbol != symbol:
                continue
            
            if record.result is not None:
                continue  # 已有结果
            
            # 使用三重标签法评估
            result = self._triple_barrier.evaluate_signal(record, price_data)
            
            if result.touched:
                record.result = result.result
                record.exit_price = result.exit_price
                record.exit_time = result.exit_time
                record.exit_reason = result.touch_type
                record.pnl_percent = result.pnl_percent
                record.r_multiple = self._triple_barrier.calculate_r_multiple(
                    record, result.pnl_percent
                )
                updated.append(record)
                
                logger.info(
                    f"Signal {record.signal_id} evaluated: {result.result} "
                    f"({result.touch_type}) PnL={result.pnl_percent:.2f}% "
                    f"R={record.r_multiple:.2f}"
                )
        
        if updated:
            self._save_history()
        
        return updated
    
    def evaluate_pending_signals(
        self,
        current_price: float,
        current_time: datetime = None
    ) -> List[SignalRecord]:
        """
        评估所有待处理信号（使用当前价格）
        
        对于开发/测试环境，使用模拟数据评估
        """
        current_time = current_time or datetime.now()
        updated = []
        
        for record in self._history:
            if record.result is not None:
                continue
            
            # 生成模拟价格路径
            simulated_prices = self._generate_simulated_path(
                record.entry_price,
                current_price,
                record.timestamp,
                current_time
            )
            
            # 使用三重标签法评估
            result = self._triple_barrier.evaluate_signal(record, simulated_prices)
            
            if result.touched:
                record.result = result.result
                record.exit_price = result.exit_price
                record.exit_time = result.exit_time
                record.exit_reason = result.touch_type
                record.pnl_percent = result.pnl_percent
                record.r_multiple = self._triple_barrier.calculate_r_multiple(
                    record, result.pnl_percent
                )
                updated.append(record)
        
        if updated:
            self._save_history()
        
        return updated
    
    def _generate_simulated_path(
        self,
        start_price: float,
        end_price: float,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """生成模拟价格路径（用于评估）"""
        import random
        
        minutes = int((end_time - start_time).total_seconds() / 60)
        if minutes <= 0:
            return [{'price': end_price, 'timestamp': end_time}]
        
        # 生成价格路径
        prices = []
        price = start_price
        dt = start_time
        
        # 计算总变化
        total_change = end_price - start_price
        
        for i in range(minutes):
            # 添加一些随机波动
            noise = random.uniform(-0.001, 0.001) * price
            trend = total_change / minutes if minutes > 0 else 0
            
            price = price + trend + noise
            dt = dt + timedelta(minutes=1)
            
            prices.append({
                'price': price,
                'timestamp': dt
            })
        
        # 确保最后价格匹配
        if prices:
            prices[-1]['price'] = end_price
        
        return prices
    
    def _calculate_summary_stats(self) -> Dict:
        """计算汇总统计"""
        completed = [r for r in self._history if r.result in ["WIN", "LOSS", "NEUTRAL"]]
        
        wins = [r for r in completed if r.result == "WIN"]
        losses = [r for r in completed if r.result == "LOSS"]
        
        total = len(completed)
        win_count = len(wins)
        loss_count = len(losses)
        
        win_rate = win_count / total if total > 0 else 0
        
        avg_win = np.mean([r.pnl_percent for r in wins]) if wins else 0
        avg_loss = np.mean([abs(r.pnl_percent) for r in losses]) if losses else 0
        
        total_win = sum(r.pnl_percent for r in wins)
        total_loss = sum(abs(r.pnl_percent) for r in losses)
        
        profit_factor = total_win / (total_loss + 1e-8)
        
        # 期望值
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # 平均R倍数
        r_multiples = [r.r_multiple for r in completed if r.r_multiple is not None]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        
        return {
            "total": len(self._history),
            "completed": total,
            "win_rate": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(expectancy, 4),
            "avg_r_multiple": round(avg_r, 2),
        }
    
    def get_stats(self, symbol: Optional[str] = None, limit: int = 100) -> PerformanceStats:
        """获取性能统计"""
        records = self._history[-limit:]
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        
        completed = [r for r in records if r.result in ["WIN", "LOSS", "NEUTRAL"]]
        pending = [r for r in records if r.result == "PENDING" or r.result is None]
        
        wins = [r for r in completed if r.result == "WIN"]
        losses = [r for r in completed if r.result == "LOSS"]
        neutrals = [r for r in completed if r.result == "NEUTRAL"]
        
        total = len(completed)
        win_count = len(wins)
        loss_count = len(losses)
        neutral_count = len(neutrals)
        
        win_rate = win_count / total if total > 0 else 0
        
        avg_win = np.mean([r.pnl_percent for r in wins]) if wins else 0
        avg_loss = np.mean([abs(r.pnl_percent) for r in losses]) if losses else 0
        
        total_win = sum(r.pnl_percent for r in wins)
        total_loss = sum(abs(r.pnl_percent) for r in losses)
        
        profit_factor = total_win / (total_loss + 1e-8)
        
        # 期望值
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # 平均R倍数
        r_multiples = [r.r_multiple for r in completed if r.r_multiple is not None]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        
        # 平均持仓时间
        hold_times = []
        for r in completed:
            if r.exit_time and r.timestamp:
                hold_times.append((r.exit_time - r.timestamp).total_seconds() / 60)
        avg_hold = np.mean(hold_times) if hold_times else 0
        
        # 最佳/最差交易
        all_pnl = [r.pnl_percent for r in completed if r.pnl_percent is not None]
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
            elif r.result == "LOSS":
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)
        
        # Sharpe比率 (简化版)
        all_returns = [r.pnl_percent / 100 for r in completed if r.pnl_percent is not None]
        if len(all_returns) > 1:
            sharpe = np.mean(all_returns) / (np.std(all_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0
        
        return PerformanceStats(
            total_signals=len(records),
            completed_signals=total,
            pending_signals=len(pending),
            wins=win_count,
            losses=loss_count,
            neutrals=neutral_count,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 4),
            avg_r_multiple=round(avg_r, 2),
            avg_win_percent=round(avg_win, 4),
            avg_loss_percent=round(avg_loss, 4),
            avg_hold_minutes=round(avg_hold, 1),
            best_trade=round(best, 4),
            worst_trade=round(worst, 4),
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            sharpe_ratio=round(sharpe, 2)
        )
    
    def get_reliability(self, symbol: str = None) -> Dict[str, Any]:
        """获取信号可靠度"""
        stats = self.get_stats(symbol)
        
        # 计算可靠度分数 (0-100)
        if stats.completed_signals < 5:
            score = 50  # 数据不足，默认50分
            level = "数据不足"
        else:
            # 综合评分
            win_rate_score = stats.win_rate * 40  # 最高40分
            pf_score = min(stats.profit_factor, 3) / 3 * 30  # 最高30分
            r_score = min(stats.avg_r_multiple, 2) / 2 * 20  # 最高20分
            consistency_score = (1 - abs(0.5 - stats.win_rate)) * 10  # 最高10分
            
            score = win_rate_score + pf_score + r_score + consistency_score
            score = min(100, max(0, score))
            
            if score >= 80:
                level = "极高可靠"
            elif score >= 60:
                level = "高可靠"
            elif score >= 40:
                level = "中等可靠"
            else:
                level = "低可靠"
        
        return {
            "score": round(score, 0),
            "level": level,
            "sample_size": stats.completed_signals,
            "win_rate": round(stats.win_rate * 100, 1),
            "profit_factor": stats.profit_factor,
            "expectancy": stats.expectancy,
            "avg_r_multiple": stats.avg_r_multiple,
        }


# 全局实例
_tracker: Optional[SignalHistoryTracker] = None


def get_tracker() -> SignalHistoryTracker:
    """获取全局追踪器"""
    global _tracker
    if _tracker is None:
        _tracker = SignalHistoryTracker()
    return _tracker


def record_signal(
    symbol: str,
    signal: str,
    price: float,
    confidence: float,
    stop_loss: float = 0,
    take_profit: float = 0
) -> SignalRecord:
    """记录信号（便捷函数）"""
    return get_tracker().record_signal(
        symbol, signal, price, confidence, stop_loss, take_profit
    )


def get_signal_reliability(symbol: str = None) -> Dict[str, Any]:
    """获取信号可靠度（便捷函数）"""
    return get_tracker().get_reliability(symbol)


def evaluate_signals(current_price: float) -> List[SignalRecord]:
    """评估待处理信号（便捷函数）"""
    return get_tracker().evaluate_pending_signals(current_price)
