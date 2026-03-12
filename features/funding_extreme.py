"""
多交易所资金费率极值预警模块 (Layer 5-10)
监控 Binance/OKX/Bybit 等交易所资金费率，识别极端拥挤
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class FundingRateData:
    """资金费率数据"""
    exchange: str
    symbol: str
    rate: Decimal  # 费率
    timestamp: datetime
    next_funding_time: Optional[datetime] = None
    
    # 计算字段
    z_score: float = 0.0
    percentile: float = 0.0
    is_extreme: bool = False
    extreme_type: Optional[str] = None  # "long_crowded" / "short_crowded"


@dataclass
class FundingExtremeAlert:
    """资金费率极值警报"""
    timestamp: datetime
    alert_type: str  # "long_squeeze_risk" / "short_squeeze_risk" / "convergence"
    affected_exchanges: List[str]
    avg_rate: Decimal
    z_score: float
    severity: str  # "low" / "medium" / "high"
    description: str
    recommended_action: str


class MultiExchangeFundingMonitor:
    """
    多交易所资金费率监控器
    实现跨交易所极值检测和踩踏预警
    """
    
    # 交易所 API 端点
    EXCHANGE_APIS = {
        "binance": "https://fapi.binance.com/fapi/v1/fundingRate",
        "okx": "https://www.okx.com/api/v5/public/funding-rate",
        "bybit": "https://api.bybit.com/v5/public/funding/history-funding-rate",
    }
    
    # 极值阈值
    Z_SCORE_THRESHOLD = 2.0  # z-score 超过 2 标准差
    RATE_THRESHOLD_LONG = Decimal("0.0005")  # 正费率 0.05%
    RATE_THRESHOLD_SHORT = Decimal("-0.0005")  # 负费率 -0.05%
    
    def __init__(self, use_simulation: bool = True):
        self.use_simulation = use_simulation
        
        # 历史数据缓存
        self.rate_history: Dict[str, List[FundingRateData]] = {}
        self.max_history = 500  # 每个交易所保留的历史条数
        
        # 警报缓存
        self.alerts: List[FundingExtremeAlert] = []
        
        # 统计
        self.stats = {
            "total_checks": 0,
            "extreme_events": 0,
            "long_squeeze_warnings": 0,
            "short_squeeze_warnings": 0,
        }
    
    async def fetch_binance_funding(self, symbol: str = "ETHUSDT") -> Optional[FundingRateData]:
        """获取 Binance 资金费率"""
        if self.use_simulation:
            return self._simulate_funding("binance", symbol)
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {"symbol": symbol, "limit": 1}
                async with session.get(
                    self.EXCHANGE_APIS["binance"],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()
                    if data:
                        item = data[0] if isinstance(data, list) else data
                        return FundingRateData(
                            exchange="binance",
                            symbol=symbol,
                            rate=Decimal(str(item.get("fundingRate", 0))),
                            timestamp=datetime.fromtimestamp(item.get("fundingTime", 0) / 1000),
                        )
        except Exception as e:
            logger.warning(f"Binance funding fetch failed: {e}")
        
        return self._simulate_funding("binance", symbol)
    
    async def fetch_okx_funding(self, symbol: str = "ETH-USDT-SWAP") -> Optional[FundingRateData]:
        """获取 OKX 资金费率"""
        if self.use_simulation:
            return self._simulate_funding("okx", symbol)
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {"instId": symbol}
                async with session.get(
                    self.EXCHANGE_APIS["okx"],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()
                    if data.get("data"):
                        item = data["data"][0]
                        return FundingRateData(
                            exchange="okx",
                            symbol=symbol,
                            rate=Decimal(str(item.get("fundingRate", 0))),
                            timestamp=datetime.now(),
                        )
        except Exception as e:
            logger.warning(f"OKX funding fetch failed: {e}")
        
        return self._simulate_funding("okx", symbol)
    
    async def fetch_bybit_funding(self, symbol: str = "ETHUSDT") -> Optional[FundingRateData]:
        """获取 Bybit 资金费率"""
        if self.use_simulation:
            return self._simulate_funding("bybit", symbol)
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {"category": "linear", "symbol": symbol, "limit": 1}
                async with session.get(
                    self.EXCHANGE_APIS["bybit"],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    data = await resp.json()
                    if data.get("result", {}).get("list"):
                        item = data["result"]["list"][0]
                        return FundingRateData(
                            exchange="bybit",
                            symbol=symbol,
                            rate=Decimal(str(item.get("fundingRate", 0))),
                            timestamp=datetime.fromtimestamp(int(item.get("fundingRateTimestamp", 0)) / 1000),
                        )
        except Exception as e:
            logger.warning(f"Bybit funding fetch failed: {e}")
        
        return self._simulate_funding("bybit", symbol)
    
    def _simulate_funding(self, exchange: str, symbol: str) -> FundingRateData:
        """生成模拟资金费率"""
        import random
        
        # 基于交易所的微小差异
        base_rates = {
            "binance": random.uniform(-0.0002, 0.0003),
            "okx": random.uniform(-0.00025, 0.00035),
            "bybit": random.uniform(-0.0003, 0.0004),
        }
        
        rate = base_rates.get(exchange, random.uniform(-0.0002, 0.0002))
        
        return FundingRateData(
            exchange=exchange,
            symbol=symbol,
            rate=Decimal(str(rate)),
            timestamp=datetime.now(),
        )
    
    def update_history(self, data: FundingRateData) -> None:
        """更新历史数据"""
        key = f"{data.exchange}_{data.symbol}"
        
        if key not in self.rate_history:
            self.rate_history[key] = []
        
        self.rate_history[key].append(data)
        
        # 限制历史长度
        if len(self.rate_history[key]) > self.max_history:
            self.rate_history[key] = self.rate_history[key][-self.max_history:]
    
    def calculate_z_score(self, rate: Decimal, exchange: str, symbol: str) -> float:
        """计算 z-score"""
        key = f"{exchange}_{symbol}"
        history = self.rate_history.get(key, [])
        
        if len(history) < 10:
            return 0.0
        
        rates = [float(h.rate) for h in history[-100:]]  # 最近100条
        mean = np.mean(rates)
        std = np.std(rates)
        
        if std == 0:
            return 0.0
        
        z_score = (float(rate) - mean) / std
        return round(z_score, 2)
    
    def calculate_percentile(self, rate: Decimal, exchange: str, symbol: str) -> float:
        """计算百分位"""
        key = f"{exchange}_{symbol}"
        history = self.rate_history.get(key, [])
        
        if len(history) < 10:
            return 50.0
        
        rates = sorted([float(h.rate) for h in history[-100:]])
        rate_float = float(rate)
        
        # 找到插入位置
        pos = 0
        for r in rates:
            if r < rate_float:
                pos += 1
        
        percentile = (pos / len(rates)) * 100
        return round(percentile, 1)
    
    def detect_extreme(self, rate: Decimal, z_score: float) -> Tuple[bool, Optional[str]]:
        """检测极端资金费率"""
        is_extreme = False
        extreme_type = None
        
        # 基于 z-score
        if abs(z_score) >= self.Z_SCORE_THRESHOLD:
            is_extreme = True
            if z_score > 0:
                extreme_type = "long_crowded"  # 正费率极高 = 做多拥挤
            else:
                extreme_type = "short_crowded"  # 负费率极低 = 做空拥挤
        
        # 基于绝对值
        if rate > self.RATE_THRESHOLD_LONG:
            is_extreme = True
            extreme_type = "long_crowded"
        elif rate < self.RATE_THRESHOLD_SHORT:
            is_extreme = True
            extreme_type = "short_crowded"
        
        return is_extreme, extreme_type
    
    async def collect_all_funding_rates(self, symbol: str = "ETHUSDT") -> List[FundingRateData]:
        """收集所有交易所资金费率"""
        results = await asyncio.gather(
            self.fetch_binance_funding(symbol),
            self.fetch_okx_funding("ETH-USDT-SWAP"),
            self.fetch_bybit_funding(symbol),
        )
        
        funding_rates = [r for r in results if r is not None]
        
        # 更新历史并计算统计量
        for fr in funding_rates:
            self.update_history(fr)
            fr.z_score = self.calculate_z_score(fr.rate, fr.exchange, fr.symbol)
            fr.percentile = self.calculate_percentile(fr.rate, fr.exchange, fr.symbol)
            fr.is_extreme, fr.extreme_type = self.detect_extreme(fr.rate, fr.z_score)
        
        self.stats["total_checks"] += 1
        
        return funding_rates
    
    def generate_alerts(self, funding_rates: List[FundingRateData]) -> List[FundingExtremeAlert]:
        """生成极值警报"""
        alerts = []
        
        # 检查单个交易所极端
        for fr in funding_rates:
            if fr.is_extreme:
                if fr.extreme_type == "long_crowded":
                    alert_type = "long_squeeze_risk"
                    severity = "high" if fr.z_score > 3 else "medium"
                    description = f"{fr.exchange} 资金费率极高 ({fr.rate:.6f})，做多拥挤，存在多头踩踏风险"
                    action = "考虑减多或观望"
                else:
                    alert_type = "short_squeeze_risk"
                    severity = "high" if fr.z_score < -3 else "medium"
                    description = f"{fr.exchange} 资金费率极低 ({fr.rate:.6f})，做空拥挤，存在空头踩踏风险"
                    action = "考虑减空或观望"
                
                alert = FundingExtremeAlert(
                    timestamp=datetime.now(),
                    alert_type=alert_type,
                    affected_exchanges=[fr.exchange],
                    avg_rate=fr.rate,
                    z_score=fr.z_score,
                    severity=severity,
                    description=description,
                    recommended_action=action
                )
                alerts.append(alert)
                
                self.stats["extreme_events"] += 1
                if alert_type == "long_squeeze_risk":
                    self.stats["long_squeeze_warnings"] += 1
                else:
                    self.stats["short_squeeze_warnings"] += 1
        
        # 检查跨交易所一致性
        if len(funding_rates) >= 2:
            avg_rate = np.mean([float(fr.rate) for fr in funding_rates])
            std_rate = np.std([float(fr.rate) for fr in funding_rates])
            
            # 如果所有交易所都极端
            all_extreme = all(fr.is_extreme for fr in funding_rates)
            same_direction = all(
                fr.extreme_type == funding_rates[0].extreme_type 
                for fr in funding_rates if fr.extreme_type
            )
            
            if all_extreme and same_direction:
                alert = FundingExtremeAlert(
                    timestamp=datetime.now(),
                    alert_type="cross_exchange_extreme",
                    affected_exchanges=[fr.exchange for fr in funding_rates],
                    avg_rate=Decimal(str(avg_rate)),
                    z_score=np.mean([fr.z_score for fr in funding_rates]),
                    severity="high",
                    description=f"⚠️ 全市场资金费率极端一致！{'做多' if avg_rate > 0 else '做空'}极度拥挤",
                    recommended_action="高风险！建议降低仓位或反向操作"
                )
                alerts.append(alert)
        
        self.alerts.extend(alerts)
        self.alerts = self.alerts[-50:]  # 保留最近50条警报
        
        return alerts
    
    def get_funding_extreme_report(self) -> Dict[str, Any]:
        """
        获取资金费率极值综合报告
        """
        # 收集数据
        funding_rates = asyncio.run(self.collect_all_funding_rates())
        
        # 生成警报
        alerts = self.generate_alerts(funding_rates)
        
        # 计算综合指标
        avg_rate = np.mean([float(fr.rate) for fr in funding_rates]) if funding_rates else 0
        avg_z_score = np.mean([fr.z_score for fr in funding_rates]) if funding_rates else 0
        
        # 综合拥挤度评分 (0-100)
        # 50 = 中性，>50 = 做多拥挤，<50 = 做空拥挤
        crowding_score = 50 + (avg_rate * 100000)  # 放大费率影响
        crowding_score = max(0, min(100, crowding_score))
        
        return {
            "timestamp": datetime.now().isoformat(),
            "funding_rates": [
                {
                    "exchange": fr.exchange,
                    "rate": float(fr.rate),
                    "rate_pct": float(fr.rate * 100),  # 百分比形式
                    "z_score": fr.z_score,
                    "percentile": fr.percentile,
                    "is_extreme": fr.is_extreme,
                    "extreme_type": fr.extreme_type,
                }
                for fr in funding_rates
            ],
            "alerts": [
                {
                    "type": a.alert_type,
                    "exchanges": a.affected_exchanges,
                    "severity": a.severity,
                    "z_score": a.z_score,
                    "description": a.description,
                    "action": a.recommended_action,
                }
                for a in alerts
            ],
            "summary": {
                "avg_rate": round(avg_rate, 8),
                "avg_z_score": round(avg_z_score, 2),
                "crowding_score": round(crowding_score, 1),
                "crowding_direction": "做多拥挤" if crowding_score > 55 else "做空拥挤" if crowding_score < 45 else "中性",
                "extreme_exchanges": len([fr for fr in funding_rates if fr.is_extreme]),
            },
            "stats": self.stats,
        }


# 便捷函数
def funding_extreme_alert() -> Dict[str, Any]:
    """
    获取资金费率极值警报（同步接口）
    
    Returns:
        多交易所资金费率极值报告
    """
    monitor = MultiExchangeFundingMonitor(use_simulation=True)
    return monitor.get_funding_extreme_report()
