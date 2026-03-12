"""
链上巨鲸监控模块 (Layer 2-4)
监控以太坊大额转账，预警潜在抛售/吸筹压力
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from decimal import Decimal
import json
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class WhaleAlert:
    """巨鲸警报数据结构"""
    timestamp: datetime
    wallet_from: str
    wallet_to: str
    amount: Decimal  # ETH
    amount_usd: Decimal
    direction: str  # 'exchange_in', 'exchange_out', 'whale_transfer'
    exchange_name: Optional[str] = None
    impact_score: float = 0.0  # 0-100 影响程度
    details: str = ""


@dataclass
class ExchangeWallets:
    """交易所钱包地址库"""
    # 主流交易所冷钱包/热钱包地址 (部分示例)
    BINANCE: List[str] = field(default_factory=lambda: [
        "0x28c6c06298d514db08993407b35580b8b8d0e1c8",  # Binance Cold
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance Hot
        "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 14
    ])
    COINBASE: List[str] = field(default_factory=lambda: [
        "0x503828976d22510aad0201ac7ec88293211d23da",  # Coinbase Prime
        "0x71c7656ec7ab88b098defb751b7401b7f3d49faf",  # Coinbase
    ])
    KRaken: List[str] = field(default_factory=lambda: [
        "0x2910543af39aba0cd09dbb2d50200b3e800a63d2",  # Kraken
    ])
    OKX: List[str] = field(default_factory=lambda: [
        "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b",  # OKX
    ])
    BYBIT: List[str] = field(default_factory=lambda: [
        "0x5a52e96bacdabb82fd05763e25335261b27002cb",  # Bybit
    ])


class WhaleMonitor:
    """
    链上巨鲸监控器
    支持多数据源：Etherscan API、Alchemy WebSocket、模拟数据
    """
    
    # 已知大户/机构地址
    KNOWN_WHALES = {
        "0x7758e507850da52a7b7c840ef3f212f92750e3be": "Grayscale",
        "0xd551234ae421e3bcba99a0da6d736074f22192ff": "FTX Estate",
        "0x73bceb1cd57c711feac4224d062b0f6ff338501e": "Jump Trading",
        "0x28c6c06298d514db08993407b35580b8b8d0e1c8": "Binance Cold",
    }
    
    # 监控阈值
    ALERT_THRESHOLD_ETH = Decimal("100")  # 100 ETH 以上触发预警
    MAJOR_THRESHOLD_ETH = Decimal("1000")  # 1000 ETH 以上为重大转账
    
    def __init__(
        self,
        etherscan_api_key: Optional[str] = None,
        alchemy_api_key: Optional[str] = None,
        use_simulation: bool = True
    ):
        self.etherscan_api_key = etherscan_api_key
        self.alchemy_api_key = alchemy_api_key
        self.use_simulation = use_simulation
        
        self.exchange_wallets = ExchangeWallets()
        self.all_exchange_addresses = self._build_exchange_address_map()
        
        # 缓存
        self.recent_alerts: List[WhaleAlert] = []
        self.alert_history: List[WhaleAlert] = []
        self._last_check_time: Optional[datetime] = None
        
        # 统计
        self.stats = {
            "total_alerts": 0,
            "exchange_inflow": Decimal("0"),
            "exchange_outflow": Decimal("0"),
            "whale_transfers": 0,
        }
    
    def _build_exchange_address_map(self) -> Dict[str, str]:
        """构建交易所地址到名称的映射"""
        address_map = {}
        for exchange, addresses in [
            ("Binance", self.exchange_wallets.BINANCE),
            ("Coinbase", self.exchange_wallets.COINBASE),
            ("Kraken", self.exchange_wallets.KRaken),  # 注意属性名
            ("OKX", self.exchange_wallets.OKX),
            ("Bybit", self.exchange_wallets.BYBIT),
        ]:
            for addr in addresses:
                address_map[addr.lower()] = exchange
        return address_map
    
    def _classify_transfer(
        self,
        from_addr: str,
        to_addr: str
    ) -> Tuple[str, Optional[str]]:
        """
        分类转账类型
        返回: (direction, exchange_name)
        """
        from_lower = from_addr.lower()
        to_lower = to_addr.lower()
        
        from_exchange = self.all_exchange_addresses.get(from_lower)
        to_exchange = self.all_exchange_addresses.get(to_lower)
        
        if to_exchange and not from_exchange:
            # 转入交易所 = 潜在抛售
            return "exchange_in", to_exchange
        elif from_exchange and not to_exchange:
            # 从交易所转出 = 潜在吸筹/提现
            return "exchange_out", from_exchange
        elif from_exchange and to_exchange:
            # 交易所间转账
            return "exchange_transfer", f"{from_exchange}->{to_exchange}"
        else:
            # 普通巨鲸转账
            return "whale_transfer", None
    
    def _calculate_impact_score(
        self,
        amount: Decimal,
        direction: str,
        eth_price: Decimal
    ) -> float:
        """计算影响分数 (0-100)"""
        amount_usd = amount * eth_price
        
        # 基础分数基于金额
        if amount >= self.MAJOR_THRESHOLD_ETH:
            base_score = 80
        elif amount >= Decimal("500"):
            base_score = 60
        elif amount >= Decimal("200"):
            base_score = 40
        else:
            base_score = 20
        
        # 方向调整
        if direction == "exchange_in":
            # 转入交易所 = 卖压，影响更大
            multiplier = 1.3
        elif direction == "exchange_out":
            # 转出 = 看涨信号
            multiplier = 1.1
        else:
            multiplier = 1.0
        
        score = min(100, base_score * multiplier)
        return round(score, 1)
    
    async def fetch_whale_transactions(
        self,
        eth_price: Decimal,
        limit: int = 20
    ) -> List[WhaleAlert]:
        """
        获取巨鲸交易
        优先级：Etherscan API > Alchemy WebSocket > 模拟数据
        """
        if self.use_simulation:
            return self._generate_simulated_alerts(eth_price, limit)
        
        # 尝试 Etherscan API
        if self.etherscan_api_key:
            try:
                return await self._fetch_from_etherscan(eth_price, limit)
            except Exception as e:
                logger.warning(f"Etherscan API failed: {e}")
        
        # 降级到模拟数据
        return self._generate_simulated_alerts(eth_price, limit)
    
    async def _fetch_from_etherscan(
        self,
        eth_price: Decimal,
        limit: int
    ) -> List[WhaleAlert]:
        """从 Etherscan API 获取大额交易"""
        url = "https://api.etherscan.io/api"
        params = {
            "module": "account",
            "action": "txlist",
            "address": "0x28c6c06298d514db08993407b35580b8b8d0e1c8",  # Binance Cold
            "startblock": "0",
            "endblock": "99999999",
            "page": 1,
            "offset": limit,
            "sort": "desc",
            "apikey": self.etherscan_api_key
        }
        
        alerts = []
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                
                if data.get("status") == "1":
                    for tx in data.get("result", []):
                        value_wei = int(tx.get("value", 0))
                        amount_eth = Decimal(value_wei) / Decimal(1e18)
                        
                        if amount_eth >= self.ALERT_THRESHOLD_ETH:
                            direction, exchange = self._classify_transfer(
                                tx["from"], tx["to"]
                            )
                            
                            alert = WhaleAlert(
                                timestamp=datetime.fromtimestamp(int(tx["timeStamp"])),
                                wallet_from=tx["from"],
                                wallet_to=tx["to"],
                                amount=amount_eth,
                                amount_usd=amount_eth * eth_price,
                                direction=direction,
                                exchange_name=exchange,
                                impact_score=self._calculate_impact_score(
                                    amount_eth, direction, eth_price
                                )
                            )
                            alerts.append(alert)
        
        return alerts
    
    def _generate_simulated_alerts(
        self,
        eth_price: Decimal,
        limit: int
    ) -> List[WhaleAlert]:
        """生成模拟巨鲸警报（开发/测试用）"""
        import random
        
        alerts = []
        now = datetime.now()
        
        # 模拟最近几笔大额转账
        scenarios = [
            ("exchange_in", "Binance", Decimal(str(random.uniform(150, 800)))),
            ("exchange_out", "Coinbase", Decimal(str(random.uniform(200, 600)))),
            ("whale_transfer", None, Decimal(str(random.uniform(100, 400)))),
            ("exchange_in", "Kraken", Decimal(str(random.uniform(120, 350)))),
            ("exchange_out", "OKX", Decimal(str(random.uniform(180, 500)))),
        ]
        
        for i, (direction, exchange, amount) in enumerate(scenarios[:limit]):
            alert = WhaleAlert(
                timestamp=now - timedelta(minutes=random.randint(5, 120)),
                wallet_from=f"0x{''.join(random.choices('abcdef0123456789', k=40))}",
                wallet_to=f"0x{''.join(random.choices('abcdef0123456789', k=40))}",
                amount=amount,
                amount_usd=amount * eth_price,
                direction=direction,
                exchange_name=exchange,
                impact_score=self._calculate_impact_score(amount, direction, eth_price),
                details=self._generate_details(direction, exchange, amount)
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_details(
        self,
        direction: str,
        exchange: Optional[str],
        amount: Decimal
    ) -> str:
        """生成警报详情描述"""
        if direction == "exchange_in":
            return f"⚠️ 巨鲸转入 {exchange}: {amount:.2f} ETH，潜在抛压"
        elif direction == "exchange_out":
            return f"🟢 巨鲸从 {exchange} 转出: {amount:.2f} ETH，看涨信号"
        else:
            return f"🐋 巨鲸转账: {amount:.2f} ETH"
    
    def update_stats(self, alerts: List[WhaleAlert]) -> None:
        """更新统计数据"""
        for alert in alerts:
            self.stats["total_alerts"] += 1
            
            if alert.direction == "exchange_in":
                self.stats["exchange_inflow"] += alert.amount
            elif alert.direction == "exchange_out":
                self.stats["exchange_outflow"] += alert.amount
            else:
                self.stats["whale_transfers"] += 1
        
        self.alert_history.extend(alerts)
        # 保持最近100条记录
        self.alert_history = self.alert_history[-100:]
    
    def get_exchange_flow_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取交易所流入流出摘要
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.alert_history if a.timestamp >= cutoff]
        
        inflow = sum(a.amount for a in recent if a.direction == "exchange_in")
        outflow = sum(a.amount for a in recent if a.direction == "exchange_out")
        net_flow = outflow - inflow  # 正值 = 净流出 = 看涨
        
        return {
            "period_hours": hours,
            "total_alerts": len(recent),
            "exchange_inflow_eth": float(inflow),
            "exchange_outflow_eth": float(outflow),
            "net_flow_eth": float(net_flow),
            "sentiment": "看涨" if net_flow > 0 else "看跌",
            "pressure_score": round((inflow / (inflow + outflow + 1)) * 100, 1) if recent else 50,
        }
    
    def get_whale_alert(self, eth_price: Decimal) -> Dict[str, Any]:
        """
        获取巨鲸监控综合报告
        """
        # 生成模拟数据
        alerts = self._generate_simulated_alerts(eth_price, 5)
        self.update_stats(alerts)
        
        # 计算流入流出
        flow_summary = self.get_exchange_flow_summary()
        
        # 筛选高影响警报
        high_impact = [a for a in alerts if a.impact_score >= 60]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "alerts": [
                {
                    "time": a.timestamp.isoformat(),
                    "direction": a.direction,
                    "exchange": a.exchange_name,
                    "amount_eth": float(a.amount),
                    "amount_usd": float(a.amount_usd),
                    "impact": a.impact_score,
                    "details": a.details
                }
                for a in alerts[:5]
            ],
            "high_impact_count": len(high_impact),
            "flow_summary": flow_summary,
            "stats": {
                "total_monitored": self.stats["total_alerts"],
                "recent_inflow_usd": float(self.stats["exchange_inflow"] * eth_price),
                "recent_outflow_usd": float(self.stats["exchange_outflow"] * eth_price),
            }
        }


# 便捷函数
def whale_alert(price: float, limit: int = 5) -> Dict[str, Any]:
    """
    获取巨鲸警报（同步接口）
    
    Args:
        price: ETH 当前价格
        limit: 返回警报数量
    
    Returns:
        巨鲸监控报告
    """
    monitor = WhaleMonitor(use_simulation=True)
    return monitor.get_whale_alert(Decimal(str(price)))
