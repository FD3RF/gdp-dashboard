"""
Telegram 预警推送模块
=====================
将关键预警推送到 Telegram

功能：
1. 实时推送 CRITICAL/EMERGENCY 级别预警
2. 支持 Markdown 格式
3. 支持消息去重和限流
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Telegram 配置"""
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = False
    min_severity: str = "warning"  # 最低推送级别
    rate_limit_seconds: int = 60   # 同类型消息限流间隔
    quiet_hours: tuple = (0, 6)    # 静默时段 (0-6点不推送)


class TelegramNotifier:
    """
    Telegram 通知器
    
    使用方法：
    1. 创建 Telegram Bot 并获取 token
    2. 获取 chat_id（个人或群组）
    3. 设置环境变量：
       - TELEGRAM_BOT_TOKEN
       - TELEGRAM_CHAT_ID
    """
    
    SEVERITY_EMOJI = {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🔴",
        "emergency": "🚨",
    }
    
    SEVERITY_PRIORITY = {
        "info": 0,
        "warning": 1,
        "critical": 2,
        "emergency": 3,
    }
    
    def __init__(self, config: TelegramConfig = None):
        self.config = config or TelegramConfig()
        
        # 从环境变量加载配置
        if not self.config.bot_token:
            self.config.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not self.config.chat_id:
            self.config.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        if self.config.bot_token and self.config.chat_id:
            self.config.enabled = True
        
        # 消息历史（用于去重和限流）
        self._message_history: Dict[str, datetime] = {}
        self._max_history = 100
        
        # 会话
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取 HTTP 会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """关闭会话"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _should_send(self, message_key: str, severity: str) -> bool:
        """检查是否应该发送消息（去重和限流）"""
        if not self.config.enabled:
            return False
        
        # 检查最低级别
        if self.SEVERITY_PRIORITY.get(severity, 0) < self.SEVERITY_PRIORITY.get(self.config.min_severity, 1):
            return False
        
        # 检查静默时段
        now = datetime.now()
        if self.config.quiet_hours[0] <= now.hour < self.config.quiet_hours[1]:
            # 静默时段只推送 emergency
            if severity != "emergency":
                return False
        
        # 检查限流
        if message_key in self._message_history:
            last_sent = self._message_history[message_key]
            if (now - last_sent).total_seconds() < self.config.rate_limit_seconds:
                return False
        
        return True
    
    def _mark_sent(self, message_key: str):
        """标记消息已发送"""
        self._message_history[message_key] = datetime.now()
        
        # 清理过期记录
        if len(self._message_history) > self._max_history:
            cutoff = datetime.now() - timedelta(hours=1)
            self._message_history = {
                k: v for k, v in self._message_history.items()
                if v > cutoff
            }
    
    async def send_message(
        self,
        text: str,
        severity: str = "info",
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
    ) -> bool:
        """
        发送 Telegram 消息
        
        Args:
            text: 消息内容
            severity: 严重级别 (info/warning/critical/emergency)
            parse_mode: 解析模式 (Markdown/HTML)
            disable_notification: 是否静音
        
        Returns:
            是否发送成功
        """
        if not self.config.enabled:
            logger.debug("Telegram notifications disabled")
            return False
        
        # 生成消息键（用于去重）
        message_key = f"{severity}:{text[:50]}"
        
        if not self._should_send(message_key, severity):
            return False
        
        try:
            session = await self._get_session()
            
            url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
            
            # 添加图标和时间戳
            emoji = self.SEVERITY_EMOJI.get(severity, "ℹ️")
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_text = f"{emoji} *{severity.upper()}* [{timestamp}]\n\n{text}"
            
            payload = {
                "chat_id": self.config.chat_id,
                "text": formatted_text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    self._mark_sent(message_key)
                    logger.info(f"Telegram message sent: {severity}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Telegram API error: {error}")
                    return False
        
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def send_alert(self, alert: Dict) -> bool:
        """
        发送预警消息
        
        Args:
            alert: 预警字典（来自 Alert.to_dict()）
        """
        severity = alert.get("severity", "info")
        title = alert.get("title", "预警")
        message = alert.get("message", "")
        details = alert.get("details", {})
        
        # 格式化消息
        text_lines = [
            f"*{title}*",
            "",
            message,
        ]
        
        # 添加关键详情
        if details:
            text_lines.append("")
            text_lines.append("📊 *详情:*")
            for key, value in details.items():
                if key in ["level_price", "liquidation_price", "breakout_price"]:
                    text_lines.append(f"• 价格: ${value:,.0f}")
                elif key in ["distance_pct", "range_width_pct"]:
                    text_lines.append(f"• 距离: {value:.2f}%")
                elif key in ["strength", "total_strength"]:
                    text_lines.append(f"• 强度: {value:.1f}x")
                elif key == "action_hint":
                    text_lines.append(f"• 建议: {value}")
        
        text = "\n".join(text_lines)
        
        return await self.send_message(text, severity)
    
    async def send_trade_signal(
        self,
        signal: str,
        price: float,
        confidence: float,
        stop_loss: float = None,
        take_profit: float = None,
    ) -> bool:
        """
        发送交易信号
        
        Args:
            signal: 信号 (LONG/SHORT)
            price: 价格
            confidence: 置信度
            stop_loss: 止损价
            take_profit: 止盈价
        """
        emoji = "🟢" if signal == "LONG" else "🔴"
        
        text_lines = [
            f"{emoji} *{signal} 信号*",
            "",
            f"💰 价格: ${price:,.2f}",
            f"📊 置信度: {confidence:.1f}%",
        ]
        
        if stop_loss:
            text_lines.append(f"🛡️ 止损: ${stop_loss:,.2f}")
        if take_profit:
            text_lines.append(f"🎯 止盈: ${take_profit:,.2f}")
        
        text = "\n".join(text_lines)
        
        return await self.send_message(text, "critical")
    
    async def send_daily_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        total_pnl: float,
        win_rate: float,
    ) -> bool:
        """
        发送每日总结
        
        Args:
            total_trades: 总交易数
            wins: 盈利数
            losses: 亏损数
            total_pnl: 总盈亏
            win_rate: 胜率
        """
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        
        text_lines = [
            "📊 *每日交易总结*",
            "",
            f"交易次数: {total_trades}",
            f"盈利: {wins} | 亏损: {losses}",
            f"胜率: {win_rate:.1f}%",
            f"{pnl_emoji} 盈亏: ${total_pnl:,.2f}",
        ]
        
        text = "\n".join(text_lines)
        
        return await self.send_message(text, "info")


# 全局实例
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """获取通知器"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


async def send_telegram_alert(alert: Dict) -> bool:
    """发送预警（便捷函数）"""
    return await get_notifier().send_alert(alert)


async def send_telegram_signal(signal: str, price: float, confidence: float, **kwargs) -> bool:
    """发送信号（便捷函数）"""
    return await get_notifier().send_trade_signal(signal, price, confidence, **kwargs)


def send_telegram_sync(text: str, severity: str = "warning") -> bool:
    """同步发送消息（用于非异步环境）"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(get_notifier().send_message(text, severity))
