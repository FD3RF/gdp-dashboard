"""
AI 量化交易系统
================

全自动化交易系统主入口：
- 系统编排
- Agent 协调
- 配置管理
- 启动入口
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

# 核心组件
from core.scheduler import Scheduler, TaskPriority
from core.base import BaseModule

# Agent
from agents.trade_agent import TradeAgent, TradeMode, TradeSignal
from agents.strategy_ai import StrategyAI, StrategyType
from agents.monitor_agent import MonitorAgent, RiskLevel, Alert

# 交易所
from execution.okx_sync_manager import OKXSyncManager


class TradingSystem:
    """
    AI 量化交易系统
    
    架构：
    ┌─────────────────────────────────────┐
    │           TradingSystem              │
    │  ┌─────────────────────────────┐    │
    │  │        Scheduler            │    │
    │  │   (任务调度 & 心跳控制)      │    │
    │  └─────────────────────────────┘    │
    │              │                       │
    │  ┌───────────┼───────────┐          │
    │  ▼           ▼           ▼          │
    │ ┌─────┐   ┌─────┐   ┌─────┐        │
    │ │Trade│   │Strat│   │Moni │        │
    │ │Agent│   │ AI │   │tor  │        │
    │ └──┬──┘   └──┬──┘   └──┬──┘        │
    │    │         │         │            │
    │    ▼         ▼         ▼            │
    │ ┌─────────────────────────┐        │
    │ │   OKXSyncManager        │        │
    │ │   (数据同步 & 订单执行)  │        │
    │ └─────────────────────────┘        │
    └─────────────────────────────────────┘
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 交易模式
        'trade_mode': 'simulation',  # full_auto, semi_auto, simulation, manual
        
        # 交易所配置
        'exchange': {
            'name': 'okx',
            'testnet': True,
            'account_type': 'swap',  # spot, swap, futures
        },
        
        # 策略配置
        'strategies': ['trend', 'reversal', 'momentum'],
        
        # 风控配置
        'risk': {
            'max_position_ratio': 0.50,
            'max_single_position': 0.30,
            'max_loss_ratio': 0.15,
            'max_daily_loss': 0.05,
            'auto_protection': True,
        },
        
        # 监控的交易对
        'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        
        # K线周期
        'timeframe': '1h',
        
        # 数据刷新间隔
        'refresh_interval': 5,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger("TradingSystem")
        
        # 核心组件
        self._scheduler: Optional[Scheduler] = None
        self._trade_agent: Optional[TradeAgent] = None
        self._strategy_ai: Optional[StrategyAI] = None
        self._monitor_agent: Optional[MonitorAgent] = None
        self._sync_manager: Optional[OKXSyncManager] = None
        
        # 运行状态
        self._running = False
        self._start_time: Optional[datetime] = None
        
        # 数据缓存
        self._kline_cache: Dict[str, List] = {}
        self._price_cache: Dict[str, float] = {}
    
    async def initialize(self) -> bool:
        """初始化系统"""
        self.logger.info("=" * 50)
        self.logger.info("AI 量化交易系统 - 初始化中...")
        self.logger.info("=" * 50)
        
        # 1. 初始化调度器
        self._scheduler = Scheduler()
        await self._scheduler.start()
        self.logger.info("✓ 调度器启动")
        
        # 2. 初始化交易代理
        self._trade_agent = TradeAgent({
            'mode': self.config['trade_mode'],
            'exchange': self.config.get('exchange_config', {}),
            'leverage': self.config.get('leverage', 1),
        })
        await self._trade_agent.initialize()
        await self._trade_agent.start()
        self._scheduler.register_agent('trade', self._trade_agent)
        self.logger.info(f"✓ 交易代理启动 (模式: {self.config['trade_mode']})")
        
        # 3. 初始化策略 AI
        self._strategy_ai = StrategyAI({
            'strategies': self.config['strategies'],
        })
        await self._strategy_ai.initialize()
        await self._strategy_ai.start()
        self._scheduler.register_agent('strategy', self._strategy_ai)
        self.logger.info(f"✓ 策略 AI 启动 (策略: {self.config['strategies']})")
        
        # 4. 初始化监控代理
        self._monitor_agent = MonitorAgent({
            'thresholds': self.config['risk'],
            'auto_protection': self.config['risk'].get('auto_protection', True),
        })
        await self._monitor_agent.initialize()
        await self._monitor_agent.start()
        self._scheduler.register_agent('monitor', self._monitor_agent)
        self.logger.info("✓ 监控代理启动")
        
        # 5. 设置回调
        self._setup_callbacks()
        
        # 6. 注册定时任务
        self._register_tasks()
        
        self.logger.info("=" * 50)
        self.logger.info("系统初始化完成！")
        self.logger.info("=" * 50)
        
        return True
    
    def _setup_callbacks(self):
        """设置回调"""
        # 信号回调：策略 -> 交易
        self._strategy_ai.on_signal_callbacks.append(self._on_signal_generated)
        
        # 警报回调：监控 -> 通知
        self._monitor_agent.add_alert_callback(self._on_alert)
        
        # 风控回调：监控 -> 交易
        self._monitor_agent.set_risk_breach_callback(self._on_risk_breach)
    
    def _register_tasks(self):
        """注册定时任务"""
        # 数据同步任务
        self._scheduler.add_task(
            'data_sync',
            self._data_sync_task,
            self.config['refresh_interval'],
            TaskPriority.HIGH
        )
        
        # 信号生成任务
        self._scheduler.add_task(
            'signal_generation',
            self._signal_generation_task,
            10,  # 每10秒
            TaskPriority.NORMAL
        )
        
        # 风控检查任务
        self._scheduler.add_task(
            'risk_check',
            self._risk_check_task,
            5,  # 每5秒
            TaskPriority.CRITICAL
        )
        
        # 仓位更新任务
        self._scheduler.add_task(
            'position_update',
            self._position_update_task,
            3,
            TaskPriority.HIGH
        )
        
        # 状态报告任务
        self._scheduler.add_task(
            'status_report',
            self._status_report_task,
            60,  # 每60秒
            TaskPriority.LOW
        )
    
    # ==================== 定时任务 ====================
    
    async def _data_sync_task(self):
        """数据同步任务"""
        if not self._sync_manager:
            return
        
        # 同步价格
        for symbol in self.config['symbols']:
            ticker = self._sync_manager.get_ticker(symbol)
            if ticker:
                self._price_cache[symbol] = ticker.get('last', 0)
    
    async def _signal_generation_task(self):
        """信号生成任务"""
        if not self._strategy_ai or not self._trade_agent:
            return
        
        for symbol in self.config['symbols']:
            # 获取K线数据（从缓存或API）
            klines = self._kline_cache.get(symbol)
            if not klines:
                continue
            
            # 转换为DataFrame
            import pandas as pd
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 生成信号
            signals = await self._strategy_ai.analyze(symbol, df)
            
            # 处理信号
            for signal in signals:
                await self._trade_agent.process_signal(signal)
    
    async def _risk_check_task(self):
        """风控检查任务"""
        if not self._monitor_agent or not self._trade_agent:
            return
        
        # 获取当前状态
        trade_status = self._trade_agent.get_status()
        
        # 更新风控指标
        await self._monitor_agent.update_metrics(
            total_equity=trade_status.get('total_equity', 0),
            positions=trade_status.get('positions', {}),
            daily_pnl=0,  # 需要计算
            realized_pnl=trade_status.get('stats', {}).get('total_pnl', 0),
            margin_used=trade_status.get('margin_used', 0),
            leverage=self.config.get('leverage', 1),
        )
        
        # 执行风控检查
        result = await self._monitor_agent.check_risk()
        
        if not result['passed']:
            self.logger.warning(f"风控检查未通过: {result['risk_level']}")
    
    async def _position_update_task(self):
        """仓位更新任务"""
        if not self._trade_agent:
            return
        
        # 更新持仓价格
        await self._trade_agent.update_positions(self._price_cache)
    
    async def _status_report_task(self):
        """状态报告任务"""
        self.logger.info("=" * 30)
        self.logger.info("系统状态报告")
        self.logger.info(f"运行时间: {(datetime.now() - self._start_time).total_seconds() / 60:.0f} 分钟" if self._start_time else "未启动")
        
        if self._trade_agent:
            status = self._trade_agent.get_status()
            self.logger.info(f"总权益: ${status['total_equity']:.2f}")
            self.logger.info(f"持仓数: {len(status['positions'])}")
            self.logger.info(f"总盈亏: ${status['stats']['total_pnl']:.2f}")
            self.logger.info(f"胜率: {status['stats']['win_rate']:.1f}%")
        
        if self._monitor_agent:
            risk_level = self._monitor_agent.get_risk_level()
            self.logger.info(f"风险等级: {risk_level.value}")
        
        self.logger.info("=" * 30)
    
    # ==================== 回调处理 ====================
    
    async def _on_signal_generated(self, signal: TradeSignal):
        """信号生成回调"""
        self.logger.info(
            f"📢 信号: {signal.symbol} {signal.side.value} "
            f"{signal.signal_type} (强度: {signal.strength:.2f})"
        )
        self.logger.info(f"   原因: {signal.reason}")
    
    async def _on_alert(self, alert: Alert):
        """警报回调"""
        self.logger.warning(
            f"⚠️ 警报 [{alert.level.value}]: {alert.message}"
        )
        
        # 这里可以添加通知逻辑
        # - 发送邮件
        # - 发送 Telegram
        # - 发送微信
        # - 发送 Discord
    
    async def _on_risk_breach(self, risk_result: Dict[str, Any]):
        """风控违规回调"""
        self.logger.critical("🚨 风控违规！触发保护机制")
        
        # 自动平仓
        if self._trade_agent and self.config['risk'].get('auto_protection', True):
            positions = self._trade_agent.get_positions()
            
            for symbol, position in positions.items():
                try:
                    await self._trade_agent.process_signal(TradeSignal(
                        symbol=symbol,
                        side='sell' if position.side == 'long' else 'buy',
                        signal_type='exit',
                        strength=1.0,
                        reason='风控保护性平仓',
                    ))
                    self.logger.info(f"已平仓: {symbol}")
                except Exception as e:
                    self.logger.error(f"平仓失败 {symbol}: {e}")
    
    # ==================== 公共接口 ====================
    
    async def start(self):
        """启动系统"""
        if self._running:
            return
        
        if not await self.initialize():
            self.logger.error("系统初始化失败")
            return
        
        self._running = True
        self._start_time = datetime.now()
        
        self.logger.info("🚀 系统已启动！")
    
    async def stop(self):
        """停止系统"""
        self._running = False
        
        # 停止各组件
        if self._trade_agent:
            await self._trade_agent.stop()
        if self._strategy_ai:
            await self._strategy_ai.stop()
        if self._monitor_agent:
            await self._monitor_agent.stop()
        if self._scheduler:
            await self._scheduler.stop()
        if self._sync_manager:
            await self._sync_manager.stop()
        
        self.logger.info("系统已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self._running,
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'uptime': (
                (datetime.now() - self._start_time).total_seconds()
                if self._start_time else 0
            ),
            'trade_mode': self.config['trade_mode'],
            'symbols': self.config['symbols'],
            'trade_agent': self._trade_agent.get_status() if self._trade_agent else None,
            'monitor_agent': self._monitor_agent.get_status() if self._monitor_agent else None,
            'scheduler': self._scheduler.get_status() if self._scheduler else None,
        }
    
    def set_trade_mode(self, mode: str):
        """设置交易模式"""
        if self._trade_agent:
            from agents.trade_agent import TradeMode
            self._trade_agent.set_mode(TradeMode(mode))
            self.config['trade_mode'] = mode
            self.logger.info(f"交易模式已切换为: {mode}")
    
    async def manual_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """手动交易"""
        if not self._trade_agent:
            return {'error': 'Trade agent not initialized'}
        
        from core.constants import OrderSide
        
        signal = TradeSignal(
            symbol=symbol,
            side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
            signal_type='entry',
            strength=1.0,
            quantity=quantity,
            price=price,
            reason='手动交易',
        )
        
        return await self._trade_agent.process_signal(signal)
    
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """平仓"""
        if not self._trade_agent:
            return {'error': 'Trade agent not initialized'}
        
        positions = self._trade_agent.get_positions()
        position = positions.get(symbol)
        
        if not position:
            return {'error': f'No position for {symbol}'}
        
        from core.constants import OrderSide
        
        signal = TradeSignal(
            symbol=symbol,
            side=OrderSide.SELL if position.side == 'long' else OrderSide.BUY,
            signal_type='exit',
            strength=1.0,
            reason='手动平仓',
        )
        
        return await self._trade_agent.process_signal(signal)


# ==================== 主入口 ====================

async def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建系统实例
    system = TradingSystem({
        'trade_mode': 'simulation',  # 模拟模式
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'timeframe': '1h',
        'refresh_interval': 5,
        'risk': {
            'max_position_ratio': 0.50,
            'max_single_position': 0.30,
            'max_loss_ratio': 0.15,
            'auto_protection': True,
        },
    })
    
    try:
        # 启动系统
        await system.start()
        
        # 运行
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        print("\n正在停止系统...")
    finally:
        await system.stop()


if __name__ == '__main__':
    asyncio.run(main())


# 导出
__all__ = ['TradingSystem']
