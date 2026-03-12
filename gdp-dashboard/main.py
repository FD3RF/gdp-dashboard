"""
AI Quant Trading System - Main Entry Point
==========================================

Institutional-grade AI-powered quantitative trading system.
Supports automated strategy research, backtesting, and execution.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Core imports
from core.base import BaseModule
from core.utils import setup_logger

# Configuration
from config.settings import Settings, get_settings, load_config

# Infrastructure
from infra.scheduler import Scheduler
from infra.task_queue import TaskQueue, TaskPriority
from infra.vector_memory import VectorMemory
from infra.model_manager import ModelManager
from infra.config_manager import ConfigManager
from infra.logging_system import LoggingSystem

# Data Layer
from data.collectors import (
    MarketDataCollector,
    OrderBookCollector,
    FundingRateCollector,
    OnChainDataCollector,
    NewsCollector,
    SocialSentimentCollector,
    MacroDataCollector
)
from data.processors import (
    DataCleaner,
    DataNormalizer,
    FeatureEngineering,
    DataWarehouse
)

# Agents
from agents import (
    PlannerAgent,
    ResearchAgent,
    StrategyAgent,
    CodingAgent,
    BacktestAgent,
    RiskAgent,
    ExecutionAgent,
    MonitoringAgent,
    OptimizationAgent,
    MemoryAgent,
    SelfImprovementAgent
)

# Strategies
from strategies import (
    TrendStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    StrategyCombiner
)
from strategies.base_strategy import StrategyConfig

# Backtest
from backtest import (
    BacktestEngine,
    HistoricalDataLoader,
    PerformanceAnalyzer
)

# Risk
from risk import (
    PositionSizing,
    StopLossEngine,
    DrawdownProtection,
    ExposureControl,
    VolatilityFilter,
    RiskDashboard
)

# Execution
from execution import (
    OrderManager,
    SmartOrderRouter,
    ExchangeAdapter,
    TWAPEngine,
    VWAPEngine
)

# Monitor
from monitor import (
    SystemHealthMonitor,
    StrategyPerformanceMonitor,
    TradeLogger,
    AlertSystem,
    DashboardAPI
)

# AI Automation
from ai_automation import (
    AutoStrategyGenerator,
    AutoParameterOptimizer,
    AutoBacktestRunner
)


class AIQuantSystem:
    """
    Main AI Quant Trading System class.
    Orchestrates all components and manages system lifecycle.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.settings = load_config(config_path)
        self.logger = setup_logger(
            'ai_quant_system',
            level=logging.INFO,
            log_file='logs/quant_system.log'
        )
        
        # Component containers
        self.infra: Dict[str, Any] = {}
        self.data: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}
        self.strategies: Dict[str, Any] = {}
        self.backtest: Dict[str, Any] = {}
        self.risk: Dict[str, Any] = {}
        self.execution: Dict[str, Any] = {}
        self.monitor: Dict[str, Any] = {}
        self.ai_automation: Dict[str, Any] = {}
        
        self._running = False
        self._startup_time: Optional[datetime] = None
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        self.logger.info("Initializing AI Quant Trading System...")
        self._startup_time = datetime.now()
        
        try:
            # Initialize infrastructure
            await self._initialize_infrastructure()
            
            # Initialize data layer
            await self._initialize_data()
            
            # Initialize risk management
            await self._initialize_risk()
            
            # Initialize execution
            await self._initialize_execution()
            
            # Initialize backtest
            await self._initialize_backtest()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Initialize strategies
            await self._initialize_strategies()
            
            # Initialize monitoring
            await self._initialize_monitoring()
            
            # Initialize AI automation
            await self._initialize_ai_automation()
            
            self.logger.info("System initialization complete!")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def _initialize_infrastructure(self) -> None:
        """Initialize infrastructure components."""
        self.logger.info("Initializing infrastructure...")
        
        # Logging system
        self.infra['logging'] = LoggingSystem({
            'level': self.settings.monitor.log_level,
            'log_dir': 'logs'
        })
        await self.infra['logging'].initialize()
        
        # Config manager
        self.infra['config'] = ConfigManager({'env_prefix': 'QUANT_'})
        await self.infra['config'].initialize()
        
        # Vector memory
        self.infra['vector_memory'] = VectorMemory({
            'embedding_dim': 768,
            'max_memories': 10000
        })
        await self.infra['vector_memory'].initialize()
        
        # Model manager (Ollama)
        self.infra['model_manager'] = ModelManager({
            'host': self.settings.ollama.host,
            'port': self.settings.ollama.port,
            'model': self.settings.ollama.model,
            'timeout': self.settings.ollama.timeout
        })
        await self.infra['model_manager'].initialize()
        
        # Task queue
        self.infra['task_queue'] = TaskQueue({'worker_count': 4})
        await self.infra['task_queue'].initialize()
        
        # Scheduler
        self.infra['scheduler'] = Scheduler({'check_interval': 1.0})
        await self.infra['scheduler'].initialize()
        
        self.logger.info("Infrastructure initialized")
    
    async def _initialize_data(self) -> None:
        """Initialize data collectors and processors."""
        self.logger.info("Initializing data layer...")
        
        # Data warehouse
        self.data['warehouse'] = DataWarehouse({
            'data_dir': 'data/warehouse'
        })
        await self.data['warehouse'].initialize()
        
        # Data processors
        self.data['cleaner'] = DataCleaner()
        await self.data['cleaner'].initialize()
        
        self.data['normalizer'] = DataNormalizer()
        await self.data['normalizer'].initialize()
        
        self.data['features'] = FeatureEngineering()
        await self.data['features'].initialize()
        
        # Data collectors
        self.data['market'] = MarketDataCollector({
            'default_exchange': self.settings.data.default_exchange,
            'default_timeframe': self.settings.data.default_timeframe
        })
        await self.data['market'].initialize()
        
        self.data['orderbook'] = OrderBookCollector()
        await self.data['orderbook'].initialize()
        
        self.data['funding'] = FundingRateCollector()
        await self.data['funding'].initialize()
        
        self.data['onchain'] = OnChainDataCollector()
        await self.data['onchain'].initialize()
        
        self.data['news'] = NewsCollector()
        await self.data['news'].initialize()
        
        self.data['sentiment'] = SocialSentimentCollector()
        await self.data['sentiment'].initialize()
        
        self.data['macro'] = MacroDataCollector()
        await self.data['macro'].initialize()
        
        self.logger.info("Data layer initialized")
    
    async def _initialize_risk(self) -> None:
        """Initialize risk management components."""
        self.logger.info("Initializing risk management...")
        
        self.risk['position_sizing'] = PositionSizing({
            'method': 'fixed_fractional',
            'risk_per_trade': self.settings.risk.max_single_trade_risk_pct
        })
        await self.risk['position_sizing'].initialize()
        
        self.risk['stop_loss'] = StopLossEngine({
            'default_stop_pct': 0.05,
            'trailing_enabled': True
        })
        await self.risk['stop_loss'].initialize()
        
        self.risk['drawdown'] = DrawdownProtection({
            'max_drawdown': self.settings.risk.max_drawdown_pct
        })
        await self.risk['drawdown'].initialize()
        
        self.risk['exposure'] = ExposureControl({
            'max_leverage': self.settings.risk.max_portfolio_leverage
        })
        await self.risk['exposure'].initialize()
        
        self.risk['volatility'] = VolatilityFilter()
        await self.risk['volatility'].initialize()
        
        self.risk['dashboard'] = RiskDashboard()
        await self.risk['dashboard'].initialize()
        
        # Set module references
        self.risk['dashboard'].set_modules(
            position_sizing=self.risk['position_sizing'],
            stop_loss=self.risk['stop_loss'],
            drawdown=self.risk['drawdown'],
            exposure=self.risk['exposure'],
            volatility=self.risk['volatility']
        )
        
        self.logger.info("Risk management initialized")
    
    async def _initialize_execution(self) -> None:
        """Initialize execution components."""
        self.logger.info("Initializing execution layer...")
        
        self.execution['adapter'] = ExchangeAdapter({
            'testnet': self.settings.environment != 'production'
        })
        await self.execution['adapter'].initialize()
        
        self.execution['order_manager'] = OrderManager()
        await self.execution['order_manager'].initialize()
        self.execution['order_manager'].set_exchange_adapter(self.execution['adapter'])
        
        self.execution['smart_router'] = SmartOrderRouter()
        await self.execution['smart_router'].initialize()
        
        self.execution['twap'] = TWAPEngine()
        await self.execution['twap'].initialize()
        self.execution['twap'].set_order_manager(self.execution['order_manager'])
        
        self.execution['vwap'] = VWAPEngine()
        await self.execution['vwap'].initialize()
        self.execution['vwap'].set_order_manager(self.execution['order_manager'])
        
        self.logger.info("Execution layer initialized")
    
    async def _initialize_backtest(self) -> None:
        """Initialize backtest components."""
        self.logger.info("Initializing backtest system...")
        
        self.backtest['data_loader'] = HistoricalDataLoader()
        await self.backtest['data_loader'].initialize()
        
        self.backtest['engine'] = BacktestEngine({
            'initial_capital': self.settings.backtest.initial_capital
        })
        await self.backtest['engine'].initialize()
        self.backtest['engine'].set_data_loader(self.backtest['data_loader'])
        
        self.backtest['analyzer'] = PerformanceAnalyzer()
        await self.backtest['analyzer'].initialize()
        
        self.logger.info("Backtest system initialized")
    
    async def _initialize_agents(self) -> None:
        """Initialize AI agents."""
        self.logger.info("Initializing AI agents...")
        
        model_manager = self.infra['model_manager']
        vector_memory = self.infra['vector_memory']
        
        # Planner
        self.agents['planner'] = PlannerAgent(model_manager=model_manager, vector_memory=vector_memory)
        await self.agents['planner'].initialize()
        
        # Research
        self.agents['research'] = ResearchAgent(
            model_manager=model_manager,
            vector_memory=vector_memory
        )
        await self.agents['research'].initialize()
        
        # Strategy
        self.agents['strategy'] = StrategyAgent(
            model_manager=model_manager,
            vector_memory=vector_memory
        )
        await self.agents['strategy'].initialize()
        
        # Coding
        self.agents['coding'] = CodingAgent(
            model_manager=model_manager,
            vector_memory=vector_memory
        )
        await self.agents['coding'].initialize()
        
        # Backtest
        self.agents['backtest'] = BacktestAgent(
            model_manager=model_manager,
            vector_memory=vector_memory,
            backtest_engine=self.backtest['engine']
        )
        await self.agents['backtest'].initialize()
        
        # Risk
        self.agents['risk'] = RiskAgent(
            model_manager=model_manager,
            vector_memory=vector_memory,
            risk_manager=self.risk['dashboard']
        )
        await self.agents['risk'].initialize()
        
        # Execution
        self.agents['execution'] = ExecutionAgent(
            model_manager=model_manager,
            vector_memory=vector_memory,
            order_manager=self.execution['order_manager']
        )
        await self.agents['execution'].initialize()
        
        # Monitoring
        self.agents['monitoring'] = MonitoringAgent(
            model_manager=model_manager,
            vector_memory=vector_memory
        )
        await self.agents['monitoring'].initialize()
        
        # Optimization
        self.agents['optimization'] = OptimizationAgent(
            model_manager=model_manager,
            vector_memory=vector_memory,
            backtest_agent=self.agents['backtest']
        )
        await self.agents['optimization'].initialize()
        
        # Memory
        self.agents['memory'] = MemoryAgent(
            model_manager=model_manager,
            vector_memory=vector_memory
        )
        await self.agents['memory'].initialize()
        
        # Self-improvement
        self.agents['self_improvement'] = SelfImprovementAgent(
            model_manager=model_manager,
            vector_memory=vector_memory
        )
        await self.agents['self_improvement'].initialize()
        
        # Register agents with planner
        for name, agent in self.agents.items():
            if name != 'planner':
                self.agents['planner'].register_agent(agent)
        
        self.logger.info("AI agents initialized")
    
    async def _initialize_strategies(self) -> None:
        """Initialize trading strategies."""
        self.logger.info("Initializing strategies...")
        
        # Trend strategy
        trend_config = StrategyConfig(
            name='trend_following',
            timeframe='1h',
            symbols=['BTC/USDT'],
            parameters={'fast_period': 10, 'slow_period': 30}
        )
        self.strategies['trend'] = TrendStrategy(trend_config)
        await self.strategies['trend'].initialize()
        
        # Mean reversion strategy
        mr_config = StrategyConfig(
            name='mean_reversion',
            timeframe='1h',
            symbols=['BTC/USDT'],
            parameters={'bb_period': 20, 'rsi_period': 14}
        )
        self.strategies['mean_reversion'] = MeanReversionStrategy(mr_config)
        await self.strategies['mean_reversion'].initialize()
        
        # Momentum strategy
        mom_config = StrategyConfig(
            name='momentum',
            timeframe='1h',
            symbols=['BTC/USDT'],
            parameters={'momentum_period': 14}
        )
        self.strategies['momentum'] = MomentumStrategy(mom_config)
        await self.strategies['momentum'].initialize()
        
        self.logger.info("Strategies initialized")
    
    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring components."""
        self.logger.info("Initializing monitoring...")
        
        self.monitor['health'] = SystemHealthMonitor()
        await self.monitor['health'].initialize()
        
        self.monitor['performance'] = StrategyPerformanceMonitor()
        await self.monitor['performance'].initialize()
        
        self.monitor['trade_logger'] = TradeLogger()
        await self.monitor['trade_logger'].initialize()
        
        self.monitor['alerts'] = AlertSystem({
            'telegram_enabled': self.settings.monitor.telegram_enabled,
            'telegram_token': self.settings.monitor.telegram_token,
            'telegram_chat_id': self.settings.monitor.telegram_chat_id
        })
        await self.monitor['alerts'].initialize()
        
        self.monitor['dashboard'] = DashboardAPI({
            'port': self.settings.monitor.metrics_port
        })
        await self.monitor['dashboard'].initialize()
        
        # Set system reference for dashboard
        self.monitor['dashboard'].set_system_reference({
            'agents': self.agents,
            'strategies': self.strategies,
            'risk_dashboard': self.risk['dashboard'],
            'alert_system': self.monitor['alerts']
        })
        
        self.logger.info("Monitoring initialized")
    
    async def _initialize_ai_automation(self) -> None:
        """Initialize AI automation components."""
        self.logger.info("Initializing AI automation...")
        
        model_manager = self.infra['model_manager']
        
        self.ai_automation['strategy_generator'] = AutoStrategyGenerator()
        await self.ai_automation['strategy_generator'].initialize()
        self.ai_automation['strategy_generator'].set_model_manager(model_manager)
        
        self.ai_automation['parameter_optimizer'] = AutoParameterOptimizer()
        await self.ai_automation['parameter_optimizer'].initialize()
        self.ai_automation['parameter_optimizer'].set_model_manager(model_manager)
        
        self.ai_automation['backtest_runner'] = AutoBacktestRunner()
        await self.ai_automation['backtest_runner'].initialize()
        self.ai_automation['backtest_runner'].set_model_manager(model_manager)
        self.ai_automation['backtest_runner'].set_backtest_engine(self.backtest['engine'])
        
        self.logger.info("AI automation initialized")
    
    async def start(self) -> None:
        """Start all system components."""
        self.logger.info("Starting AI Quant Trading System...")
        
        # Start infrastructure
        for name, component in self.infra.items():
            await component.start()
            self.logger.debug(f"Started: {name}")
        
        # Start data collectors
        for name, component in self.data.items():
            await component.start()
            self.logger.debug(f"Started: {name}")
        
        # Start risk management
        for name, component in self.risk.items():
            await component.start()
            self.logger.debug(f"Started: {name}")
        
        # Start execution
        for name, component in self.execution.items():
            await component.start()
            self.logger.debug(f"Started: {name}")
        
        # Start backtest
        for name, component in self.backtest.items():
            await component.start()
            self.logger.debug(f"Started: {name}")
        
        # Start agents
        for name, agent in self.agents.items():
            await agent.start()
            self.logger.debug(f"Started: {name}")
        
        # Start strategies
        for name, strategy in self.strategies.items():
            await strategy.start()
            self.logger.debug(f"Started: {name}")
        
        # Start monitoring
        for name, component in self.monitor.items():
            await component.start()
            self.logger.debug(f"Started: {name}")
        
        # Start AI automation
        for name, component in self.ai_automation.items():
            await component.start()
            self.logger.debug(f"Started: {name}")
        
        self._running = True
        self.logger.info("System started successfully!")
    
    async def stop(self) -> None:
        """Stop all system components."""
        self.logger.info("Stopping AI Quant Trading System...")
        self._running = False
        
        # Stop in reverse order
        components = [
            self.ai_automation, self.monitor, self.strategies,
            self.agents, self.backtest, self.execution,
            self.risk, self.data, self.infra
        ]
        
        for component_dict in components:
            for name, component in component_dict.items():
                try:
                    await component.stop()
                    self.logger.debug(f"Stopped: {name}")
                except Exception as e:
                    self.logger.error(f"Error stopping {name}: {e}")
        
        self.logger.info("System stopped")
    
    async def run_mvp_demo(self) -> None:
        """
        Run MVP demonstration:
        1. Fetch BTC market data
        2. Generate strategy using AI
        3. Backtest the strategy
        4. Check risk
        5. Simulate execution
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting MVP Demonstration")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Fetch BTC market data
            self.logger.info("\n[Step 1] Fetching BTC/USDT market data...")
            
            market_data = await self.data['market'].fetch_ohlcv(
                symbol='BTC/USDT',
                timeframe='1h',
                limit=100
            )
            
            self.logger.info(f"Fetched {len(market_data)} candles")
            self.logger.info(f"Date range: {market_data.index[0]} to {market_data.index[-1]}")
            self.logger.info(f"Latest price: ${market_data['close'].iloc[-1]:,.2f}")
            
            # Step 2: AI generates strategy
            self.logger.info("\n[Step 2] AI generating trading strategy...")
            
            strategy_result = await self.ai_automation['strategy_generator'].generate_strategy(
                market_condition='trending',
                risk_profile='moderate',
                timeframe='1h',
                symbols=['BTC/USDT']
            )
            
            self.logger.info(f"Generated strategy: {strategy_result.get('name', 'Unknown')}")
            self.logger.info(f"Strategy type: {strategy_result.get('type', 'Unknown')}")
            
            # Step 3: Run backtest
            self.logger.info("\n[Step 3] Running backtest...")
            
            backtest_result = await self.ai_automation['backtest_runner'].run_backtest(
                strategy=self.strategies['trend'],
                auto_analyze=True
            )
            
            self.logger.info(f"Total Return: {backtest_result.get('total_return', 0)*100:.2f}%")
            self.logger.info(f"Sharpe Ratio: {backtest_result.get('sharpe_ratio', 0):.2f}")
            self.logger.info(f"Max Drawdown: {backtest_result.get('max_drawdown', 0)*100:.2f}%")
            self.logger.info(f"Win Rate: {backtest_result.get('win_rate', 0)*100:.1f}%")
            
            if 'ai_analysis' in backtest_result:
                self.logger.info(f"AI Assessment: {backtest_result['ai_analysis'].get('assessment', 'Unknown')}")
            
            # Step 4: Risk check
            self.logger.info("\n[Step 4] Checking risk metrics...")
            
            risk_dashboard = self.risk['dashboard'].get_dashboard(100000)
            self.logger.info(f"Risk Level: {risk_dashboard.get('overall_risk_level', 'Unknown')}")
            self.logger.info(f"Risk Score: {self.risk['dashboard'].calculate_risk_score():.1f}/100")
            
            # Step 5: Simulate execution
            self.logger.info("\n[Step 5] Simulating trade execution...")
            
            order = await self.execution['order_manager'].submit_order(
                symbol='BTC/USDT',
                side='buy',
                quantity=0.1,
                order_type='market'
            )
            
            self.logger.info(f"Order ID: {order.get('order_id')}")
            self.logger.info(f"Status: {order.get('status')}")
            self.logger.info(f"Filled: {order.get('filled_quantity')} @ ${order.get('filled_price', 0):,.2f}")
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("MVP Demonstration Complete!")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"MVP demo error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'running': self._running,
            'startup_time': self._startup_time.isoformat() if self._startup_time else None,
            'uptime_seconds': (datetime.now() - self._startup_time).total_seconds() if self._startup_time else 0,
            'components': {
                'infrastructure': len(self.infra),
                'data_collectors': len(self.data),
                'agents': len(self.agents),
                'strategies': len(self.strategies),
                'risk_modules': len(self.risk),
                'execution_modules': len(self.execution),
                'monitor_modules': len(self.monitor),
                'ai_automation': len(self.ai_automation)
            }
        }


async def main():
    """Main entry point."""
    # Parse config path
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Create system instance
    system = AIQuantSystem(config_path)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logging.info("Received shutdown signal")
        asyncio.create_task(system.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Initialize
        if not await system.initialize():
            logging.error("Initialization failed")
            sys.exit(1)
        
        # Start
        await system.start()
        
        # Run MVP demo
        await system.run_mvp_demo()
        
        # Keep running
        logging.info("System running. Press Ctrl+C to stop.")
        while system._running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
    finally:
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())
