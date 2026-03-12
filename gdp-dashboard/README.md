# AI Quant Trading System
## 机构级 AI 量化交易系统

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

### 🚀 系统概述

这是一个接近量化基金级别的 **AI 自动交易平台**，支持：

- **AI 自动研究市场** - 智能分析市场数据和新闻
- **AI 自动生成交易策略** - 基于 Ollama + Qwen2.5-Coder
- **AI 自动回测策略** - 完整的回测引擎
- **AI 自动优化策略参数** - 多种优化算法
- **AI 自动执行交易** - 智能订单路由
- **AI 自动监控与风控** - 实时风险监控

---

### 📦 项目结构

```
ai_quant_system/
├── core/                   # 核心基础模块
│   ├── base.py            # 基础类
│   ├── constants.py       # 常量定义
│   ├── exceptions.py      # 自定义异常
│   └── utils.py           # 工具函数
│
├── config/                 # 配置管理
│   └── settings.py        # 系统配置
│
├── infra/                  # 基础设施
│   ├── scheduler.py       # 任务调度器
│   ├── task_queue.py      # 任务队列
│   ├── vector_memory.py   # 向量记忆
│   ├── model_manager.py   # AI模型管理
│   ├── config_manager.py  # 配置管理器
│   └── logging_system.py  # 日志系统
│
├── data/                   # 数据层
│   ├── collectors/        # 数据采集器
│   │   ├── market_data.py
│   │   ├── orderbook.py
│   │   ├── funding_rate.py
│   │   ├── onchain.py
│   │   ├── news.py
│   │   ├── social_sentiment.py
│   │   └── macro_data.py
│   │
│   └── processors/        # 数据处理器
│       ├── data_cleaner.py
│       ├── data_normalizer.py
│       ├── feature_engineering.py
│       └── data_warehouse.py
│
├── agents/                 # AI 智能体层 (11 Agents)
│   ├── planner.py         # 规划 Agent
│   ├── research.py        # 研究 Agent
│   ├── strategy_agent.py  # 策略 Agent
│   ├── coding.py          # 代码生成 Agent
│   ├── backtest_agent.py  # 回测 Agent
│   ├── risk_agent.py      # 风控 Agent
│   ├── execution_agent.py # 执行 Agent
│   ├── monitoring_agent.py# 监控 Agent
│   ├── optimization.py    # 优化 Agent
│   ├── memory_agent.py    # 记忆 Agent
│   └── self_improvement.py# 自我改进 Agent
│
├── strategies/             # 策略层
│   ├── base_strategy.py   # 策略基类
│   ├── trend_strategy.py  # 趋势策略
│   ├── mean_reversion.py  # 均值回归策略
│   ├── momentum.py        # 动量策略
│   ├── statistical_arb.py # 统计套利策略
│   ├── funding_arb.py     # 资金费率套利
│   ├── market_making.py   # 做市策略
│   ├── portfolio_optimizer.py
│   └── strategy_combiner.py
│
├── backtest/               # 回测系统
│   ├── backtest_engine.py # 回测引擎
│   ├── historical_loader.py
│   ├── slippage_model.py  # 滑点模型
│   ├── fee_model.py       # 手续费模型
│   ├── performance_analyzer.py
│   └── walk_forward.py    # 滚动前向分析
│
├── risk/                   # 风控层
│   ├── position_sizing.py # 仓位管理
│   ├── stop_loss.py       # 止损引擎
│   ├── drawdown_protection.py
│   ├── exposure_control.py
│   ├── volatility_filter.py
│   └── risk_dashboard.py
│
├── execution/              # 交易执行层
│   ├── order_manager.py   # 订单管理
│   ├── smart_router.py    # 智能路由
│   ├── twap_engine.py     # TWAP 引擎
│   ├── vwap_engine.py     # VWAP 引擎
│   ├── liquidity_scanner.py
│   └── exchange_adapter.py
│
├── monitor/                # 监控系统
│   ├── system_health.py
│   ├── strategy_performance.py
│   ├── trade_logger.py
│   ├── alert_system.py
│   └── dashboard_api.py
│
├── ai_automation/          # AI 自动化
│   ├── auto_strategy_generator.py
│   ├── auto_parameter_optimizer.py
│   ├── auto_backtest_runner.py
│   ├── auto_code_refactor.py
│   └── auto_bug_fix.py
│
├── main.py                 # 主入口
├── requirements.txt        # Python 依赖
├── .env.example           # 环境变量模板
├── docker-compose.yml     # Docker 编排
├── Dockerfile             # Docker 镜像
└── README.md              # 说明文档
```

---

### 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| 数据处理 | Pandas, NumPy |
| 交易接口 | CCXT |
| Web API | FastAPI |
| 缓存 | Redis |
| 数据库 | PostgreSQL |
| AI 模型 | Ollama + Qwen2.5-Coder |
| 容器化 | Docker |

---

### 🚀 快速开始

#### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-repo/ai_quant_system.git
cd ai_quant_system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置
nano .env
```

#### 3. 安装 Ollama 和模型

```bash
# 安装 Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# 拉取 Qwen2.5-Coder 模型
ollama pull qwen2.5-coder:latest

# 启动 Ollama 服务
ollama serve
```

#### 4. 启动系统

```bash
# 直接运行
python main.py

# 或使用 Docker
docker-compose up -d
```

---

### 📊 MVP 流程演示

系统启动后会自动运行 MVP 演示：

1. **获取 BTC 行情** - 从交易所获取实时数据
2. **AI 生成策略** - 使用 Qwen2.5-Coder 生成交易策略
3. **回测策略** - 在历史数据上测试策略表现
4. **风控检查** - 验证风险指标是否合规
5. **模拟执行** - 模拟下单交易

---

### 🤖 AI Agent 功能

| Agent | 功能 |
|-------|------|
| PlannerAgent | 任务规划与协调 |
| ResearchAgent | 市场研究分析 |
| StrategyAgent | 策略生成与管理 |
| CodingAgent | 代码生成与修改 |
| BacktestAgent | 策略回测 |
| RiskAgent | 风险分析管理 |
| ExecutionAgent | 交易执行管理 |
| MonitoringAgent | 系统监控 |
| OptimizationAgent | 参数优化 |
| MemoryAgent | 记忆管理 |
| SelfImprovementAgent | 自我改进 |

---

### 📈 支持的策略

1. **趋势跟踪策略** - 基于均线交叉
2. **均值回归策略** - 布林带 + RSI
3. **动量策略** - 价格动量 + 成量确认
4. **统计套利** - 协整对冲
5. **资金费率套利** - 期现套利
6. **做市策略** - 流动性提供

---

### ⚠️ 风险警告

**本系统仅供学习和研究使用！**

- 所有策略在实盘使用前必须经过充分测试
- 加密货币交易具有极高风险
- 请使用测试网络 (Testnet) 进行开发测试
- 永远不要投入超过您能承受损失的金额

---

### 📝 License

MIT License

---

### 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

### 🖥️ Streamlit Dashboard

系统包含一个完整的 Web 可视化界面：

```bash
# 启动 Dashboard
streamlit run dashboard/streamlit_app.py
```

**Dashboard 功能：**

| 模块 | 功能 |
|------|------|
| 📈 市场数据 | 实时价格图表、OHLCV K线、交易信号 |
| 📊 性能分析 | 收益曲线、Sharpe Ratio、回撤分析 |
| 💼 持仓管理 | 当前持仓、历史交易、订单状态 |
| ⚠️ 风控面板 | 风险敞口、账户余额、最大回撤监控 |
| 🤖 AI Agent | 11个Agent状态、任务队列、执行历史 |

**自动刷新：** 每5秒自动更新数据

---

### 📞 联系方式

- Issues: [GitHub Issues](https://github.com/your-repo/ai_quant_system/issues)
