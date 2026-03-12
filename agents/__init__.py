"""
AI Agents module for AI Quant Trading System.
Multi-Agent Software Factory architecture with:
- Planner, Research, Strategy, Coding agents
- Code Reviewer (quality gate)
- Debugger (with retry limits)
- File Manager (safe file operations)
- Risk, Execution, Monitoring, Optimization agents
- Memory and Self-Improvement agents
"""

from .base_agent import BaseAgent, AgentState, AgentTask
from .planner import PlannerAgent
from .research import ResearchAgent
from .strategy_agent import StrategyAgent
from .coding import CodingAgent
from .backtest_agent import BacktestAgent
from .risk_agent import RiskAgent
from .execution_agent import ExecutionAgent
from .monitoring_agent import MonitoringAgent
from .optimization import OptimizationAgent
from .memory_agent import MemoryAgent
from .self_improvement import SelfImprovementAgent

# New agents for improved architecture
from .code_reviewer import CodeReviewer, CodeIssue, IssueSeverity
from .debugger import DebuggerAgent, DebugSession, FixStatus
from .file_manager import FileManager, FileVersion

__all__ = [
    # Base
    'BaseAgent',
    'AgentState',
    'AgentTask',
    # Core agents
    'PlannerAgent',
    'ResearchAgent',
    'StrategyAgent',
    'CodingAgent',
    'BacktestAgent',
    'RiskAgent',
    'ExecutionAgent',
    'MonitoringAgent',
    'OptimizationAgent',
    'MemoryAgent',
    'SelfImprovementAgent',
    # New agents (v2)
    'CodeReviewer',
    'CodeIssue',
    'IssueSeverity',
    'DebuggerAgent',
    'DebugSession',
    'FixStatus',
    'FileManager',
    'FileVersion'
]
