#!/usr/bin/env python3
"""全面测试脚本"""
import sys
import traceback

print("=" * 60)
print("全面模块测试")
print("=" * 60)

modules = [
    ("core.base", "BaseModule"),
    ("core.constants", "TimeFrame"),
    ("core.utils", "async_retry"),
    ("core.memory", "MemoryStore"),
    ("core.security", "safe_json_parse"),
    ("config.settings", "Settings"),
    ("agents.base_agent", "BaseAgent"),
    ("agents.code_reviewer", "CodeReviewer"),
    ("agents.debugger", "DebuggerAgent"),
    ("agents.file_manager", "FileManager"),
    ("strategies.base_strategy", "BaseStrategy"),
    ("backtest.backtest_engine", "BacktestEngine"),
    ("backtest.performance_analyzer", "PerformanceAnalyzer"),
    ("backtest.precise_performance", "PrecisePerformanceCalculator"),
    ("risk.position_sizing", "PositionSizing"),
    ("execution.order_manager", "OrderManager"),
    ("execution.exchange_adapter", "ExchangeAdapter"),
    ("execution.exchange_sync_manager", "ExchangeSyncManager"),
    ("data.data_consistency", "DataConsistencyValidator"),
]

errors = []
for mod, cls in modules:
    try:
        exec(f"from {mod} import {cls}")
        print(f"✓ {mod}.{cls}")
    except Exception as e:
        print(f"✗ {mod}.{cls}: {str(e)[:80]}")
        errors.append((mod, traceback.format_exc()))

print("\n" + "=" * 60)
if errors:
    print(f"发现 {len(errors)} 个错误")
    for mod, err in errors:
        print(f"\n{'='*40}")
        print(f"模块: {mod}")
        print(err)
else:
    print("✅ 所有模块导入成功")
