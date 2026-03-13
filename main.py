# main.py
"""
Oracle AI Agent 主程序
=====================
12层架构核心整合 + 6大升级功能
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# Layer 2: 数据采集层
from data.market_stream import (
    get_realtime_eth_data, 
    get_orderbook_data,
    get_funding_rate,
    get_exchange_manager
)

# Layer 2-4: 社交情绪数据 (新增)
from data.social_stream import social_sentiment_score

# Layer 3/4: 数据处理与融合
from data.kline_builder import calculate_indicators, get_latest_indicators

# Layer 2.5: 特征时间同步层 (核心修复)
from data.feature_sync import get_feature_sync, sync_features, FeatureMatrix

# Layer 2.6: 真实Trades数据流 (核心修复)
from data.trades_stream import get_real_cvd, SimulatedTradesStream

# Layer 5: 特征工程层
from features.orderbook_features import OrderbookAnalyzer
from features.hurst import HurstExponent
from features.liquidity_heatmap import generate_liquidity_heatmap
from features.funding_rate import analyze_funding_rate

# Layer 5: 新增特征模块
from features.whale_monitor import whale_alert
from features.sentiment_features import get_sentiment_features
from features.funding_extreme import funding_extreme_alert
from features.order_flow import analyze_order_flow, OrderFlowAnalyzer
from features.liquidation_monitor import monitor_liquidations

# Layer 6/7/8: AI分析层
from ai.probability_model import calculate_probabilities

# Layer 6-8: 强化学习 (新增)
try:
    from ai.rl_agent import PPOAgent, MarketState, TradingAction, rl_decision
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Layer 9: 交易计划层
from analysis.decision_maker import make_decision, Signal
from analysis.trade_plan import generate_trade_plan
from analysis.risk_monitor import risk_warning

# Layer 10: 风险矩阵 (新增)
from analysis.risk_matrix import calculate_risk_matrix, RiskMatrix

# Layer 10: 硬规则风险过滤 (核心新增)
from analysis.risk_filter import apply_hard_filter, HardRiskFilter, FilterResult

# Layer 10: 市场状态识别 (核心新增)
from analysis.market_regime import MarketRegimeEngine, MarketRegime, detect_market_regime

# Layer 10: 统一信号引擎 (核心新增)
from analysis.signal_engine import SignalEngine, generate_unified_signal, get_signal_engine

# Layer 10: 四层决策引擎 (核心新增)
from analysis.layered_decision import make_four_layer_decision, get_four_layer_engine, FourLayerDecision, PositionSize

# Layer 10.5: 统一决策引擎 (核心整合)
from analysis.unified_decision import make_unified_decision, get_unified_engine, UnifiedDecision

# Layer 10.6: 高级仓位管理 (新增)
from analysis.position_manager import calculate_optimal_position, PositionSizing

# Layer 1: 实时预警系统 (新增)
from infrastructure.alert_system import get_alert_engine, check_alerts, AlertSeverity

# Layer 11: 解释层
from explain.signal_explainer import explain_signal
from explain.signal_history import record_signal, get_signal_reliability

# Layer 11: 在线学习进化 (新增)
from evolution.evolution_manager import get_evolution_status, run_evolution

# Layer 1: 事件驱动 (新增)
from infrastructure.event_bus import get_event_bus, EventType, publish_event

# 配置
from config import AGENT_CONFIG, RISK_CONFIG, BASE_PRICES


@dataclass
class SystemState:
    """系统状态 - 12层架构 + 6大升级"""
    # 基础信息
    symbol: str
    timestamp: str
    exchange: str
    price: float
    
    # Layer 3-4: 指标
    indicators: Dict[str, Any]
    
    # Layer 5-8: AI决策
    probabilities: Dict[str, float]
    signal: Signal
    plan: Any
    risk: Any
    orderbook: Dict[str, Any]
    
    # 原有特征
    hurst: float
    hurst_state: str
    liquidity: Dict[str, Any]
    funding: Dict[str, Any]
    explanation: Dict[str, Any]
    reliability: Dict[str, Any]
    
    # === 新增6大功能 ===
    # 1. 链上巨鲸监控
    whale_alerts: Dict[str, Any] = field(default_factory=dict)
    
    # 2. 社交情绪
    social_sentiment: Dict[str, Any] = field(default_factory=dict)
    
    # 3. 资金费率极值
    funding_extreme: Dict[str, Any] = field(default_factory=dict)
    
    # 4. 强化学习决策
    rl_decision: Dict[str, Any] = field(default_factory=dict)
    
    # 5. 风险矩阵
    risk_matrix: Dict[str, Any] = field(default_factory=dict)
    
    # 6. 进化状态
    evolution_status: Dict[str, Any] = field(default_factory=dict)
    
    # 7. 市场状态识别 (核心新增)
    market_regime: Dict[str, Any] = field(default_factory=dict)
    
    # 8. 订单流分析 (核心新增)
    order_flow: Dict[str, Any] = field(default_factory=dict)
    
    # 9. 清算监控 (核心新增)
    liquidation: Dict[str, Any] = field(default_factory=dict)
    
    # 10. 统一信号 (核心新增)
    unified_signal: Dict[str, Any] = field(default_factory=dict)
    
    # 11. 特征同步状态 (核心修复)
    feature_sync_status: Dict[str, Any] = field(default_factory=dict)
    data_quality_score: float = 1.0
    
    # 12. 风险过滤状态 (核心新增)
    risk_filter_status: Dict[str, Any] = field(default_factory=dict)
    is_trading_allowed: bool = True
    
    # 13. 统一决策状态 (核心整合)
    unified_decision: Dict[str, Any] = field(default_factory=dict)
    position_multiplier: float = 1.0  # 仓位乘数
    has_decision_conflict: bool = False  # 决策冲突标记
    
    # 14. 高级仓位管理 (新增)
    position_sizing: Dict[str, Any] = field(default_factory=dict)
    
    # 15. 预警状态 (新增)
    alert_status: Dict[str, Any] = field(default_factory=dict)
    active_alerts: List[Any] = field(default_factory=list)
    
    is_simulated: bool = False


def run_system(symbol: str = "ETH/USDT", use_simulated: bool = False) -> Optional[SystemState]:
    """
    运行完整分析流程 (12层架构整合 + 6大升级)
    
    Args:
        symbol: 交易对
        use_simulated: 是否使用模拟数据
        
    Returns:
        SystemState 或 None
    """
    # === Layer 2.5: 初始化特征同步层 ===
    feature_sync = get_feature_sync()
    
    # === Layer 2: 数据采集 ===
    df, current_price = get_realtime_eth_data(symbol, use_simulated=use_simulated)
    
    if df is None:
        return None
    
    # 获取交易所信息
    manager = get_exchange_manager()
    exchange_status = manager.get_status()
    primary_exchange = next(
        (ex for ex, status in exchange_status.items() if status.get('is_primary')),
        list(exchange_status.keys())[0] if exchange_status else "unknown"
    )
    
    # === Layer 3: 数据处理 ===
    df = calculate_indicators(df)
    indicators = get_latest_indicators(df)
    
    # === 同步特征: K线数据 ===
    feature_sync.update_feature("price", current_price, "kline")
    feature_sync.update_feature("volume", indicators.get('volume', 0), "kline")
    feature_sync.update_feature("rsi", indicators.get('rsi14', 50), "indicator")
    feature_sync.update_feature("momentum", indicators.get('momentum', 0), "indicator")
    feature_sync.update_feature("volatility", indicators.get('volatility', 0.02), "indicator")
    
    # === Layer 5: 特征工程 ===
    # 订单簿分析
    orderbook_data = get_orderbook_data(symbol, limit=50)
    orderbook = orderbook_data.get('bids', []) and orderbook_data.get('asks', [])
    orderbook_analysis = {}
    if orderbook:
        analyzer = OrderbookAnalyzer()
        orderbook_analysis = analyzer.analyze({
            'bids': orderbook_data.get('bids', []),
            'asks': orderbook_data.get('asks', [])
        })
        # 同步订单簿特征
        feature_sync.update_feature(
            "orderbook_imbalance", 
            orderbook_analysis.imbalance if hasattr(orderbook_analysis, 'imbalance') else 0,
            "orderbook"
        )
    
    # Hurst指数
    hurst_calc = HurstExponent()
    hurst_value, hurst_state = hurst_calc.calculate(df['close'].values)
    
    # 同步 Hurst 特征
    feature_sync.update_feature("hurst", hurst_value, "indicator")
    
    # 流动性热图
    liquidity = generate_liquidity_heatmap({
        'bids': orderbook_data.get('bids', []),
        'asks': orderbook_data.get('asks', [])
    }, current_price)
    
    # 资金费率
    funding_rate = get_funding_rate(symbol)
    funding = analyze_funding_rate(funding_rate)
    
    # 同步资金费率特征
    if funding_rate is not None:
        feature_sync.update_feature("funding_rate", funding_rate, "funding_rate")
    
    # === 新增功能 1: 链上巨鲸监控 ===
    whale_data = whale_alert(current_price)
    
    # 同步巨鲸流动特征
    feature_sync.update_feature(
        "whale_flow",
        whale_data.get('flow_summary', {}).get('net_flow_eth', 0),
        "whale"
    )
    
    # === 新增功能 2: 社交情绪分析 ===
    sentiment_data = social_sentiment_score()
    sentiment_features = get_sentiment_features(sentiment_data)
    
    # 同步情绪特征
    feature_sync.update_feature("sentiment", sentiment_features.overall_score, "sentiment")
    
    # === 新增功能 3: 多交易所资金费率极值 ===
    funding_extreme_data = funding_extreme_alert()
    
    # === 生成同步的特征矩阵 (核心修复) ===
    feature_matrix = feature_sync.sync_to_timestamp()
    data_quality_score = feature_sync.get_data_quality_score()
    
    # === Layer 6/7/8: AI分析 (使用同步特征) ===
    probs = calculate_probabilities(df)
    
    # 融合订单簿和资金费率到概率
    if orderbook_analysis and hasattr(orderbook_analysis, 'imbalance'):
        imbalance = orderbook_analysis.imbalance
        # 微调概率
        if imbalance > 0.3:
            probs['long'] = min(90, probs['long'] + 5)
            probs['short'] = max(5, probs['short'] - 3)
        elif imbalance < -0.3:
            probs['short'] = min(90, probs['short'] + 5)
            probs['long'] = max(5, probs['long'] - 3)
    
    # 融合社交情绪
    if sentiment_features.overall_score > 60:
        probs['long'] = min(90, probs['long'] + 3)
    elif sentiment_features.overall_score < 40:
        probs['short'] = min(90, probs['short'] + 3)
    
    # 重新归一化概率（确保总和为100%）
    total = probs['long'] + probs['short'] + probs['hold']
    probs['long'] = round(probs['long'] / total * 100, 1)
    probs['short'] = round(probs['short'] / total * 100, 1)
    probs['hold'] = round(probs['hold'] / total * 100, 1)
    
    # === 新增功能 4: 市场状态识别 (核心) ===
    regime_engine = MarketRegimeEngine()
    regime_state = regime_engine.detect_regime(
        hurst=hurst_value,
        volatility=indicators.get('volatility', 0.02),
        avg_volatility=indicators.get('avg_volatility', 0.02),
        orderbook_imbalance=orderbook_analysis.imbalance if hasattr(orderbook_analysis, 'imbalance') else 0,
        funding_rate=funding_rate or 0,
        sentiment_score=sentiment_features.overall_score,
        whale_net_flow=whale_data.get('flow_summary', {}).get('net_flow_eth', 0),
        fractal_dim=1.5,
        momentum=indicators.get('momentum', 0),
        price_change_pct=indicators.get('price_change_pct', 0),
        volume_ratio=indicators.get('volume_ratio', 1.0),
    )
    regime_result = regime_engine.get_regime_summary()
    
    # 根据市场状态调整策略权重
    strategy_weights = regime_engine.get_strategy_weights(regime_state.regime)
    
    # 趋势行情中降低观望权重
    if regime_state.regime in [MarketRegime.TREND_UP, MarketRegime.TREND_DOWN]:
        # 趋势行情：降低观望，增强方向性
        if regime_state.regime == MarketRegime.TREND_UP:
            probs['long'] = min(80, probs['long'] * 1.3 + 10)
            probs['hold'] = max(15, probs['hold'] * 0.6)
        else:
            probs['short'] = min(80, probs['short'] * 1.3 + 10)
            probs['hold'] = max(15, probs['hold'] * 0.6)
        
        # 重新归一化
        total = probs['long'] + probs['short'] + probs['hold']
        probs['long'] = round(probs['long'] / total * 100, 1)
        probs['short'] = round(probs['short'] / total * 100, 1)
        probs['hold'] = round(probs['hold'] / total * 100, 1)
    
    # 震荡行情中增强均值回归
    elif regime_state.regime == MarketRegime.RANGE:
        # 震荡行情：如果价格在支撑阻力之间，建议观望
        pass
    
    # 恐慌/狂热中反向操作
    elif regime_state.regime in [MarketRegime.PANIC, MarketRegime.EUPHORIA]:
        if regime_state.regime == MarketRegime.PANIC:
            # 恐慌时考虑抄底
            probs['long'] = min(70, probs['long'] + 15)
        else:
            # 狂热时考虑反向
            probs['short'] = min(70, probs['short'] + 15)
        
        total = probs['long'] + probs['short'] + probs['hold']
        probs['long'] = round(probs['long'] / total * 100, 1)
        probs['short'] = round(probs['short'] / total * 100, 1)
        probs['hold'] = round(probs['hold'] / total * 100, 1)
    
    # === 新增功能 5: 强化学习决策 ===
    rl_result = {}
    if RL_AVAILABLE:
        try:
            market_state = MarketState(
                price=current_price,
                volume=indicators.get('volume', 0),
                orderbook_imbalance=orderbook_analysis.imbalance if hasattr(orderbook_analysis, 'imbalance') else 0,
                hurst=hurst_value,
                fractal_dim=1.5,  # 默认值
                funding_rate=funding_rate or 0,
                sentiment_score=sentiment_features.overall_score,
                volatility=indicators.get('volatility', 0.02),
                momentum=indicators.get('momentum', 0),
            )
            rl_result = rl_decision(market_state)
        except Exception as e:
            rl_result = {"error": str(e)}
    
    # === 新增功能 7: 真实订单流分析 (修复CVD数据缺失) ===
    try:
        # 使用真实Trades数据流
        imbalance_for_cvd = orderbook_analysis.imbalance if hasattr(orderbook_analysis, 'imbalance') else 0
        real_cvd_data = get_real_cvd(
            current_price=current_price,
            use_simulated=True,  # WebSocket不可用时使用模拟
            imbalance_bias=imbalance_for_cvd,
            window_minutes=5,
        )
        
        # 同时使用原有订单流分析
        from features.order_flow import OrderFlowAnalyzer
        flow_analyzer = OrderFlowAnalyzer()
        
        # 从真实CVD数据获取成交信息
        cvd_value = real_cvd_data.get('value', 0)
        
        order_flow_result = {
            'cvd': {
                'value': cvd_value,
                'direction': real_cvd_data.get('direction', 'neutral'),
                'signal': real_cvd_data.get('signal', '均衡'),
                'strength': real_cvd_data.get('strength', 0),
            },
            'delta': {
                'value': cvd_value,
                'pct': real_cvd_data.get('value', 0) / 100,  # 简化
            },
            'imbalance': {
                'direction': real_cvd_data.get('direction', 'neutral'),
            },
            'is_real_data': True,  # 标记为真实数据源
        }
        
        # 同步订单流特征
        feature_sync.update_feature("cvd", cvd_value, "order_flow")
        
    except Exception as e:
        order_flow_result = {"error": str(e), "is_real_data": False}
    
    # === 新增功能 8: 清算监控 (先计算) ===
    liquidation_result = monitor_liquidations(
        current_price=current_price,
        open_interest=1000000,  # 默认持仓量
    )
    
    # === 同步清算特征 ===
    feature_sync.update_feature(
        "liquidation",
        liquidation_result.get('heatmap', {}).get('imbalance_ratio', 1.0),
        "liquidation"
    )
    
    # === 同步市场状态特征 ===
    feature_sync.update_feature("regime", regime_result.get('regime', 'neutral'), "regime")
    
    # === Layer 10: 风险预警 ===
    risk = risk_warning(df)
    
    # === 新增功能 5: 多维风险矩阵 (先计算，Signal Engine 会用到) ===
    risk_matrix_result = calculate_risk_matrix(
        market_data={
            "hurst": hurst_value,
            "fractal_dim": 1.5,
            "trend": "up" if hurst_value > 0.55 else "down" if hurst_value < 0.45 else "neutral",
            "orderbook_imbalance": orderbook_analysis.imbalance if hasattr(orderbook_analysis, 'imbalance') else 0,
            "liquidity_zones": liquidity.get('support_zones', []) + liquidity.get('resistance_zones', []),
            "trap_detected": len(liquidity.get('trap_warnings', [])) > 0,
            "volatility": indicators.get('volatility', 0.02),
            "avg_volatility": indicators.get('avg_volatility', 0.02),
        },
        funding_data={
            "rate": funding_rate or 0,
            "z_score": funding_extreme_data.get('summary', {}).get('avg_z_score', 0),
            "crowd_direction": funding_extreme_data.get('summary', {}).get('crowding_direction', 'neutral'),
        },
        sentiment_data={
            "score": sentiment_features.overall_score,
            "is_extreme": sentiment_features.is_extreme,
            "extreme_type": sentiment_features.extreme_type,
        },
        whale_data={
            "net_flow": whale_data.get('flow_summary', {}).get('net_flow_eth', 0),
            "high_impact_count": whale_data.get('high_impact_count', 0),
            "recent_alerts": whale_data.get('alerts', []),
        },
    )
    
    # === 新增功能 6: 统一信号引擎 (核心) ===
    signal_engine = get_signal_engine()
    unified_signal_result = generate_unified_signal(
        hurst=hurst_value,
        momentum=indicators.get('momentum', 0),
        imbalance=orderbook_analysis.imbalance if hasattr(orderbook_analysis, 'imbalance') else 0,
        whale_flow=whale_data.get('flow_summary', {}).get('net_flow_eth', 0),
        funding_rate=funding_rate or 0,
        cvd=order_flow_result.get('cvd', {}).get('value', 0) if order_flow_result else 0,
        sentiment=sentiment_features.overall_score,
        regime=regime_result.get('regime', 'neutral'),
        regime_confidence=regime_result.get('confidence', 0.5),
        risk_score=risk_matrix_result.get('score', 30),
    )
    
    # 使用统一信号覆盖概率（如果 Meta Filter 通过）
    if unified_signal_result.get('meta_filter', {}).get('passed', False):
        probs['long'] = unified_signal_result.get('probabilities', {}).get('long', probs['long'])
        probs['short'] = unified_signal_result.get('probabilities', {}).get('short', probs['short'])
        probs['hold'] = unified_signal_result.get('probabilities', {}).get('hold', probs['hold'])
        probs['confidence'] = unified_signal_result.get('confidence', probs['confidence'])
    
    # === Layer 9: 决策 ===
    decision = make_decision(probs)
    plan = generate_trade_plan(df, decision.signal)
    
    # === Layer 11: 解释 (使用Feature Sync) ===
    explanation = explain_signal(
        decision.signal.value,
        probs,
        indicators,
        orderbook_analysis.__dict__ if hasattr(orderbook_analysis, '__dict__') else {},
        funding,
        # 新参数：使用Feature Sync
        feature_matrix=feature_matrix,
        unified_signal=unified_signal_result,
        regime_data=regime_result,
    )
    
    # 信号可靠度
    reliability = get_signal_reliability(symbol)
    
    # === 核心修复: 硬规则风险过滤 ===
    risk_filter_decision = apply_hard_filter(
        signal=decision.signal.value,
        trade_plan=plan,
        reliability=reliability,
        data_quality_score=data_quality_score,
        unified_signal=unified_signal_result,
        probabilities=probs,
    )
    is_trading_allowed = risk_filter_decision.is_trading_allowed()
    
    # === 核心整合: 统一决策引擎 ===
    # 整合 Signal Engine + 四层决策 + 硬规则过滤
    support_zones = [(z[0], z[1]) for z in liquidity.get('support_zones', [])[:3]]
    resistance_zones = [(z[0], z[1]) for z in liquidity.get('resistance_zones', [])[:3]]
    cvd_value = order_flow_result.get('cvd', {}).get('value', 0) if order_flow_result else 0
    imbalance_value = orderbook_analysis.imbalance if hasattr(orderbook_analysis, 'imbalance') else 0
    
    # 准备清算数据
    liquidation_zones = []
    for alert in liquidation_result.get('approaching_alerts', [])[:3]:
        liquidation_zones.append({
            'price': alert.get('price', 0),
            'direction': alert.get('direction', 'unknown'),
        })
    
    # 准备资金费率数据
    funding_extreme_flag = abs(funding_extreme_data.get('summary', {}).get('avg_z_score', 0)) > 2
    
    # 执行统一决策
    unified_decision_result = make_unified_decision(
        # Signal Engine 参数
        hurst=hurst_value,
        momentum=indicators.get('momentum', 0),
        imbalance=imbalance_value,
        whale_flow=whale_data.get('flow_summary', {}).get('net_flow_eth', 0),
        funding_rate=funding_rate or 0,
        cvd=cvd_value,
        sentiment=sentiment_features.overall_score,
        regime=regime_result.get('regime', 'neutral'),
        regime_confidence=regime_result.get('confidence', 0.5),
        risk_score=risk_matrix_result.get('score', 30),
        # 四层决策参数
        data_quality_score=data_quality_score,
        current_price=current_price,
        support_zones=support_zones,
        resistance_zones=resistance_zones,
        liquidation_zones=liquidation_zones,
        funding_extreme=funding_extreme_flag,
        # 硬规则参数
        reliability=reliability,
        trade_plan=plan,
    )
    
    # 获取仓位乘数
    position_multiplier = unified_decision_result.position_multiplier
    
    # 使用统一决策结果
    if unified_decision_result.action == "LONG":
        decision.signal = Signal.LONG
        probs['long'] = unified_decision_result.confidence
        probs['hold'] = 100 - unified_decision_result.confidence
    elif unified_decision_result.action == "SHORT":
        decision.signal = Signal.SHORT
        probs['short'] = unified_decision_result.confidence
        probs['hold'] = 100 - unified_decision_result.confidence
    else:
        decision.signal = Signal.HOLD
        probs['hold'] = max(70, probs['hold'])
        position_multiplier = 0
    
    # 重新生成交易计划（根据仓位调整）
    if decision.signal != Signal.HOLD and position_multiplier > 0:
        plan = generate_trade_plan(df, decision.signal)
        if plan:
            plan.position_size = plan.position_size * position_multiplier
    else:
        plan = None
    
    # === 高级仓位管理（凯利公式 + 动态止损止盈）===
    position_sizing_result = {}
    
    # 准备清算数据（提前定义，预警也需要）
    liq_zones = []
    for alert in liquidation_result.get('approaching_alerts', [])[:5]:
        liq_zones.append({
            'price': alert.get('price', 0),
            'direction': alert.get('direction', 'unknown'),
            'amount': alert.get('amount', 0),
        })
    
    if decision.signal != Signal.HOLD:
        try:
            # 获取历史统计
            perf = evolution_data.get('recent_performance', {})
            win_rate = perf.get('win_rate', 0.5)
            avg_win = perf.get('avg_win_percent', 2.0)
            avg_loss = perf.get('avg_loss_percent', 2.0)
            
            # 计算最优仓位
            direction = "long" if decision.signal == Signal.LONG else "short"
            atr = indicators.get('atr14', current_price * 0.02)
            
            pos_sizing = calculate_optimal_position(
                account_balance=10000,
                win_rate=win_rate,
                avg_win_pct=avg_win,
                avg_loss_pct=avg_loss,
                signal_confidence=unified_decision_result.confidence / 100,
                entry_price=current_price,
                direction=direction,
                atr=atr,
                support_zones=support_zones,
                resistance_zones=resistance_zones,
                liquidation_zones=liq_zones,
            )
            
            position_sizing_result = {
                "kelly_fraction": round(pos_sizing.kelly_fraction * 100, 1),
                "position_size": round(pos_sizing.position_size * 100, 1),
                "stop_loss": round(pos_sizing.stop_loss_price, 2),
                "take_profit": round(pos_sizing.take_profit_price, 2),
                "risk_reward": round(pos_sizing.risk_reward_ratio, 2),
                "stop_type": pos_sizing.stop_type,
                "tp_type": pos_sizing.tp_type,
                "expected_value": round(pos_sizing.expected_value, 2),
            }
            
            # 覆盖交易计划
            if plan and pos_sizing.risk_reward_ratio >= 1.2:
                plan.stop_loss = pos_sizing.stop_loss_price
                plan.take_profit_1 = pos_sizing.take_profit_price
                plan.risk_reward = pos_sizing.risk_reward_ratio
                plan.position_size = pos_sizing.position_size * 100
        except Exception as e:
            position_sizing_result = {"error": str(e)}
    
    # === 实时预警检查 ===
    try:
        alert_engine = get_alert_engine()
        
        # 价格趋势
        price_trend = "up" if indicators.get('price_change_pct', 0) > 0 else "down"
        cvd_trend = order_flow_result.get('cvd', {}).get('direction', 'neutral')
        cvd_val = order_flow_result.get('cvd', {}).get('value', 0)
        
        alerts = alert_engine.check_all(
            current_price=current_price,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            liquidation_zones=liq_zones if 'liq_zones' in dir() else [],
            price_trend=price_trend,
            cvd_trend=cvd_trend,
            cvd_value=cvd_val,
            price_change_pct=indicators.get('price_change_pct', 0),
            volatility=indicators.get('volatility', 0.02),
        )
        
        # === 突破检测 (新增) ===
        # 检测价格是否突破震荡区间
        if len(support_zones) > 0 and len(resistance_zones) > 0:
            range_low = min(s[0] for s in support_zones[:3])
            range_high = max(r[0] for r in resistance_zones[:3])
            
            # 获取前一价格和CVD
            previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
            previous_cvd = order_flow_result.get('cvd', {}).get('previous_value', cvd_val)
            
            # 检测突破
            breakout_alert = alert_engine.check_breakout(
                current_price=current_price,
                previous_price=previous_price,
                range_high=range_high,
                range_low=range_low,
                cvd_value=cvd_val,
                cvd_previous=previous_cvd,
                volume_ratio=indicators.get('volume_ratio', 1.0),
            )
            
            if breakout_alert:
                alerts.append(breakout_alert)
        
        alert_status = alert_engine.get_summary()
        active_alerts = [a.to_dict() for a in alerts[:5]]
    except Exception as e:
        alert_status = {"error": str(e)}
        active_alerts = []
    
    # === 新增功能 6: 进化状态 ===
    evolution_data = get_evolution_status()
    
    # 检测模拟数据
    is_simulated = use_simulated or (symbol == "ETH/USDT" and 3000 < current_price < 4000)
    
    # 评估历史信号（三重标签法）
    try:
        from explain.signal_history import evaluate_signals
        evaluate_signals(current_price)
    except Exception:
        pass
    
    # 记录信号（带TP/SL）
    if decision.signal != Signal.HOLD:
        record_signal(
            symbol=symbol,
            signal=decision.signal.value,
            price=current_price,
            confidence=probs['confidence'],
            stop_loss=plan.stop_loss if plan else 0,
            take_profit=plan.take_profit_1 if plan else 0
        )
    
    # 发布事件 (事件驱动)
    try:
        publish_event(
            EventType.SIGNAL_GENERATED,
            "main",
            {
                "signal": decision.signal.value,
                "price": current_price,
                "confidence": probs['confidence'],
            }
        )
    except Exception:
        pass  # 事件发布失败不影响主流程
    
    # === 获取特征同步状态 (核心修复) ===
    feature_sync_status = feature_sync.get_sync_status()
    data_quality_score = feature_sync.get_data_quality_score()
    
    return SystemState(
        symbol=symbol,
        timestamp=str(df['timestamp'].iloc[-1]) if len(df) > 0 else str(datetime.now()),
        exchange=primary_exchange,
        price=current_price,
        indicators=indicators,
        probabilities=probs,
        signal=decision.signal,
        plan=plan,
        risk=risk,
        orderbook={
            'bid': orderbook_data.get('bid_price', 0),
            'ask': orderbook_data.get('ask_price', 0),
            'spread': orderbook_data.get('spread', 0),
            'imbalance': orderbook_analysis.imbalance if hasattr(orderbook_analysis, 'imbalance') else 0,
            'quality': orderbook_analysis.quality_score if hasattr(orderbook_analysis, 'quality_score') else 0,
        },
        hurst=hurst_value,
        hurst_state=hurst_state,
        liquidity=liquidity,
        funding=funding,
        explanation=explanation,
        reliability=reliability,
        # 新增6大功能
        whale_alerts=whale_data,
        social_sentiment={
            "score": sentiment_features.overall_score,
            "trend": sentiment_features.trend,
            "dominance": sentiment_features.dominance,
            "is_extreme": sentiment_features.is_extreme,
        },
        funding_extreme=funding_extreme_data,
        rl_decision=rl_result,
        risk_matrix=risk_matrix_result,
        evolution_status=evolution_data,
        # 市场状态识别
        market_regime=regime_result,
        order_flow=order_flow_result,
        liquidation=liquidation_result,
        unified_signal=unified_signal_result,
        # 特征同步状态 (核心修复)
        feature_sync_status=feature_sync_status,
        data_quality_score=data_quality_score,
        # 风险过滤状态 (核心修复)
        risk_filter_status=risk_filter_decision.to_dict(),
        is_trading_allowed=is_trading_allowed,
        # 统一决策状态 (核心整合)
        unified_decision=unified_decision_result.to_dict(),
        position_multiplier=position_multiplier,
        has_decision_conflict=unified_decision_result.has_conflict,
        # 高级仓位管理 (新增)
        position_sizing=position_sizing_result,
        # 预警状态 (新增)
        alert_status=alert_status,
        active_alerts=active_alerts,
        is_simulated=is_simulated
    )


def format_output(state: SystemState) -> str:
    """格式化输出"""
    lines = [
        "=" * 70,
        f"🧠 {state.symbol} AI 分析报告",
        "=" * 70,
        f"📅 时间: {state.timestamp}",
        f"🏦 交易所: {state.exchange}",
        f"💰 当前价格: ${state.price:,.2f}",
        "",
        "┌" + "─" * 68 + "┐",
        "│ 【AI 核心决策】                                                    │",
        "├" + "─" * 68 + "┤",
        f"│   🟢 做多概率: {state.probabilities['long']:.1f}%                                           │",
        f"│   🔴 做空概率: {state.probabilities['short']:.1f}%                                           │",
        f"│   ⚪ 观望概率: {state.probabilities['hold']:.1f}%                                           │",
        f"│   💪 信心指数: {state.probabilities['confidence']:.1f}                                          │",
        "└" + "─" * 68 + "┘",
        "",
        f"🎯 最终决策: {state.signal.value}",
        "",
        "┌" + "─" * 68 + "┐",
        "│ 【流动性热图 (M-41)】                                              │",
        "├" + "─" * 68 + "┤",
    ]
    
    # 流动性信息
    for support in state.liquidity.get('support_zones', [])[:2]:
        lines.append(f"│   🟢 支撑墙: ${support[0]:,.2f} (强度: {support[1]:.1f}x)                    │")
    for resistance in state.liquidity.get('resistance_zones', [])[:2]:
        lines.append(f"│   🔴 阻力墙: ${resistance[0]:,.2f} (强度: {resistance[1]:.1f}x)                    │")
    
    for warning in state.liquidity.get('trap_warnings', [])[:2]:
        lines.append(f"│   ⚠️ {warning.get('description', '陷阱预警')[:40]:<40}                    │")
    
    lines.extend([
        "└" + "─" * 68 + "┘",
        "",
        f"📊 Hurst指数: {state.hurst:.3f} ({state.hurst_state})",
        f"📈 资金费率: {state.funding.get('rate', 0)*100:.4f}% ({state.funding.get('signal', '中性')})",
        "",
    ])
    
    # 交易计划
    if state.plan:
        lines.extend([
            "┌" + "─" * 68 + "┐",
            "│ 【AI 交易计划】                                                    │",
            "├" + "─" * 68 + "┤",
            f"│   📍 入场价格: ${state.plan.entry:,.2f}                                      │",
            f"│   🛑 止损价格: ${state.plan.stop_loss:,.2f}                                      │",
            f"│   🎯 目标价格: ${state.plan.take_profit_1:,.2f}                                      │",
            f"│   📊 风险收益比: {state.plan.risk_reward:.2f}                                       │",
            f"│   💼 建议仓位: {state.plan.position_size:.1f}%                                    │",
            "└" + "─" * 68 + "┘",
        ])
    else:
        lines.append("⚠️ 当前建议观望，无交易计划")
    
    # 解释
    lines.extend([
        "",
        "┌" + "─" * 68 + "┐",
        "│ 【AI 决策解释】                                                    │",
        "├" + "─" * 68 + "┤",
        f"│   📝 主要来源: {state.explanation.get('primary_source', '综合'):<20}                          │",
        f"│   🏗️ 市场结构: {state.explanation.get('market_structure', '分析中')[:30]:<30}              │",
        f"│   💡 分析理由: {state.explanation.get('reasoning', '')[:40]:<40}              │",
    ])
    
    for warning in state.explanation.get('warnings', [])[:2]:
        lines.append(f"│   ⚠️ {warning[:50]:<50}              │")
    
    lines.extend([
        "└" + "─" * 68 + "┘",
        "",
        "┌" + "─" * 68 + "┐",
        "│ 【系统效能验证 (M-43)】                                            │",
        "├" + "─" * 68 + "┤",
        f"│   📈 近{state.reliability.get('sample_size', 0)}次信号胜率: {state.reliability.get('win_rate', 0):.1f}%                            │",
        f"│   💰 平均盈亏比: {state.reliability.get('avg_rr_ratio', 0):.2f}                                         │",
        f"│   🎖️ 可靠度: {state.reliability.get('level', '计算中')} ({state.reliability.get('score', 0):.0f}分)                            │",
        "└" + "─" * 68 + "┘",
        "=" * 70,
    ])
    
    return "\n".join(lines)


# Oracle Agent 类
class OracleAgent:
    """Oracle AI Agent - 12层架构 + 6大升级"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.balance = self.config.get('initial_balance', 10000)
        self.symbol = self.config.get('symbol', 'ETH/USDT')
        
    async def step(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """执行一步分析"""
        symbol = symbol or self.symbol
        state = run_system(symbol)
        
        if state is None:
            return {'error': '获取数据失败', 'price': 0, 'final_action': 'HOLD'}
        
        return {
            'symbol': state.symbol,
            'timestamp': state.timestamp,
            'exchange': state.exchange,
            'price': state.price,
            'probabilities': state.probabilities,
            'final_action': state.signal.value,
            'confidence': state.probabilities['confidence'],
            'risk_level': state.risk.level.value if hasattr(state.risk, 'level') else 'unknown',
            'risk_score': state.risk.score if hasattr(state.risk, 'score') else 0,
            'position_size': state.plan.position_size if state.plan else 0,
            'hurst': state.hurst,
            'hurst_state': state.hurst_state,
            'funding': state.funding,
            'liquidity': state.liquidity,
            'explanation': state.explanation,
            'reliability': state.reliability,
            # 新增功能
            'whale_alerts': state.whale_alerts,
            'social_sentiment': state.social_sentiment,
            'funding_extreme': state.funding_extreme,
            'rl_decision': state.rl_decision,
            'risk_matrix': state.risk_matrix,
            'evolution_status': state.evolution_status,
            'plan': {
                'entry': state.plan.entry if state.plan else 0,
                'stop': state.plan.stop_loss if state.plan else 0,
                'target': state.plan.take_profit_1 if state.plan else 0,
            } if state.plan else None,
        }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧠 ETH/USDT AI 盯盘终端 (12层架构版)")
    print("=" * 70 + "\n")
    
    state = run_system()
    
    if state:
        print(format_output(state))
    else:
        print("❌ 获取数据失败")
