"""
数据一致性验证器
==================

确保本地数据与交易所数据保持一致。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import hashlib


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    timestamp: datetime
    details: Dict[str, Any]


class DataConsistencyValidator:
    """
    数据一致性验证器
    
    验证项目：
    1. 订单状态一致性
    2. 余额一致性
    3. 持仓一致性
    4. 行情数据完整性
    5. 时间戳有效性
    """
    
    # 允许的时间偏差（秒）
    MAX_TIME_DRIFT = 30
    
    # 数据有效期限（秒）
    DATA_EXPIRY = {
        'ticker': 10,
        'orderbook': 5,
        'balance': 60,
        'position': 60,
        'order': 120
    }
    
    def __init__(self):
        self.logger = logging.getLogger("DataConsistencyValidator")
        self._validation_history: List[ValidationResult] = []
    
    # ==================== 订单验证 ====================
    
    def validate_order_consistency(
        self,
        local_order: Dict[str, Any],
        exchange_order: Dict[str, Any]
    ) -> ValidationResult:
        """验证订单状态一致性"""
        errors = []
        warnings = []
        details = {}
        
        # 检查订单ID
        if local_order.get('order_id') != exchange_order.get('id'):
            errors.append(f"Order ID mismatch: local={local_order.get('order_id')}, exchange={exchange_order.get('id')}")
        
        # 检查状态
        local_status = local_order.get('status')
        exchange_status = exchange_order.get('status')
        
        status_map = {
            'open': ['pending', 'submitted', 'partial'],
            'closed': ['filled'],
            'canceled': ['cancelled']
        }
        
        valid_states = []
        for ex_state, local_states in status_map.items():
            if ex_state == exchange_status:
                valid_states = local_states
                break
        
        if local_status not in valid_states:
            errors.append(
                f"Status inconsistent: local={local_status}, exchange={exchange_status}"
            )
        
        # 检查成交数量
        local_filled = float(local_order.get('filled_quantity', 0))
        exchange_filled = float(exchange_order.get('filled', 0))
        
        if abs(local_filled - exchange_filled) > 0.0001:
            errors.append(
                f"Filled quantity mismatch: local={local_filled}, exchange={exchange_filled}"
            )
        
        details['local'] = local_order
        details['exchange'] = exchange_order
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            details=details
        )
        
        self._validation_history.append(result)
        return result
    
    # ==================== 余额验证 ====================
    
    def validate_balance_consistency(
        self,
        local_balance: Dict[str, float],
        exchange_balance: Dict[str, float],
        tolerance: float = 0.001
    ) -> ValidationResult:
        """验证余额一致性"""
        errors = []
        warnings = []
        details = {'mismatches': []}
        
        all_assets = set(local_balance.keys()) | set(exchange_balance.keys())
        
        for asset in all_assets:
            local_val = local_balance.get(asset, 0)
            exchange_val = exchange_balance.get(asset, 0)
            
            # 计算相对差异
            if exchange_val > 0:
                diff_pct = abs(local_val - exchange_val) / exchange_val
                
                if diff_pct > tolerance:
                    errors.append(
                        f"{asset} balance mismatch: local={local_val:.8f}, exchange={exchange_val:.8f} (diff={diff_pct*100:.2f}%)"
                    )
                    details['mismatches'].append({
                        'asset': asset,
                        'local': local_val,
                        'exchange': exchange_val,
                        'diff_pct': diff_pct
                    })
            elif local_val > 0 and exchange_val == 0:
                warnings.append(f"{asset} exists locally but not on exchange")
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            details=details
        )
        
        self._validation_history.append(result)
        return result
    
    # ==================== 持仓验证 ====================
    
    def validate_position_consistency(
        self,
        local_positions: List[Dict],
        exchange_positions: List[Dict]
    ) -> ValidationResult:
        """验证持仓一致性"""
        errors = []
        warnings = []
        details = {}
        
        # 转换为字典便于比较
        local_by_symbol = {p['symbol']: p for p in local_positions if p.get('quantity', 0) != 0}
        exchange_by_symbol = {p['symbol']: p for p in exchange_positions if float(p.get('contracts', 0)) != 0}
        
        all_symbols = set(local_by_symbol.keys()) | set(exchange_by_symbol.keys())
        
        for symbol in all_symbols:
            local_pos = local_by_symbol.get(symbol, {})
            exchange_pos = exchange_by_symbol.get(symbol, {})
            
            local_qty = float(local_pos.get('quantity', 0))
            exchange_qty = float(exchange_pos.get('contracts', 0))
            
            if abs(local_qty - exchange_qty) > 0.0001:
                errors.append(
                    f"{symbol} position mismatch: local={local_qty}, exchange={exchange_qty}"
                )
            
            # 检查方向
            local_side = local_pos.get('side', 'none')
            exchange_side = 'long' if exchange_qty > 0 else ('short' if exchange_qty < 0 else 'none')
            
            if local_qty > 0 and local_side != 'long':
                warnings.append(f"{symbol} local side inconsistent with quantity")
        
        details['local_count'] = len(local_by_symbol)
        details['exchange_count'] = len(exchange_by_symbol)
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            details=details
        )
        
        self._validation_history.append(result)
        return result
    
    # ==================== 数据新鲜度验证 ====================
    
    def validate_data_freshness(
        self,
        data_type: str,
        timestamp: datetime
    ) -> ValidationResult:
        """验证数据新鲜度"""
        errors = []
        warnings = []
        
        if data_type not in self.DATA_EXPIRY:
            warnings.append(f"Unknown data type: {data_type}")
        else:
            age = (datetime.now() - timestamp).total_seconds()
            expiry = self.DATA_EXPIRY[data_type]
            
            if age > expiry:
                errors.append(
                    f"{data_type} data expired: age={age:.1f}s, expiry={expiry}s"
                )
            elif age > expiry * 0.8:
                warnings.append(
                    f"{data_type} data nearing expiry: age={age:.1f}s"
                )
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            details={'age': (datetime.now() - timestamp).total_seconds()}
        )
    
    # ==================== 行情数据验证 ====================
    
    def validate_market_data(
        self,
        ticker: Dict[str, Any]
    ) -> ValidationResult:
        """验证行情数据完整性"""
        errors = []
        warnings = []
        required_fields = ['symbol', 'bid', 'ask', 'last']
        
        for field in required_fields:
            if field not in ticker:
                errors.append(f"Missing required field: {field}")
            elif ticker[field] is None:
                errors.append(f"Field is None: {field}")
        
        # 验证价格合理性
        if 'bid' in ticker and 'ask' in ticker:
            bid = ticker['bid']
            ask = ticker['ask']
            
            if bid and ask:
                if bid > ask:
                    errors.append(f"Invalid spread: bid({bid}) > ask({ask})")
                spread_pct = (ask - bid) / ask * 100
                
                if spread_pct > 5:
                    warnings.append(f"Large spread: {spread_pct:.2f}%")
        
        # 验证时间戳
        if 'timestamp' in ticker:
            ts = ticker['timestamp']
            if isinstance(ts, datetime):
                age = (datetime.now() - ts).total_seconds()
                if age > self.DATA_EXPIRY['ticker']:
                    errors.append(f"Ticker data too old: {age:.1f}s")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            details={'ticker': ticker}
        )
    
    # ==================== K线数据验证 ====================
    
    def validate_ohlcv_data(
        self,
        ohlcv: List[Dict]
    ) -> ValidationResult:
        """验证K线数据"""
        errors = []
        warnings = []
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if not ohlcv:
            errors.append("Empty OHLCV data")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                timestamp=datetime.now(),
                details={}
            )
        
        for i, candle in enumerate(ohlcv):
            # 检查必需字段
            for field in required_fields:
                if field not in candle:
                    errors.append(f"Candle {i}: missing {field}")
            
            # 检查价格逻辑
            o = candle.get('open', 0)
            h = candle.get('high', 0)
            l = candle.get('low', 0)
            c = candle.get('close', 0)
            
            if h < max(o, c):
                errors.append(f"Candle {i}: high < max(open, close)")
            if l > min(o, c):
                errors.append(f"Candle {i}: low > min(open, close)")
            
            # 检查负值
            if any(v < 0 for v in [o, h, l, c]):
                errors.append(f"Candle {i}: negative price detected")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            details={'candle_count': len(ohlcv)}
        )
    
    # ==================== 订单簿验证 ====================
    
    def validate_orderbook(
        self,
        orderbook: Dict[str, Any],
        min_depth: int = 10
    ) -> ValidationResult:
        """验证订单簿"""
        errors = []
        warnings = []
        
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids:
            errors.append("No bids in orderbook")
        if not asks:
            errors.append("No asks in orderbook")
        
        # 检查深度
        if len(bids) < min_depth:
            warnings.append(f"Low bid depth: {len(bids)}")
        if len(asks) < min_depth:
            warnings.append(f"Low ask depth: {len(asks)}")
        
        # 验证价格排序
        if len(bids) > 1:
            for i in range(len(bids) - 1):
                if bids[i][0] < bids[i+1][0]:
                    errors.append("Bids not sorted descending")
                    break
        
        if len(asks) > 1:
            for i in range(len(asks) - 1):
                if asks[i][0] > asks[i+1][0]:
                    errors.append("Asks not sorted ascending")
                    break
        
        # 检查交叉
        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            
            if best_bid >= best_ask:
                errors.append(f"Orderbook crossed: bid={best_bid} >= ask={best_ask}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            timestamp=datetime.now(),
            details={
                'bid_depth': len(bids),
                'ask_depth': len(asks)
            }
        )
    
    # ==================== 综合验证 ====================
    
    def full_validation(
        self,
        exchange_adapter,
        local_state: Dict[str, Any]
    ) -> Dict[str, ValidationResult]:
        """执行全面验证"""
        results = {}
        
        # 验证余额
        try:
            exchange_balance = exchange_adapter.get_balance()
            results['balance'] = self.validate_balance_consistency(
                local_state.get('balance', {}),
                exchange_balance.get('total', {})
            )
        except Exception as e:
            results['balance'] = ValidationResult(
                is_valid=False,
                errors=[f"Failed to validate balance: {e}"],
                warnings=[],
                timestamp=datetime.now(),
                details={}
            )
        
        # 验证持仓
        try:
            exchange_positions = exchange_adapter.get_positions()
            results['positions'] = self.validate_position_consistency(
                local_state.get('positions', []),
                exchange_positions
            )
        except Exception as e:
            results['positions'] = ValidationResult(
                is_valid=False,
                errors=[f"Failed to validate positions: {e}"],
                warnings=[],
                timestamp=datetime.now(),
                details={}
            )
        
        return results
    
    # ==================== 历史记录 ====================
    
    def get_validation_history(
        self,
        limit: int = 100
    ) -> List[ValidationResult]:
        """获取验证历史"""
        return self._validation_history[-limit:]
    
    def get_error_summary(self) -> Dict[str, int]:
        """获取错误统计"""
        error_counts = {}
        
        for result in self._validation_history:
            for error in result.errors:
                # 提取错误类型
                error_type = error.split(':')[0] if ':' in error else error[:30]
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return error_counts


# 导出
__all__ = ['DataConsistencyValidator', 'ValidationResult']
