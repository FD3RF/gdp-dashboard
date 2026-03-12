# features/hurst.py
"""
模块 17: Hurst指数
==================
判断趋势持续性（趋势）还是回归性（震荡）
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class HurstExponent:
    """
    Hurst指数计算器
    
    功能：
    - 判断市场状态（趋势/震荡）
    - 预测价格行为持续性
    
    解释：
    H < 0.5: 均值回归（反趋势）
    H = 0.5: 随机游走
    H > 0.5: 趋势持续
    H > 0.7: 强趋势
    """
    
    def __init__(self, min_window: int = 20, max_window: int = 100):
        """
        Args:
            min_window: 最小窗口
            max_window: 最大窗口
        """
        self.min_window = min_window
        self.max_window = max_window
    
    def calculate(self, prices: np.ndarray) -> Tuple[float, str]:
        """
        计算Hurst指数 (R/S分析法)
        
        Args:
            prices: 价格序列
            
        Returns:
            (hurst_value, 市场状态描述)
        """
        if len(prices) < self.min_window:
            return 0.5, "数据不足"
        
        # 使用最近的数据
        prices = prices[-self.max_window:]
        
        try:
            # R/S 分析法
            n = len(prices)
            returns = np.diff(np.log(prices))
            
            # 计算不同窗口的R/S
            rs_values = []
            window_sizes = []
            
            for window in range(10, n // 2, 5):
                # 分割数据
                num_segments = n // window
                rs_sum = 0
                
                for i in range(num_segments):
                    segment = returns[i * window : (i + 1) * window]
                    
                    # 计算R/S
                    mean = np.mean(segment)
                    deviations = np.cumsum(segment - mean)
                    r = np.max(deviations) - np.min(deviations)
                    s = np.std(segment)
                    
                    if s > 0:
                        rs_sum += r / s
                
                if num_segments > 0:
                    rs_values.append(np.log(rs_sum / num_segments))
                    window_sizes.append(np.log(window))
            
            # 线性回归求斜率
            if len(rs_values) >= 3:
                coeffs = np.polyfit(window_sizes, rs_values, 1)
                hurst = coeffs[0]
            else:
                hurst = 0.5
            
            # 限制范围
            hurst = max(0, min(1, hurst))
            
            # 状态描述
            if hurst < 0.4:
                state = "强均值回归"
            elif hurst < 0.5:
                state = "均值回归"
            elif hurst < 0.55:
                state = "随机游走"
            elif hurst < 0.7:
                state = "趋势持续"
            else:
                state = "强趋势"
            
            return hurst, state
            
        except Exception:
            return 0.5, "计算失败"
    
    def get_trend_strength(self, prices: np.ndarray) -> float:
        """
        获取趋势强度
        
        Args:
            prices: 价格序列
            
        Returns:
            趋势强度 (-1 到 1)
        """
        hurst, _ = self.calculate(prices)
        
        # 将hurst转换为趋势强度
        strength = (hurst - 0.5) * 2
        
        return strength


def calculate_hurst(prices: np.ndarray) -> Tuple[float, str]:
    """快速计算Hurst指数"""
    h = HurstExponent()
    return h.calculate(prices)
