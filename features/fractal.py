# features/fractal.py
"""
模块 18: 分形维度
================
量化市场粗糙程度，识别庄家控盘痕迹
"""

import numpy as np
from typing import Tuple, Optional


class FractalDimension:
    """
    分形维度计算器
    
    功能：
    - 计算价格曲线的分形维度
    - 识别市场状态
    
    解释：
    D ≈ 1.0: 光滑曲线，趋势明确
    D ≈ 1.5: 随机游走，震荡市场
    D > 1.5: 粗糙曲线，高波动/控盘
    """
    
    def __init__(self, window: int = 100):
        """
        Args:
            window: 计算窗口
        """
        self.window = window
    
    def calculate(self, prices: np.ndarray) -> Tuple[float, str]:
        """
        计算分形维度 (使用盒子计数法简化版)
        
        Args:
            prices: 价格序列
            
        Returns:
            (分形维度, 市场状态)
        """
        if len(prices) < self.window:
            return 1.5, "数据不足"
        
        prices = prices[-self.window:]
        
        try:
            # 简化方法：基于Higuchi分形维度
            L = []
            k_max = min(10, len(prices) // 4)
            
            for k in range(1, k_max + 1):
                L_k = []
                for m in range(k):
                    idx = np.arange(m, len(prices), k)
                    if len(idx) > 1:
                        segment = prices[idx]
                        length = np.sum(np.abs(np.diff(segment)))
                        L_k.append(length * (len(prices) - 1) / (k * len(idx)))
                
                if L_k:
                    L.append(np.mean(L_k))
            
            if len(L) >= 3:
                # 线性回归求斜率
                log_k = np.log(np.arange(1, len(L) + 1))
                log_L = np.log(L)
                coeffs = np.polyfit(log_k, log_L, 1)
                dimension = -coeffs[0]  # 斜率取负
            else:
                dimension = 1.5
            
            # 限制范围
            dimension = max(1.0, min(2.0, dimension))
            
            # 状态判断
            if dimension < 1.2:
                state = "光滑趋势"
            elif dimension < 1.4:
                state = "趋势主导"
            elif dimension < 1.6:
                state = "随机震荡"
            elif dimension < 1.8:
                state = "高波动"
            else:
                state = "极端粗糙"
            
            return dimension, state
            
        except Exception:
            return 1.5, "计算失败"


def calculate_fractal_dimension(prices: np.ndarray) -> Tuple[float, str]:
    """快速计算分形维度"""
    fd = FractalDimension()
    return fd.calculate(prices)
