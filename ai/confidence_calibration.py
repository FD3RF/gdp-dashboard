"""
置信度校准模块 (Confidence Calibration)
=====================================
将模型输出映射为真实概率

方法：
1. Platt Scaling - 逻辑回归校准
2. Isotonic Regression - 保序回归
3. Temperature Scaling - 温度缩放
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """校准结果"""
    original_confidence: float
    calibrated_confidence: float
    calibration_method: str
    calibration_score: float  # 校准质量评分


class ConfidenceCalibrator:
    """
    置信度校准器
    
    问题：模型输出的置信度（如0.9）可能不代表真实的90%胜率
    解决：使用历史数据校准，将模型输出映射为真实概率
    
    示例：
    - 模型输出 0.9 → 历史数据显示实际胜率只有 55%
    - 校准后：0.9 → 0.55
    """
    
    def __init__(self, method: str = "platt"):
        """
        Args:
            method: 校准方法
                - "platt": Platt Scaling (逻辑回归)
                - "isotonic": 保序回归 (非参数)
                - "temperature": 温度缩放 (神经网络常用)
        """
        self.method = method
        self._platt_model: Optional[LogisticRegression] = None
        self._isotonic_model: Optional[IsotonicRegression] = None
        self._temperature: float = 1.0
        self._is_fitted = False
        self._calibration_data: List[Tuple[float, int]] = []  # (confidence, win=1/loss=0)
    
    def add_sample(self, confidence: float, actual_result: int):
        """
        添加校准样本
        
        Args:
            confidence: 原始置信度 (0-1)
            actual_result: 实际结果 (1=成功, 0=失败)
        """
        self._calibration_data.append((confidence, actual_result))
        
        # 样本足够时自动校准
        if len(self._calibration_data) >= 20 and len(self._calibration_data) % 10 == 0:
            self.fit()
    
    def fit(self) -> Dict:
        """
        拟合校准模型
        
        Returns:
            拟合结果统计
        """
        if len(self._calibration_data) < 10:
            logger.warning("Insufficient data for calibration")
            return {"status": "insufficient_data", "samples": len(self._calibration_data)}
        
        # 准备数据
        X = np.array([c for c, _ in self._calibration_data]).reshape(-1, 1)
        y = np.array([r for _, r in self._calibration_data])
        
        # 计算校准前误差
        original_ece = self._calculate_ece(X.flatten(), y)
        
        if self.method == "platt":
            # Platt Scaling
            self._platt_model = LogisticRegression()
            self._platt_model.fit(X, y)
            
            # 计算校准后误差
            calibrated = self._platt_model.predict_proba(X)[:, 1]
            calibrated_ece = self._calculate_ece(calibrated, y)
            
        elif self.method == "isotonic":
            # Isotonic Regression
            self._isotonic_model = IsotonicRegression(out_of_bounds='clip')
            self._isotonic_model.fit(X.flatten(), y)
            
            # 计算校准后误差
            calibrated = self._isotonic_model.predict(X.flatten())
            calibrated_ece = self._calculate_ece(calibrated, y)
            
        elif self.method == "temperature":
            # Temperature Scaling
            self._temperature = self._optimize_temperature(X.flatten(), y)
            
            # 计算校准后误差
            calibrated = self._apply_temperature(X.flatten())
            calibrated_ece = self._calculate_ece(calibrated, y)
        
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self._is_fitted = True
        
        result = {
            "status": "fitted",
            "method": self.method,
            "samples": len(self._calibration_data),
            "original_ece": original_ece,
            "calibrated_ece": calibrated_ece,
            "improvement": original_ece - calibrated_ece,
        }
        
        logger.info(f"Calibration fitted: ECE {original_ece:.4f} → {calibrated_ece:.4f}")
        
        return result
    
    def calibrate(self, confidence: float) -> CalibrationResult:
        """
        校准单个置信度
        
        Args:
            confidence: 原始置信度 (0-1)
        
        Returns:
            CalibrationResult
        """
        if not self._is_fitted:
            return CalibrationResult(
                original_confidence=confidence,
                calibrated_confidence=confidence,
                calibration_method="none",
                calibration_score=0.0
            )
        
        X = np.array([[confidence]])
        
        if self.method == "platt" and self._platt_model:
            calibrated = self._platt_model.predict_proba(X)[0, 1]
        elif self.method == "isotonic" and self._isotonic_model:
            calibrated = self._isotonic_model.predict([confidence])[0]
        elif self.method == "temperature":
            calibrated = self._apply_temperature(np.array([confidence]))[0]
        else:
            calibrated = confidence
        
        return CalibrationResult(
            original_confidence=confidence,
            calibrated_confidence=calibrated,
            calibration_method=self.method,
            calibration_score=1.0 - abs(calibrated - confidence)  # 越接近1说明校准越少
        )
    
    def calibrate_batch(self, confidences: List[float]) -> List[CalibrationResult]:
        """批量校准"""
        return [self.calibrate(c) for c in confidences]
    
    def get_calibration_curve(self, n_bins: int = 10) -> Dict:
        """
        获取校准曲线数据
        
        Returns:
            {
                "bins": [(low, high), ...],
                "predicted": [avg_confidence_per_bin, ...],
                "actual": [actual_win_rate_per_bin, ...],
                "counts": [count_per_bin, ...]
            }
        """
        if len(self._calibration_data) < 10:
            return {"status": "insufficient_data"}
        
        confidences = [c for c, _ in self._calibration_data]
        results = [r for _, r in self._calibration_data]
        
        bins = np.linspace(0, 1, n_bins + 1)
        predicted = []
        actual = []
        counts = []
        
        for i in range(n_bins):
            mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i + 1])
            if mask.sum() > 0:
                predicted.append(np.array(confidences)[mask].mean())
                actual.append(np.array(results)[mask].mean())
                counts.append(mask.sum())
            else:
                predicted.append((bins[i] + bins[i + 1]) / 2)
                actual.append(np.nan)
                counts.append(0)
        
        return {
            "bins": [(bins[i], bins[i + 1]) for i in range(n_bins)],
            "predicted": predicted,
            "actual": actual,
            "counts": counts,
        }
    
    def _calculate_ece(self, predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """
        计算期望校准误差 (Expected Calibration Error)
        
        ECE = Σ |B_i|/n * |acc(B_i) - conf(B_i)|
        """
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
            if mask.sum() > 0:
                bin_acc = actual[mask].mean()
                bin_conf = predicted[mask].mean()
                ece += abs(bin_acc - bin_conf) * mask.sum() / len(predicted)
        
        return ece
    
    def _optimize_temperature(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """优化温度参数"""
        from scipy.optimize import minimize_scalar
        
        def nll(temp):
            scaled = self._apply_temperature_with_param(logits, temp)
            # 负对数似然
            nll = -np.mean(labels * np.log(scaled + 1e-8) + (1 - labels) * np.log(1 - scaled + 1e-8))
            return nll
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        return result.x
    
    def _apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        """应用温度缩放"""
        return self._apply_temperature_with_param(logits, self._temperature)
    
    def _apply_temperature_with_param(self, logits: np.ndarray, temp: float) -> np.ndarray:
        """应用温度缩放（指定参数）"""
        # 将置信度转换为 logit 空间
        eps = 1e-8
        logits_clipped = np.clip(logits, eps, 1 - eps)
        logit = np.log(logits_clipped / (1 - logits_clipped))
        
        # 应用温度
        scaled_logit = logit / temp
        
        # 转回概率
        return 1 / (1 + np.exp(-scaled_logit))
    
    def get_stats(self) -> Dict:
        """获取校准器统计信息"""
        return {
            "method": self.method,
            "is_fitted": self._is_fitted,
            "samples": len(self._calibration_data),
            "temperature": self._temperature if self.method == "temperature" else None,
            "min_samples_for_fit": 10,
        }


# 全局校准器实例
_calibrator: Optional[ConfidenceCalibrator] = None


def get_calibrator(method: str = "platt") -> ConfidenceCalibrator:
    """获取全局校准器"""
    global _calibrator
    if _calibrator is None:
        _calibrator = ConfidenceCalibrator(method)
    return _calibrator


def calibrate_confidence(confidence: float, method: str = "platt") -> float:
    """校准单个置信度（便捷函数）"""
    calibrator = get_calibrator(method)
    result = calibrator.calibrate(confidence)
    return result.calibrated_confidence
