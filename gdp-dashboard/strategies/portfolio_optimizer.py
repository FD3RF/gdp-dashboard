"""
Portfolio Optimization Engine.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from core.base import BaseModule


class PortfolioOptimizationEngine(BaseModule):
    """
    Portfolio Optimization Engine for multi-asset allocation.
    Supports Mean-Variance, Risk Parity, and other optimization methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('portfolio_optimizer', config)
        
        self._method = self.config.get('method', 'mean_variance')
        self._risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self._target_return = self.config.get('target_return', None)
        self._max_weight = self.config.get('max_weight', 0.3)
        self._min_weight = self.config.get('min_weight', 0.0)
        
        # Optimization results
        self._optimal_weights: Dict[str, float] = {}
        self._optimization_history: List[Dict] = []
    
    async def initialize(self) -> bool:
        """Initialize the optimizer."""
        self.logger.info("Initializing portfolio optimizer...")
        self._initialized = True
        return True
    
    async def start(self) -> bool:
        """Start the optimizer."""
        self._running = True
        self._start_time = datetime.now()
        return True
    
    async def stop(self) -> bool:
        """Stop the optimizer."""
        self._running = False
        return True
    
    async def optimize(
        self,
        returns: pd.DataFrame,
        method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights.
        
        Args:
            returns: DataFrame of asset returns (columns = assets)
            method: Optimization method
        
        Returns:
            Optimization results
        """
        method = method or self._method
        
        # Calculate expected returns and covariance
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        # Run optimization based on method
        if method == 'mean_variance':
            weights = self._mean_variance_optimization(expected_returns, cov_matrix)
        elif method == 'risk_parity':
            weights = self._risk_parity_optimization(cov_matrix)
        elif method == 'minimum_variance':
            weights = self._minimum_variance_optimization(cov_matrix)
        elif method == 'maximum_sharpe':
            weights = self._maximum_sharpe_optimization(expected_returns, cov_matrix)
        else:
            weights = self._equal_weight(len(returns.columns))
        
        # Calculate portfolio metrics
        portfolio_return = sum(weights[s] * expected_returns[s] for s in weights)
        portfolio_volatility = self._calculate_portfolio_volatility(weights, cov_matrix)
        sharpe_ratio = (portfolio_return - self._risk_free_rate) / portfolio_volatility
        
        result = {
            'method': method,
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'timestamp': datetime.now().isoformat()
        }
        
        self._optimal_weights = weights
        self._optimization_history.append(result)
        
        return result
    
    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Mean-Variance optimization."""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            symbols = list(expected_returns.index)
            
            def objective(weights):
                return -sum(w * expected_returns[s] for w, s in zip(weights, symbols))
            
            def constraint_volatility(weights):
                return 0.2 - self._calculate_portfolio_volatility(
                    dict(zip(symbols, weights)), cov_matrix
                )
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: sum(w) - 1},
            ]
            
            if self._target_return:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w: sum(w[i] * expected_returns[symbols[i]] 
                                        for i in range(len(w))) - self._target_return
                })
            
            bounds = [(self._min_weight, self._max_weight) for _ in range(n_assets)]
            initial_weights = self._equal_weight(n_assets)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return dict(zip(symbols, result.x))
            else:
                return self._equal_weight(n_assets, symbols)
                
        except ImportError:
            return self._equal_weight(len(expected_returns), list(expected_returns.index))
    
    def _risk_parity_optimization(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Risk Parity optimization."""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(cov_matrix)
            symbols = list(cov_matrix.columns)
            
            def objective(weights):
                portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
                marginal_risk = cov_matrix.values @ weights / portfolio_vol
                risk_contribution = weights * marginal_risk
                target_risk = portfolio_vol / n_assets
                return sum((rc - target_risk) ** 2 for rc in risk_contribution)
            
            constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1}]
            bounds = [(0.01, self._max_weight) for _ in range(n_assets)]
            initial_weights = self._equal_weight(n_assets)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return dict(zip(symbols, result.x))
            else:
                return self._equal_weight(n_assets, symbols)
                
        except ImportError:
            return self._equal_weight(len(cov_matrix), list(cov_matrix.columns))
    
    def _minimum_variance_optimization(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Minimum Variance optimization."""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(cov_matrix)
            symbols = list(cov_matrix.columns)
            
            def objective(weights):
                return weights @ cov_matrix.values @ weights
            
            constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1}]
            bounds = [(self._min_weight, self._max_weight) for _ in range(n_assets)]
            initial_weights = self._equal_weight(n_assets)
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return dict(zip(symbols, result.x))
            else:
                return self._equal_weight(n_assets, symbols)
                
        except ImportError:
            return self._equal_weight(len(cov_matrix), list(cov_matrix.columns))
    
    def _maximum_sharpe_optimization(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Maximum Sharpe Ratio optimization."""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            symbols = list(expected_returns.index)
            
            def neg_sharpe(weights):
                ret = sum(w * expected_returns[s] for w, s in zip(weights, symbols))
                vol = self._calculate_portfolio_volatility(
                    dict(zip(symbols, weights)), cov_matrix
                )
                return -(ret - self._risk_free_rate) / vol
            
            constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1}]
            bounds = [(self._min_weight, self._max_weight) for _ in range(n_assets)]
            initial_weights = self._equal_weight(n_assets)
            
            result = minimize(
                neg_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                return dict(zip(symbols, result.x))
            else:
                return self._equal_weight(n_assets, symbols)
                
        except ImportError:
            return self._equal_weight(len(expected_returns), list(expected_returns.index))
    
    def _equal_weight(
        self,
        n_assets: int,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Return equal weight allocation."""
        weight = 1.0 / n_assets
        if symbols:
            return {s: weight for s in symbols}
        return {str(i): weight for i in range(n_assets)}
    
    def _calculate_portfolio_volatility(
        self,
        weights: Dict[str, float],
        cov_matrix: pd.DataFrame
    ) -> float:
        """Calculate portfolio volatility."""
        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])
        cov = cov_matrix.loc[symbols, symbols].values
        
        return np.sqrt(w @ cov @ w)
    
    def calculate_risk_contribution(
        self,
        weights: Dict[str, float],
        cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate risk contribution of each asset."""
        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])
        cov = cov_matrix.loc[symbols, symbols].values
        
        portfolio_vol = np.sqrt(w @ cov @ w)
        marginal_risk = cov @ w / portfolio_vol
        risk_contribution = w * marginal_risk
        
        return dict(zip(symbols, risk_contribution))
    
    def get_optimal_weights(self) -> Dict[str, float]:
        """Get current optimal weights."""
        return self._optimal_weights.copy()
    
    def rebalance_orders(
        self,
        current_weights: Dict[str, float],
        target_weights: Optional[Dict[str, float]] = None,
        portfolio_value: float = 100000
    ) -> List[Dict[str, Any]]:
        """Generate rebalancing orders."""
        target = target_weights or self._optimal_weights
        
        orders = []
        for symbol in set(current_weights.keys()) | set(target.keys()):
            current = current_weights.get(symbol, 0)
            target_weight = target.get(symbol, 0)
            
            diff = target_weight - current
            if abs(diff) > 0.01:  # 1% threshold
                orders.append({
                    'symbol': symbol,
                    'current_weight': current,
                    'target_weight': target_weight,
                    'weight_change': diff,
                    'value_change': diff * portfolio_value,
                    'action': 'buy' if diff > 0 else 'sell'
                })
        
        return orders
