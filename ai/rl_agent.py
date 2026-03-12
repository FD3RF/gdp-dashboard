"""
强化学习智能体模块 (Layer 6-8)
PPO 算法 + 自我博弈对抗训练
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from decimal import Decimal
import json

logger = logging.getLogger(__name__)

# 延迟导入 torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using simplified RL agent")


@dataclass
class MarketState:
    """市场状态"""
    price: float
    volume: float
    orderbook_imbalance: float
    hurst: float
    fractal_dim: float
    funding_rate: float
    sentiment_score: float
    volatility: float
    momentum: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_vector(self) -> np.ndarray:
        """转换为状态向量"""
        return np.array([
            self.price / 10000,  # 归一化
            self.volume / 1e6,
            self.orderbook_imbalance,
            self.hurst,
            self.fractal_dim - 1.5,  # 中心化
            self.funding_rate * 1000,
            self.sentiment_score / 100,
            self.volatility,
            self.momentum,
        ])


@dataclass
class TradingAction:
    """交易动作"""
    action_type: str  # "long", "short", "hold", "close"
    position_size: float  # 0.0 - 1.0
    confidence: float
    reasoning: str = ""


@dataclass
class Reward:
    """奖励结构"""
    pnl: float  # 盈亏
    risk_penalty: float  # 风险惩罚
    slippage_cost: float  # 滑点成本
    holding_cost: float  # 持仓成本
    total: float  # 总奖励


class ActorCriticNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Actor-Critic 网络结构
    Actor: 输出动作概率
    Critic: 输出状态价值
    """
    
    def __init__(self, state_dim: int = 9, action_dim: int = 4, hidden_dim: int = 128):
        if not TORCH_AVAILABLE:
            return
        
        super().__init__()
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor 头 - 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic 头 - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 仓位大小输出
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 0-1 之间
        )
    
    def forward(self, state):
        if not TORCH_AVAILABLE:
            return None, None, None
        
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        position_size = self.size_head(features)
        return action_probs, value, position_size


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 智能体
    支持自我博弈对抗训练
    """
    
    ACTION_MAP = {0: "hold", 1: "long", 2: "short", 3: "close"}
    
    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        use_torch: bool = True
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        # 状态归一化参数
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        
        if self.use_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        else:
            # 简化版本 - 使用规则 + 随机
            self.network = None
            self.optimizer = None
        
        # 训练数据缓冲
        self.memory: List[Dict] = []
        self.max_memory = 10000
        
        # 统计
        self.stats = {
            "total_steps": 0,
            "total_reward": 0.0,
            "win_trades": 0,
            "loss_trades": 0,
        }
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """归一化状态"""
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def select_action(
        self,
        state: MarketState,
        deterministic: bool = False
    ) -> TradingAction:
        """
        选择动作
        
        Args:
            state: 当前市场状态
            deterministic: 是否确定性选择（测试时用）
        
        Returns:
            交易动作
        """
        state_vec = self.normalize_state(state.to_vector())
        
        if self.use_torch:
            return self._select_action_torch(state_vec, deterministic)
        else:
            return self._select_action_rule(state)
    
    def _select_action_torch(
        self,
        state_vec: np.ndarray,
        deterministic: bool
    ) -> TradingAction:
        """使用神经网络选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            action_probs, value, position_size = self.network(state_tensor)
            
            if deterministic:
                action_idx = torch.argmax(action_probs, dim=-1).item()
            else:
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample().item()
            
            confidence = action_probs[0, action_idx].item()
            size = position_size.item()
            
            return TradingAction(
                action_type=self.ACTION_MAP[action_idx],
                position_size=size,
                confidence=confidence,
                reasoning=f"NN决策: 概率分布 {[f'{p:.2f}' for p in action_probs[0].tolist()]}"
            )
    
    def _select_action_rule(self, state: MarketState) -> TradingAction:
        """基于规则的决策（无torch时的替代方案）"""
        scores = {"hold": 0.0, "long": 0.0, "short": 0.0, "close": 0.0}
        reasons = []
        
        # Hurst 指数判断趋势
        if state.hurst > 0.55:
            scores["long"] += 0.2
            reasons.append("趋势向上(H>0.55)")
        elif state.hurst < 0.45:
            scores["short"] += 0.2
            reasons.append("均值回归(H<0.45)")
        
        # 订单簿失衡
        if state.orderbook_imbalance > 0.3:
            scores["long"] += 0.15
            reasons.append("买盘强势")
        elif state.orderbook_imbalance < -0.3:
            scores["short"] += 0.15
            reasons.append("卖盘强势")
        
        # 资金费率
        if state.funding_rate > 0.0005:
            scores["short"] += 0.1
            reasons.append("做多拥挤")
        elif state.funding_rate < -0.0005:
            scores["long"] += 0.1
            reasons.append("做空拥挤")
        
        # 情绪
        if state.sentiment_score > 60:
            scores["long"] += 0.1
        elif state.sentiment_score < 40:
            scores["short"] += 0.1
        
        # 动量
        if state.momentum > 0.5:
            scores["long"] += 0.15
        elif state.momentum < -0.5:
            scores["short"] += 0.15
        
        # 归一化
        total = sum(abs(v) for v in scores.values()) + 0.01
        for k in scores:
            scores[k] = (scores[k] + 0.1) / total
        
        # 选择最高分
        best_action = max(scores, key=scores.get)
        confidence = scores[best_action]
        
        # 计算仓位大小
        position_size = min(0.1, confidence * 0.2)  # 最大10%仓位
        
        return TradingAction(
            action_type=best_action,
            position_size=position_size,
            confidence=confidence,
            reasoning=" | ".join(reasons) if reasons else "综合分析"
        )
    
    def calculate_reward(
        self,
        action: TradingAction,
        prev_state: MarketState,
        next_state: MarketState,
        pnl: float,
        risk_level: float = 0.0
    ) -> Reward:
        """
        计算奖励
        
        考虑因素：
        - 盈亏
        - 风险惩罚
        - 滑点成本
        - 持仓时间成本
        """
        # 基础盈亏奖励
        pnl_reward = pnl
        
        # 风险惩罚
        risk_penalty = -abs(risk_level) * 0.01
        
        # 滑点成本估计
        slippage = abs(action.position_size) * 0.001 * prev_state.volatility
        slippage_cost = -slippage
        
        # 持仓成本（资金费率）
        holding_cost = 0.0
        if action.action_type in ["long", "short"]:
            holding_cost = -abs(prev_state.funding_rate) * 100
        
        total = pnl_reward + risk_penalty + slippage_cost + holding_cost
        
        return Reward(
            pnl=pnl_reward,
            risk_penalty=risk_penalty,
            slippage_cost=slippage_cost,
            holding_cost=holding_cost,
            total=total
        )
    
    def store_transition(self, transition: Dict) -> None:
        """存储转移数据"""
        self.memory.append(transition)
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]
    
    def update(self, batch_size: int = 64) -> Optional[float]:
        """
        PPO 更新
        返回 loss 值
        """
        if not self.use_torch or len(self.memory) < batch_size:
            return None
        
        # 采样批次
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor(np.array([t["state"] for t in batch])).to(self.device)
        actions = torch.LongTensor([t["action"] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t["reward"] for t in batch]).to(self.device)
        old_probs = torch.FloatTensor([t["old_prob"] for t in batch]).to(self.device)
        
        # 前向传播
        action_probs, values, _ = self.network(states)
        dist = torch.distributions.Categorical(action_probs)
        new_probs = dist.log_prob(actions)
        
        # PPO 目标
        ratio = torch.exp(new_probs - old_probs)
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * rewards
        
        # 策略损失
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        
        # 总损失
        loss = policy_loss + 0.5 * value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_probability_matrix(
        self,
        state: MarketState
    ) -> Dict[str, float]:
        """
        获取概率矩阵
        
        Returns:
            {"long": 0.3, "short": 0.5, "hold": 0.2, "close": 0.0}
        """
        action = self.select_action(state, deterministic=False)
        
        if self.use_torch:
            state_vec = self.normalize_state(state.to_vector())
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
                action_probs, _, _ = self.network(state_tensor)
                probs = action_probs[0].cpu().numpy()
                
                return {
                    "hold": float(probs[0]),
                    "long": float(probs[1]),
                    "short": float(probs[2]),
                    "close": float(probs[3]),
                }
        else:
            # 从规则决策推断概率
            scores = self._select_action_rule(state)
            return {
                "hold": 0.2,
                "long": 0.35 if action.action_type == "long" else 0.25,
                "short": 0.35 if action.action_type == "short" else 0.25,
                "close": 0.2,
            }


class AdversarialAgent:
    """
    对抗智能体 - 用于自我博弈训练
    模拟市场对手方的策略
    """
    
    def __init__(self, base_agent: PPOAgent):
        self.base_agent = base_agent
        self.adversarial_bias = 0.0  # 对抗性偏好
    
    def select_counter_action(
        self,
        state: MarketState,
        main_action: TradingAction
    ) -> TradingAction:
        """
        选择对抗动作
        试图预测并对抗主智能体的决策
        """
        # 对抗策略：如果主智能体做多，尝试寻找做空机会
        counter_map = {
            "long": "short",
            "short": "long",
            "hold": "hold",
            "close": "close"
        }
        
        counter_type = counter_map.get(main_action.action_type, "hold")
        
        # 使用基础智能体但有偏向
        action = self.base_agent.select_action(state)
        
        # 添加对抗性噪声
        action.action_type = counter_type
        action.confidence *= 0.8  # 对抗时降低置信度
        action.reasoning = f"对抗策略: 反向{main_action.action_type}"
        
        return action
    
    def simulate_market_reaction(
        self,
        state: MarketState,
        action: TradingAction
    ) -> MarketState:
        """
        模拟市场对交易的反应
        用于训练时预测价格变动
        """
        # 简单模型：大额交易会导致短期价格滑点
        price_impact = action.position_size * 0.01 * np.sign(1 if action.action_type == "long" else -1)
        
        new_state = MarketState(
            price=state.price * (1 + price_impact),
            volume=state.volume * (1 + abs(action.position_size) * 0.1),
            orderbook_imbalance=state.orderbook_imbalance - price_impact * 10,
            hurst=state.hurst,
            fractal_dim=state.fractal_dim,
            funding_rate=state.funding_rate,
            sentiment_score=state.sentiment_score,
            volatility=state.volatility * (1 + abs(action.position_size) * 0.5),
            momentum=state.momentum
        )
        
        return new_state


class SelfPlayTrainer:
    """
    自我博弈训练器
    主智能体 vs 对抗智能体
    """
    
    def __init__(self, main_agent: PPOAgent):
        self.main_agent = main_agent
        self.adversarial_agent = AdversarialAgent(main_agent)
        
        self.training_history: List[Dict] = []
        self.episode_rewards: List[float] = []
    
    def run_episode(
        self,
        initial_state: MarketState,
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """
        运行一个训练回合
        """
        state = initial_state
        total_reward = 0.0
        actions = []
        
        for step in range(max_steps):
            # 主智能体动作
            main_action = self.main_agent.select_action(state)
            
            # 对抗智能体动作
            adv_action = self.adversarial_agent.select_counter_action(state, main_action)
            
            # 模拟市场反应
            next_state = self.adversarial_agent.simulate_market_reaction(state, main_action)
            
            # 计算奖励（简化）
            pnl = (next_state.price - state.price) / state.price
            if main_action.action_type == "short":
                pnl = -pnl
            elif main_action.action_type == "hold":
                pnl = 0
            
            reward = self.main_agent.calculate_reward(
                main_action, state, next_state, pnl
            )
            
            # 存储转移
            if self.main_agent.use_torch:
                state_vec = self.main_agent.normalize_state(state.to_vector())
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.main_agent.device)
                    probs, _, _ = self.main_agent.network(state_tensor)
                    old_prob = probs[0, list(self.main_agent.ACTION_MAP.keys())[
                        list(self.main_agent.ACTION_MAP.values()).index(main_action.action_type)
                    ]].item()
                
                self.main_agent.store_transition({
                    "state": state_vec,
                    "action": list(self.main_agent.ACTION_MAP.keys())[
                        list(self.main_agent.ACTION_MAP.values()).index(main_action.action_type)
                    ],
                    "reward": reward.total,
                    "old_prob": old_prob,
                })
            
            total_reward += reward.total
            actions.append({
                "step": step,
                "main_action": main_action.action_type,
                "adv_action": adv_action.action_type,
                "reward": reward.total,
            })
            
            state = next_state
        
        self.episode_rewards.append(total_reward)
        
        return {
            "total_reward": total_reward,
            "avg_reward": total_reward / max_steps,
            "actions": actions,
        }
    
    def train(
        self,
        initial_state: MarketState,
        num_episodes: int = 100,
        update_interval: int = 10
    ) -> Dict[str, Any]:
        """
        执行自我博弈训练
        """
        results = []
        
        for episode in range(num_episodes):
            episode_result = self.run_episode(initial_state)
            results.append(episode_result)
            
            # 定期更新
            if episode % update_interval == 0:
                loss = self.main_agent.update()
                if loss is not None:
                    logger.info(f"Episode {episode}, Loss: {loss:.4f}")
        
        return {
            "episodes": num_episodes,
            "avg_reward": np.mean([r["total_reward"] for r in results]),
            "final_loss": loss if 'loss' in dir() else None,
            "reward_trend": [r["total_reward"] for r in results[-10:]],
        }


# 便捷函数
def rl_decision(state: MarketState) -> Dict[str, Any]:
    """
    使用 RL 智能体做决策
    
    Args:
        state: 市场状态
    
    Returns:
        决策结果和概率矩阵
    """
    agent = PPOAgent(use_torch=TORCH_AVAILABLE)
    action = agent.select_action(state)
    probs = agent.get_probability_matrix(state)
    
    return {
        "action": action.action_type,
        "position_size": action.position_size,
        "confidence": action.confidence,
        "reasoning": action.reasoning,
        "probability_matrix": probs,
    }
