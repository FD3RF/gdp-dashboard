# agent/brain.py
"""
第3层：深度强化学习大脑
======================

功能：PPO 算法核心，输出动作概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from collections import deque
import random


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class PPOBrain(nn.Module):
    """
    PPO (Proximal Policy Optimization) 大脑
    
    包含：
    - Actor 网络：策略网络，输出动作概率
    - Critic 网络：价值网络，评估状态好坏
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super(PPOBrain, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor 网络 (策略网络)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
        )
        
        # Critic 网络 (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            (动作概率, 状态价值)
        """
        logits = self.actor(state)
        action_probs = F.softmax(logits, dim=-1)
        state_value = self.critic(state)
        return action_probs, state_value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        获取动作
        
        Args:
            state: 状态张量
            deterministic: 是否确定性选择
            
        Returns:
            (动作, 动作对数概率, 状态价值)
        """
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        value = self.critic(state)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = F.log_softmax(logits, dim=-1)
        action_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        return action.item(), action_log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作
        
        Args:
            states: 状态批次
            actions: 动作批次
            
        Returns:
            (动作对数概率, 状态价值, 熵)
        """
        logits = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        values = self.critic(states)
        
        log_prob = F.log_softmax(logits, dim=-1)
        action_log_probs = log_prob.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy()
        
        return action_log_probs, values.squeeze(-1), entropy
    
    def decide(self, state: torch.Tensor) -> Tuple[int, np.ndarray, float]:
        """
        根据状态选择动作
        
        Args:
            state: 状态张量
            
        Returns:
            (动作ID, 概率分布, 信心指数)
        """
        with torch.no_grad():
            probs, value = self.forward(state)
            
            # 采样动作
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # 信心指数 = 最大概率值
            confidence = torch.max(probs).item()
            
            return action.item(), probs.cpu().numpy(), confidence


class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(
        self, 
        brain: PPOBrain, 
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5
    ):
        self.brain = brain
        self.optimizer = optim.Adam(brain.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.buffer = []
    
    def store_transition(self, transition: Dict):
        """存储转移"""
        self.buffer.append(transition)
    
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool], 
        next_value: float
    ) -> Tuple[List[float], List[float]]:
        """计算广义优势估计"""
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(self, batch_size: int = 64, epochs: int = 10) -> Dict[str, float]:
        """更新网络"""
        if len(self.buffer) < batch_size:
            return {}
        
        # 准备数据
        states = torch.stack([t['state'] for t in self.buffer])
        actions = torch.tensor([t['action'] for t in self.buffer])
        old_log_probs = torch.stack([t['log_prob'] for t in self.buffer])
        rewards = [t['reward'] for t in self.buffer]
        dones = [t['done'] for t in self.buffer]
        values = [t['value'].item() for t in self.buffer]
        
        # 计算优势
        with torch.no_grad():
            _, next_value = self.brain(self.buffer[-1]['next_state'])
        advantages, returns = self.compute_gae(rewards, values, dones, next_value.item())
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 更新
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(epochs):
            # 随机采样
            indices = np.random.permutation(len(self.buffer))
            
            for start in range(0, len(self.buffer), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 计算新的概率和值
                new_log_probs, new_values, entropy = self.brain.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # 策略损失 (PPO Clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # 总损失
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy.mean()
                )
                
                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.brain.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # 清空缓冲区
        self.buffer = []
        
        n_updates = epochs * (len(self.buffer) // batch_size + 1)
        
        return {
            'total_loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }


# 导出
__all__ = ['PPOBrain', 'PPOTrainer', 'ReplayBuffer']
