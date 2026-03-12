# agent/__init__.py
"""
Oracle AI Agent - 智能体模块
"""

# 延迟导入torch相关模块
def get_perception_encoder():
    """延迟导入感知编码器"""
    from .perception import PerceptionEncoder
    return PerceptionEncoder

def get_ppo_brain():
    """延迟导入PPO大脑"""
    from .brain import PPOBrain
    return PPOBrain

# 直接导入不需要torch的模块
from .adversarial import AdversarialJudge, AdversarialResult, TrapType

__all__ = [
    'get_perception_encoder',
    'get_ppo_brain',
    'AdversarialJudge',
    'AdversarialResult',
    'TrapType',
]
