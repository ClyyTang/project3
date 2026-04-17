"""
GSPO (Group Relative Policy Optimization) 算法包

核心组件：
- GSPOConfig: 配置类
- GSPOTrainer: 训练器
- CandidateGenerator: 候选生成器
- compute_gspo_loss: 损失函数
"""

from .gspo_config import GSPOConfig
from .gspo_trainer import GSPOTrainer
from .candidate_generator import CandidateGenerator
from .gspo_loss import compute_gspo_loss, compute_gspo_loss_with_reference

__all__ = [
    'GSPOConfig',
    'GSPOTrainer',
    'CandidateGenerator',
    'compute_gspo_loss',
    'compute_gspo_loss_with_reference'
]

__version__ = '1.0.0'