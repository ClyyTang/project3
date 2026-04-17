"""
Dynamic Loss 计算器

功能：
- GSPO Loss（复用compute_gspo_loss）
- 辅助任务Losses（keywords, direction, cot_quality, action_validity）
- 动态加权组合

使用场景：Stage 2 Multi-task GSPO训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# 导入GSPO loss计算（复用现有代码）
import sys
import os
sys.path.insert(0, '/home/ubuntu/data1/lyy/full_rlds_project/algorithms')

try:
    from gspo.gspo_loss import compute_gspo_loss
    GSPO_LOSS_AVAILABLE = True
except ImportError:
    print("⚠️  警告: 无法导入compute_gspo_loss，将使用简化版本")
    GSPO_LOSS_AVAILABLE = False


class DynamicLoss:
    """
    动态Loss计算器
    
    Total Loss = GSPO Loss + λ1*Keyword + λ2*Direction 
                 + λ3*Quality + λ4*Validity
    """
    
    def __init__(
        self,
        lambda_keyword: float = 0.15,
        lambda_direction: float = 0.1,
        lambda_quality: float = 0.1,
        lambda_validity: float = 0.1,
        device: str = "cuda",
        verbose: bool = False
    ):
        """
        初始化DynamicLoss
        """
        self.lambda_keyword = lambda_keyword
        self.lambda_direction = lambda_direction
        self.lambda_quality = lambda_quality
        self.lambda_validity = lambda_validity
        self.device = device
        self.verbose = verbose
        
        # 初始化辅助任务loss函数
        self.keyword_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.direction_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.quality_loss_fn = nn.MSELoss(reduction='mean')
        self.validity_loss_fn = nn.MSELoss(reduction='mean')
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("DynamicLoss初始化")
            print(f"{'='*60}")
            print(f"权重配置:")
            print(f"  λ_keyword: {lambda_keyword}")
            print(f"  λ_direction: {lambda_direction}")
            print(f"  λ_quality: {lambda_quality}")
            print(f"  λ_validity: {lambda_validity}")
            print(f"{'='*60}\n")
    
    def compute_gspo_loss_wrapper(
        self,
        chosen_log_probs: torch.Tensor,
        rejected_log_probs: torch.Tensor,
        beta: float,
        chosen_lengths: Optional[torch.Tensor] = None,
        rejected_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GSPO loss计算的包装器（数值稳定版本 + 长度归一化）
        
        Args:
            chosen_log_probs: [batch] - 总log概率
            rejected_log_probs: [batch] - 总log概率
            beta: DPO温度参数
            chosen_lengths: [batch] - chosen序列有效token数（可选）
            rejected_lengths: [batch] - rejected序列有效token数（可选）
            
        Returns:
            - loss: GSPO loss
            - stats: 统计信息
        """
        # ⭐ 数值稳定性处理
        # 1. 检查并处理 nan/inf
        if torch.isnan(chosen_log_probs).any() or torch.isinf(chosen_log_probs).any():
            chosen_log_probs = torch.nan_to_num(chosen_log_probs, nan=-10.0, posinf=-1.0, neginf=-100.0)
        
        if torch.isnan(rejected_log_probs).any() or torch.isinf(rejected_log_probs).any():
            rejected_log_probs = torch.nan_to_num(rejected_log_probs, nan=-10.0, posinf=-1.0, neginf=-100.0)
        
        # 2. ⭐ 关键修复：长度归一化（转换为每token平均log_prob）
        if chosen_lengths is not None and rejected_lengths is not None:
            # 使用传入的长度
            chosen_avg = chosen_log_probs / chosen_lengths.clamp(min=1)
            rejected_avg = rejected_log_probs / rejected_lengths.clamp(min=1)
            log_ratio = chosen_avg - rejected_avg
            
            if self.verbose:
                print(f"  [归一化] chosen_avg: {chosen_avg.mean().item():.2f}, rejected_avg: {rejected_avg.mean().item():.2f}")
        else:
            # 没有长度信息，使用原始值但做缩放
            # 假设平均序列长度约300，将margin缩放到合理范围
            scale_factor = 300.0
            log_ratio = (chosen_log_probs - rejected_log_probs) / scale_factor
        
        # 3. ⭐ Clamp margin 防止数值溢出
        max_margin = 10.0 / beta  # 更保守的clamp
        log_ratio_clamped = torch.clamp(log_ratio, min=-max_margin, max=max_margin)
        
        # 4. 使用 logsigmoid
        loss = -F.logsigmoid(beta * log_ratio_clamped).mean()
        
        # 5. 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.1, device=chosen_log_probs.device, requires_grad=True)
            if self.verbose:
                print("⚠️  GSPO loss 出现 nan/inf，使用 0.1 代替")
        
        # 6. 统计信息
        with torch.no_grad():
            accuracy = (log_ratio > 0).float().mean().item()
            mean_margin = log_ratio.mean().item()
            mean_margin_clamped = log_ratio_clamped.mean().item()
            mean_chosen = chosen_log_probs.mean().item()
            mean_rejected = rejected_log_probs.mean().item()
        
        stats = {
            'loss': loss.item() if not torch.isnan(loss) else 0.0,
            'accuracy': accuracy,
            'mean_margin': mean_margin,
            'mean_margin_clamped': mean_margin_clamped,
            'mean_chosen_logprob': mean_chosen,
            'mean_rejected_logprob': mean_rejected
        }
        
        return loss, stats
    
    def compute_keyword_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算关键词识别loss"""
        return self.keyword_loss_fn(pred, target.float())
    
    def compute_direction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算方向分类loss"""
        return self.direction_loss_fn(pred, target.long())
    
    def compute_quality_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算CoT质量loss"""
        return self.quality_loss_fn(pred, target.float())
    
    def compute_validity_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """计算动作有效性loss"""
        return self.validity_loss_fn(pred, target.float())
    
    def compute_total_loss(
        self,
        chosen_log_probs: torch.Tensor,
        rejected_log_probs: torch.Tensor,
        chosen_aux_outputs: Optional[Dict[str, torch.Tensor]] = None,
        aux_labels: Optional[Dict[str, torch.Tensor]] = None,
        beta: float = 0.1,
        chosen_lengths: Optional[torch.Tensor] = None,
        rejected_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总Loss
        
        Args:
            chosen_log_probs: [batch] - chosen序列的log概率
            rejected_log_probs: [batch] - rejected序列的log概率
            chosen_aux_outputs: MultiTaskVLA在chosen上的辅助输出
            aux_labels: 辅助任务的真实标签
            beta: GSPO的温度参数
            chosen_lengths: [batch] - chosen序列有效token数
            rejected_lengths: [batch] - rejected序列有效token数
            
        Returns:
            - total_loss: 总loss
            - stats: 详细统计信息
        """
        # 1. 计算GSPO loss（主任务）- 传入长度信息
        gspo_loss, gspo_stats = self.compute_gspo_loss_wrapper(
            chosen_log_probs,
            rejected_log_probs,
            beta,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths
        )
        
        # 初始化统计字典
        stats = {
            'gspo_loss': gspo_stats['loss'],
            'gspo_accuracy': gspo_stats['accuracy'],
            'gspo_margin': gspo_stats['mean_margin'],
            'keyword_loss': 0.0,
            'direction_loss': 0.0,
            'quality_loss': 0.0,
            'validity_loss': 0.0,
            'aux_total_loss': 0.0
        }
        
        # 2. 如果没有辅助任务，直接返回GSPO loss
        if chosen_aux_outputs is None or aux_labels is None:
            stats['total_loss'] = gspo_stats['loss']
            return gspo_loss, stats
        
        # 3. 计算辅助任务losses
        aux_loss_total = 0.0
        
        # 3.1 Keyword loss
        if 'keywords' in chosen_aux_outputs and 'keywords' in aux_labels:
            try:
                keyword_loss = self.compute_keyword_loss(
                    chosen_aux_outputs['keywords'],
                    aux_labels['keywords']
                )
                if not torch.isnan(keyword_loss):
                    aux_loss_total += self.lambda_keyword * keyword_loss
                    stats['keyword_loss'] = keyword_loss.item()
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Keyword loss计算失败: {e}")
        
        # 3.2 Direction loss
        if 'direction' in chosen_aux_outputs and 'direction' in aux_labels:
            try:
                direction_loss = self.compute_direction_loss(
                    chosen_aux_outputs['direction'],
                    aux_labels['direction']
                )
                if not torch.isnan(direction_loss):
                    aux_loss_total += self.lambda_direction * direction_loss
                    stats['direction_loss'] = direction_loss.item()
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Direction loss计算失败: {e}")
        
        # 3.3 CoT Quality loss
        if 'cot_quality' in chosen_aux_outputs and 'cot_quality' in aux_labels:
            try:
                quality_loss = self.compute_quality_loss(
                    chosen_aux_outputs['cot_quality'],
                    aux_labels['cot_quality']
                )
                if not torch.isnan(quality_loss):
                    aux_loss_total += self.lambda_quality * quality_loss
                    stats['quality_loss'] = quality_loss.item()
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Quality loss计算失败: {e}")
        
        # 3.4 Action Validity loss
        if 'action_validity' in chosen_aux_outputs and 'action_validity' in aux_labels:
            try:
                validity_loss = self.compute_validity_loss(
                    chosen_aux_outputs['action_validity'],
                    aux_labels['action_validity']
                )
                if not torch.isnan(validity_loss):
                    aux_loss_total += self.lambda_validity * validity_loss
                    stats['validity_loss'] = validity_loss.item()
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Validity loss计算失败: {e}")
        
        # 4. 总loss
        total_loss = gspo_loss + aux_loss_total
        
        # ⭐ 最终 nan 检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = gspo_loss  # 如果辅助loss有问题，只用gspo_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(0.01, device=chosen_log_probs.device, requires_grad=True)
        
        stats['aux_total_loss'] = aux_loss_total.item() if isinstance(aux_loss_total, torch.Tensor) else aux_loss_total
        stats['total_loss'] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
        
        return total_loss, stats


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("【测试DynamicLoss - 数值稳定性】\n")
    
    loss_calculator = DynamicLoss(verbose=True)
    
    # 测试1: 正常情况
    print("\n[测试1] 正常情况")
    print("="*60)
    
    batch_size = 4
    chosen_log_probs = torch.randn(batch_size) * 0.5 - 1.0
    rejected_log_probs = torch.randn(batch_size) * 0.5 - 1.5
    
    total_loss, stats = loss_calculator.compute_total_loss(
        chosen_log_probs, rejected_log_probs, beta=0.1
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"GSPO margin: {stats['gspo_margin']:.4f}")
    print("✅ 测试1通过\n")
    
    # 测试2: 极端margin（之前会导致nan）
    print("[测试2] 极端margin（margin=2000）")
    print("="*60)
    
    chosen_log_probs = torch.tensor([1000.0, 1000.0])
    rejected_log_probs = torch.tensor([-1000.0, -1000.0])
    
    total_loss, stats = loss_calculator.compute_total_loss(
        chosen_log_probs, rejected_log_probs, beta=0.1
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"GSPO margin (原始): {stats['gspo_margin']:.4f}")
    print(f"是否nan: {torch.isnan(total_loss).item()}")
    assert not torch.isnan(total_loss), "Loss不应该是nan"
    print("✅ 测试2通过 - 极端margin处理正常\n")
    
    # 测试3: 输入包含nan
    print("[测试3] 输入包含nan")
    print("="*60)
    
    chosen_log_probs = torch.tensor([float('nan'), -1.0])
    rejected_log_probs = torch.tensor([-2.0, -3.0])
    
    total_loss, stats = loss_calculator.compute_total_loss(
        chosen_log_probs, rejected_log_probs, beta=0.1
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"是否nan: {torch.isnan(total_loss).item()}")
    assert not torch.isnan(total_loss), "Loss不应该是nan"
    print("✅ 测试3通过 - nan输入处理正常\n")
    
    print("="*60)
    print("✅ 所有数值稳定性测试通过！")
    print("="*60)