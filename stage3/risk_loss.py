"""
Risk Loss 计算器 (RiskLoss)

基于 stage2/dynamic_loss.py 扩展，新增：
- risk_loss_1: 整体风险度预测损失（MSE，全部样本）
- risk_loss_2: 错误类型预测损失（CrossEntropy，仅弱样本）
- 风险加权的 total_loss（高风险样本权重更大）

Total Loss = risk_weight * GSPO_loss
           + aux_loss
           + alpha * (risk_loss_1 + gamma * risk_loss_2)

其中 risk_weight = 1.0 + mu * overall_risk.detach()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import json
import os


class RiskLoss:
    """
    Risk-Aware Loss 计算器

    相比 stage2 DynamicLoss 的变化：
    1. 新增 risk_loss_1（MSE）和 risk_loss_2（CrossEntropy）
    2. chosen_score 使用全局归一化（训练开始前统计一次）
    3. total_loss 用 risk_weight 加权（高风险样本学得更多）
    """

    def __init__(
        self,
        # ===== stage2 原有参数 =====
        lambda_keyword: float = 0.15,
        lambda_direction: float = 0.1,
        lambda_quality: float = 0.1,
        lambda_validity: float = 0.1,
        # ===== stage3 新增参数 =====
        alpha: float = 0.5,    # risk_loss 在 total_loss 中的权重
        gamma: float = 1.0,    # risk_loss_1 和 risk_loss_2 的平衡系数
        mu: float = 0.5,       # 风险加权系数
        # ===== 全局归一化参数（训练开始前传入）=====
        score_global_min: float = 0.0,
        score_global_max: float = 1.0,
        device: str = "cuda",
        verbose: bool = False
    ):
        self.lambda_keyword = lambda_keyword
        self.lambda_direction = lambda_direction
        self.lambda_quality = lambda_quality
        self.lambda_validity = lambda_validity

        self.alpha = alpha
        self.gamma = gamma
        self.mu = mu

        self.score_global_min = score_global_min
        self.score_global_max = score_global_max
        # 防止除以0
        self.score_range = max(score_global_max - score_global_min, 1e-6)

        self.device = device
        self.verbose = verbose

        # ===== stage2 原有 loss 函数 =====
        self.keyword_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.direction_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.quality_loss_fn = nn.MSELoss(reduction='mean')
        self.validity_loss_fn = nn.MSELoss(reduction='mean')

        # ===== stage3 新增 loss 函数 =====
        self.risk_mse_loss_fn = nn.MSELoss(reduction='mean')      # risk_head_1
        self.risk_ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')  # risk_head_2

        if self.verbose:
            print(f"\n{'='*60}")
            print("RiskLoss 初始化")
            print(f"{'='*60}")
            print(f"Stage2 原有权重:")
            print(f"  λ_keyword:   {lambda_keyword}")
            print(f"  λ_direction: {lambda_direction}")
            print(f"  λ_quality:   {lambda_quality}")
            print(f"  λ_validity:  {lambda_validity}")
            print(f"Stage3 新增参数:")
            print(f"  alpha (risk_loss权重):      {alpha}")
            print(f"  gamma (loss_1/loss_2平衡):  {gamma}")
            print(f"  mu    (风险加权系数):        {mu}")
            print(f"全局归一化:")
            print(f"  score_global_min: {score_global_min:.4f}")
            print(f"  score_global_max: {score_global_max:.4f}")
            print(f"  score_range:      {self.score_range:.4f}")
            print(f"{'='*60}\n")

    @staticmethod
    def compute_global_score_stats(aux_labels_path: str) -> Tuple[float, float]:
        """
        统计 auxiliary_labels.json 里所有 chosen_score 的全局 min 和 max
        在训练开始前调用一次，用于全局归一化

        Args:
            aux_labels_path: auxiliary_labels_round{N}.json 的路径

        Returns:
            (global_min, global_max)
        """
        with open(aux_labels_path, 'r') as f:
            data = json.load(f)

        scores = []
        # 兼容两种格式：list 或 dict with 'samples' key
        if isinstance(data, list):
            samples = data
        else:
            samples = data.get('samples', [])

        for sample in samples:
            score = sample.get('chosen_score')
            if score is not None and not (score != score):  # 排除 nan
                scores.append(float(score))

        if len(scores) == 0:
            print("⚠️  未找到 chosen_score，使用默认归一化范围 [0, 1]")
            return 0.0, 1.0

        global_min = min(scores)
        global_max = max(scores)

        print(f"✅ 全局 chosen_score 统计:")
        print(f"   样本数: {len(scores)}")
        print(f"   min: {global_min:.4f}")
        print(f"   max: {global_max:.4f}")
        print(f"   mean: {sum(scores)/len(scores):.4f}")

        return global_min, global_max

    def normalize_chosen_score(self, chosen_score: torch.Tensor) -> torch.Tensor:
        """
        对 chosen_score 做全局归一化
        normalized = (score - global_min) / (global_max - global_min)

        Args:
            chosen_score: [batch, 1] 原始分数

        Returns:
            normalized: [batch, 1] 归一化后的分数，范围 [0, 1]
        """
        normalized = (chosen_score - self.score_global_min) / self.score_range
        # clamp 到 [0, 1] 防止越界
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized

    def compute_gspo_loss_wrapper(
        self,
        chosen_log_probs: torch.Tensor,
        rejected_log_probs: torch.Tensor,
        beta: float,
        chosen_lengths: Optional[torch.Tensor] = None,
        rejected_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """GSPO loss 计算（和 stage2 完全一致，数值稳定版本）"""

        # 处理 nan/inf
        if torch.isnan(chosen_log_probs).any() or torch.isinf(chosen_log_probs).any():
            chosen_log_probs = torch.nan_to_num(
                chosen_log_probs, nan=-10.0, posinf=-1.0, neginf=-100.0)
        if torch.isnan(rejected_log_probs).any() or torch.isinf(rejected_log_probs).any():
            rejected_log_probs = torch.nan_to_num(
                rejected_log_probs, nan=-10.0, posinf=-1.0, neginf=-100.0)

        # 长度归一化
        if chosen_lengths is not None and rejected_lengths is not None:
            chosen_avg = chosen_log_probs / chosen_lengths.clamp(min=1)
            rejected_avg = rejected_log_probs / rejected_lengths.clamp(min=1)
            log_ratio = chosen_avg - rejected_avg
        else:
            log_ratio = (chosen_log_probs - rejected_log_probs) / 300.0

        # clamp 防溢出
        max_margin = 10.0 / beta
        log_ratio_clamped = torch.clamp(log_ratio, min=-max_margin, max=max_margin)

        loss = -F.logsigmoid(beta * log_ratio_clamped).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.1, device=chosen_log_probs.device, requires_grad=True)

        with torch.no_grad():
            accuracy = (log_ratio > 0).float().mean().item()
            mean_margin = log_ratio.mean().item()

        stats = {
            'loss': loss.item() if not torch.isnan(loss) else 0.0,
            'accuracy': accuracy,
            'mean_margin': mean_margin,
            'mean_margin_clamped': log_ratio_clamped.mean().item(),
            'mean_chosen_logprob': chosen_log_probs.mean().item(),
            'mean_rejected_logprob': rejected_log_probs.mean().item()
        }

        return loss, stats

    def compute_total_loss(
        self,
        chosen_log_probs: torch.Tensor,
        rejected_log_probs: torch.Tensor,
        chosen_aux_outputs: Optional[Dict[str, torch.Tensor]] = None,
        aux_labels: Optional[Dict[str, torch.Tensor]] = None,
        beta: float = 0.1,
        chosen_lengths: Optional[torch.Tensor] = None,
        rejected_lengths: Optional[torch.Tensor] = None,
        # ===== stage3 新增 =====
        normalized_chosen_score: Optional[torch.Tensor] = None,  # [batch, 1] 原始分数，这里做归一化
        root_cause_label: Optional[torch.Tensor] = None,         # [batch] long，0-3
        current_step: int = 0,                                    # 修2：warmup用
        warmup_steps: int = 200                                   # 修2：前200步alpha线性增长
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总 Loss

        Args:
            chosen_log_probs:       [batch] chosen序列log概率
            rejected_log_probs:     [batch] rejected序列log概率
            chosen_aux_outputs:     RiskAwareVLA的辅助输出（含overall_risk和error_type_probs）
            aux_labels:             辅助任务标签
            beta:                   GSPO温度参数
            chosen_lengths:         [batch] chosen序列有效token数
            rejected_lengths:       [batch] rejected序列有效token数
            normalized_chosen_score:[batch, 1] chosen_score原始值（函数内做全局归一化）
            root_cause_label:       [batch] 根因错误类型0-3，None表示该batch无弱样本

        Returns:
            total_loss, stats
        """

        # ===== 1. GSPO Loss =====
        gspo_loss, gspo_stats = self.compute_gspo_loss_wrapper(
            chosen_log_probs, rejected_log_probs, beta,
            chosen_lengths=chosen_lengths,
            rejected_lengths=rejected_lengths
        )

        # ===== 初始化 stats =====
        stats = {
            'gspo_loss': gspo_stats['loss'],
            'gspo_accuracy': gspo_stats['accuracy'],
            'gspo_margin': gspo_stats['mean_margin'],
            # stage2 原有辅助loss
            'keyword_loss': 0.0,
            'direction_loss': 0.0,
            'quality_loss': 0.0,
            'validity_loss': 0.0,
            'aux_total_loss': 0.0,
            # stage3 新增 risk loss
            'risk_loss_1': 0.0,
            'risk_loss_2': 0.0,
            'risk_loss_total': 0.0,
        }

        # ===== 2. Stage2 辅助任务 Loss =====
        aux_loss_total = torch.tensor(0.0, device=chosen_log_probs.device)

        if chosen_aux_outputs is not None and aux_labels is not None:

            if 'keywords' in chosen_aux_outputs and 'keywords' in aux_labels:
                try:
                    kl = self.keyword_loss_fn(
                        chosen_aux_outputs['keywords'], aux_labels['keywords'].float())
                    if not torch.isnan(kl):
                        aux_loss_total = aux_loss_total + self.lambda_keyword * kl
                        stats['keyword_loss'] = kl.item()
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Keyword loss失败: {e}")

            if 'direction' in chosen_aux_outputs and 'direction' in aux_labels:
                try:
                    dl = self.direction_loss_fn(
                        chosen_aux_outputs['direction'], aux_labels['direction'].long())
                    if not torch.isnan(dl):
                        aux_loss_total = aux_loss_total + self.lambda_direction * dl
                        stats['direction_loss'] = dl.item()
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Direction loss失败: {e}")

            if 'cot_quality' in chosen_aux_outputs and 'cot_quality' in aux_labels:
                try:
                    ql = self.quality_loss_fn(
                        chosen_aux_outputs['cot_quality'], aux_labels['cot_quality'].float())
                    if not torch.isnan(ql):
                        aux_loss_total = aux_loss_total + self.lambda_quality * ql
                        stats['quality_loss'] = ql.item()
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Quality loss失败: {e}")

            if 'action_validity' in chosen_aux_outputs and 'action_validity' in aux_labels:
                try:
                    vl = self.validity_loss_fn(
                        chosen_aux_outputs['action_validity'],
                        aux_labels['action_validity'].float())
                    if not torch.isnan(vl):
                        aux_loss_total = aux_loss_total + self.lambda_validity * vl
                        stats['validity_loss'] = vl.item()
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Validity loss失败: {e}")

        stats['aux_total_loss'] = aux_loss_total.item()

        # ===== 3. Stage3 Risk Loss =====
        risk_loss = torch.tensor(0.0, device=chosen_log_probs.device)

        if chosen_aux_outputs is not None:

            # --- risk_loss_1: 整体风险度（全部样本参与）---
            if ('overall_risk' in chosen_aux_outputs
                    and normalized_chosen_score is not None):
                try:
                    # 全局归一化
                    norm_score = self.normalize_chosen_score(
                        normalized_chosen_score.to(chosen_log_probs.device))
                    # 风险目标 = 1 - 归一化分数
                    risk_target = 1.0 - norm_score  # [batch, 1]

                    rl1 = self.risk_mse_loss_fn(
                        chosen_aux_outputs['overall_risk'].float(),  # [batch, 1] fp32
                        risk_target.float()
                    )
                    if not torch.isnan(rl1):
                        risk_loss = risk_loss + rl1
                        stats['risk_loss_1'] = rl1.item()
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Risk Loss 1 失败: {e}")

            # --- risk_loss_2: 错误类型预测（仅弱样本参与）---
            if ('error_type_probs' in chosen_aux_outputs
                    and root_cause_label is not None):
                try:
                    rl2 = self.risk_ce_loss_fn(
                        chosen_aux_outputs['error_type_probs'].float(),  # [batch, 4] fp32
                        root_cause_label.long().to(chosen_log_probs.device)  # [batch]
                    )
                    if not torch.isnan(rl2):
                        risk_loss = risk_loss + self.gamma * rl2
                        stats['risk_loss_2'] = rl2.item()
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Risk Loss 2 失败: {e}")

        stats['risk_loss_total'] = risk_loss.item()

        # ===== 4. 风险加权 GSPO Loss =====
        # risk_weight = 1.0 + mu * overall_risk.detach()
        # detach() 很重要：防止 GSPO 梯度反过来影响 risk_head_1 的预测
        risk_weight = torch.tensor(1.0, device=chosen_log_probs.device)
        if (chosen_aux_outputs is not None
                and 'overall_risk' in chosen_aux_outputs
                and not chosen_aux_outputs.get('_is_fallback', False)):
            try:
                mean_risk = chosen_aux_outputs['overall_risk'].detach().mean()
                risk_weight = 1.0 + self.mu * mean_risk
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  风险加权计算失败: {e}")

        # ===== 5. 汇总 Total Loss =====
        # 修2：warmup - 前 warmup_steps 步 alpha 从0线性增长到设定值
        if current_step < warmup_steps:
            effective_alpha = self.alpha * (current_step / max(warmup_steps, 1))
        else:
            effective_alpha = self.alpha
        stats['effective_alpha'] = effective_alpha
        # total_loss = risk_weight * gspo_loss + aux_loss + effective_alpha * risk_loss
        total_loss = risk_weight * gspo_loss + aux_loss_total + effective_alpha * risk_loss

        # 最终 nan 检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = gspo_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(
                    0.01, device=chosen_log_probs.device, requires_grad=True)

        stats['total_loss'] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
        stats['risk_weight'] = risk_weight.item() if hasattr(risk_weight, 'item') else float(risk_weight)

        return total_loss, stats


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("【测试 RiskLoss】\n")

    # 模拟全局归一化参数
    loss_calc = RiskLoss(
        alpha=0.5, gamma=1.0, mu=0.5,
        score_global_min=0.2,
        score_global_max=0.8,
        verbose=True
    )

    batch_size = 2
    device = 'cpu'

    chosen_log_probs = torch.tensor([-1.0, -1.5])
    rejected_log_probs = torch.tensor([-2.0, -2.5])

    chosen_aux_outputs = {
        'keywords': torch.randn(batch_size, 34),
        'direction': torch.randn(batch_size, 4),
        'cot_quality': torch.sigmoid(torch.randn(batch_size, 1)),
        'action_validity': torch.sigmoid(torch.randn(batch_size, 1)),
        'overall_risk': torch.sigmoid(torch.randn(batch_size, 1)),
        'error_type_probs': torch.randn(batch_size, 4)
    }

    aux_labels = {
        'keywords': torch.zeros(batch_size, 34),
        'direction': torch.tensor([0, 2]),
        'cot_quality': torch.tensor([[0.7], [0.4]]),
        'action_validity': torch.tensor([[0.8], [0.5]])
    }

    normalized_chosen_score = torch.tensor([[0.44], [0.55]])
    root_cause_label = torch.tensor([0, 2])  # perception, reasoning

    total_loss, stats = loss_calc.compute_total_loss(
        chosen_log_probs=chosen_log_probs,
        rejected_log_probs=rejected_log_probs,
        chosen_aux_outputs=chosen_aux_outputs,
        aux_labels=aux_labels,
        beta=0.1,
        normalized_chosen_score=normalized_chosen_score,
        root_cause_label=root_cause_label
    )

    print(f"\n结果:")
    print(f"  total_loss:      {stats['total_loss']:.4f}")
    print(f"  gspo_loss:       {stats['gspo_loss']:.4f}")
    print(f"  aux_total_loss:  {stats['aux_total_loss']:.4f}")
    print(f"  risk_loss_1:     {stats['risk_loss_1']:.4f}")
    print(f"  risk_loss_2:     {stats['risk_loss_2']:.4f}")
    print(f"  risk_loss_total: {stats['risk_loss_total']:.4f}")
    print(f"  risk_weight:     {stats['risk_weight']:.4f}")
    print(f"  gspo_accuracy:   {stats['gspo_accuracy']:.2%}")

    assert not torch.isnan(total_loss), "total_loss 不应该是 nan"
    print("\n✅ RiskLoss 测试通过")
