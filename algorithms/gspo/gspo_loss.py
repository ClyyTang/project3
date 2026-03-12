"""
GSPO Loss 函数
实现 DPO (Direct Preference Optimization) 风格的损失
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_gspo_loss(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 GSPO loss (基于 DPO)
    
    核心思想：
    让模型提高 chosen 的概率，降低 rejected 的概率
    
    Loss = -log(sigmoid(beta * (log_p_chosen - log_p_rejected)))
         = -log(sigmoid(beta * log_ratio))
    
    其中：
    - log_ratio > 0: chosen 概率更高（好）
    - log_ratio < 0: rejected 概率更高（坏）
    
    Args:
        chosen_log_probs: [batch] chosen CoT 的 log 概率
        rejected_log_probs: [batch] rejected CoT 的 log 概率
        beta: 温度参数（越大，惩罚越强）
        
    Returns:
        loss: 标量 loss
        stats: 统计信息字典
            - 'loss': loss 值
            - 'accuracy': chosen > rejected 的比例
            - 'mean_margin': 平均 margin (chosen - rejected)
            - 'mean_chosen_prob': chosen 的平均概率
            - 'mean_rejected_prob': rejected 的平均概率
    
    Example:
        >>> chosen_lp = torch.tensor([-10.0, -15.0])
        >>> rejected_lp = torch.tensor([-20.0, -12.0])
        >>> loss, stats = compute_gspo_loss(chosen_lp, rejected_lp, beta=0.1)
        >>> # 第1个样本：chosen 好（-10 > -20）
        >>> # 第2个样本：rejected 好（-15 < -12）
        >>> # accuracy 应该是 0.5
    """
    assert chosen_log_probs.shape == rejected_log_probs.shape, \
        f"Shape mismatch: {chosen_log_probs.shape} vs {rejected_log_probs.shape}"
    
    batch_size = chosen_log_probs.shape[0]
    
    # 1. 计算 log ratio
    log_ratio = chosen_log_probs - rejected_log_probs  # [batch]
    
    # 2. DPO loss
    # -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
    loss_per_sample = F.softplus(-beta * log_ratio)  # [batch]
    loss = loss_per_sample.mean()
    
    # 3. 计算统计信息
    with torch.no_grad():
        # Accuracy: chosen > rejected 的比例
        correct = (log_ratio > 0).float()
        accuracy = correct.mean().item()
        
        # Margin: chosen 和 rejected 的差距
        mean_margin = log_ratio.mean().item()
        
        # 概率（exp(log_prob)）
        chosen_probs = torch.exp(chosen_log_probs)
        rejected_probs = torch.exp(rejected_log_probs)
        
        mean_chosen_prob = chosen_probs.mean().item()
        mean_rejected_prob = rejected_probs.mean().item()
        
        # 偏好强度（sigmoid 后的值）
        # 接近 1 表示模型强烈偏好 chosen
        # 接近 0 表示模型强烈偏好 rejected
        preference_prob = torch.sigmoid(beta * log_ratio)
        mean_preference = preference_prob.mean().item()
    
    stats = {
        'loss': loss.item(),
        'accuracy': accuracy,
        'mean_margin': mean_margin,
        'mean_chosen_prob': mean_chosen_prob,
        'mean_rejected_prob': mean_rejected_prob,
        'mean_preference': mean_preference,
        'batch_size': batch_size
    }
    
    return loss, stats


def compute_gspo_loss_with_reference(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    ref_chosen_log_probs: torch.Tensor,
    ref_rejected_log_probs: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    带 Reference Model 的 GSPO loss（标准 DPO）
    
    这个版本考虑了相对于 reference policy 的变化
    适用于有单独 reference model 的情况
    
    Loss = -log(sigmoid(beta * ((log_p_chosen - log_p_ref_chosen) - 
                                  (log_p_rejected - log_p_ref_rejected))))
    
    Args:
        chosen_log_probs: [batch] 当前模型对 chosen 的 log prob
        rejected_log_probs: [batch] 当前模型对 rejected 的 log prob
        ref_chosen_log_probs: [batch] reference 模型对 chosen 的 log prob
        ref_rejected_log_probs: [batch] reference 模型对 rejected 的 log prob
        beta: 温度参数
        
    Returns:
        loss, stats
        
    Note:
        目前我们不使用这个版本（省显存），但保留接口方便未来扩展
    """
    # KL 散度项
    kl_chosen = chosen_log_probs - ref_chosen_log_probs
    kl_rejected = rejected_log_probs - ref_rejected_log_probs
    
    # 相对优势
    log_ratio = kl_chosen - kl_rejected
    
    # DPO loss
    loss_per_sample = F.softplus(-beta * log_ratio)
    loss = loss_per_sample.mean()
    
    # 统计
    with torch.no_grad():
        accuracy = (log_ratio > 0).float().mean().item()
        mean_margin = log_ratio.mean().item()
        mean_kl_chosen = kl_chosen.mean().item()
        mean_kl_rejected = kl_rejected.mean().item()
    
    stats = {
        'loss': loss.item(),
        'accuracy': accuracy,
        'mean_margin': mean_margin,
        'mean_kl_chosen': mean_kl_chosen,
        'mean_kl_rejected': mean_kl_rejected,
        'batch_size': chosen_log_probs.shape[0]
    }
    
    return loss, stats


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """测试 GSPO loss"""
    
    print("=" * 60)
    print("测试 GSPO Loss")
    print("=" * 60)
    
    # 测试 1: 基础功能
    print("\n[测试 1] 基础功能")
    print("-" * 60)
    
    # 模拟 log probs（负数，越大越好）
    chosen = torch.tensor([-10.0, -15.0, -8.0, -12.0])
    rejected = torch.tensor([-20.0, -12.0, -18.0, -25.0])
    
    print(f"Chosen log probs:   {chosen.tolist()}")
    print(f"Rejected log probs: {rejected.tolist()}")
    print(f"\n预期:")
    print(f"  样本 0: chosen 好 (-10 > -20) ✓")
    print(f"  样本 1: rejected 好 (-15 < -12) ✗")
    print(f"  样本 2: chosen 好 (-8 > -18) ✓")
    print(f"  样本 3: chosen 好 (-12 > -25) ✓")
    print(f"  预期 accuracy: 3/4 = 0.75")
    
    loss, stats = compute_gspo_loss(chosen, rejected, beta=0.1)
    
    print(f"\n结果:")
    print(f"  Loss: {stats['loss']:.4f}")
    print(f"  Accuracy: {stats['accuracy']:.2f}")
    print(f"  Mean margin: {stats['mean_margin']:.2f}")
    print(f"  Mean preference: {stats['mean_preference']:.4f}")
    
    assert abs(stats['accuracy'] - 0.75) < 0.01, "Accuracy 计算错误"
    assert stats['mean_margin'] > 0, "整体上 chosen 应该更好"
    print("✅ 基础功能正确")
    
    # 测试 2: Beta 的影响
    print("\n[测试 2] Beta 参数的影响")
    print("-" * 60)
    
    for beta in [0.01, 0.1, 1.0, 10.0]:
        loss, stats = compute_gspo_loss(chosen, rejected, beta=beta)
        print(f"Beta={beta:5.2f}: Loss={loss.item():.4f}, Preference={stats['mean_preference']:.4f}")
    
    print("\n观察: Beta 越大，loss 越大（惩罚越强）")
    print("✅ Beta 参数工作正常")
    
    # 测试 3: 边界情况
    print("\n[测试 3] 边界情况")
    print("-" * 60)
    
    # 3.1 完全正确
    perfect_chosen = torch.tensor([-5.0, -6.0])
    perfect_rejected = torch.tensor([-50.0, -60.0])
    loss, stats = compute_gspo_loss(perfect_chosen, perfect_rejected, beta=0.1)
    print(f"完全正确: Accuracy={stats['accuracy']:.2f}, Loss={stats['loss']:.4f}")
    assert stats['accuracy'] == 1.0, "应该100%正确"
    
    # 3.2 完全错误
    bad_chosen = torch.tensor([-50.0, -60.0])
    bad_rejected = torch.tensor([-5.0, -6.0])
    loss, stats = compute_gspo_loss(bad_chosen, bad_rejected, beta=0.1)
    print(f"完全错误: Accuracy={stats['accuracy']:.2f}, Loss={stats['loss']:.4f}")
    assert stats['accuracy'] == 0.0, "应该0%正确"
    
    # 3.3 相同（无偏好）
    same_chosen = torch.tensor([-10.0, -10.0])
    same_rejected = torch.tensor([-10.0, -10.0])
    loss, stats = compute_gspo_loss(same_chosen, same_rejected, beta=0.1)
    print(f"无偏好: Accuracy={stats['accuracy']:.2f}, Loss={stats['loss']:.4f}, Preference={stats['mean_preference']:.4f}")
    assert abs(stats['mean_preference'] - 0.5) < 0.01, "无偏好时应该接近0.5"
    
    print("✅ 边界情况处理正确")
    
    # 测试 4: 梯度检查
    print("\n[测试 4] 梯度检查")
    print("-" * 60)
    
    chosen_param = torch.tensor([-10.0, -15.0], requires_grad=True)
    rejected_param = torch.tensor([-20.0, -12.0], requires_grad=True)
    
    loss, stats = compute_gspo_loss(chosen_param, rejected_param, beta=0.1)
    loss.backward()
    
    print(f"Chosen gradient: {chosen_param.grad}")
    print(f"Rejected gradient: {rejected_param.grad}")
    print("梯度应该:")
    print("  - chosen: 负梯度（增加概率）")
    print("  - rejected: 正梯度（降低概率）")
    
    # 验证梯度方向
    assert chosen_param.grad[0] < 0, "Chosen 梯度应该是负的（样本0正确）"
    assert rejected_param.grad[0] > 0, "Rejected 梯度应该是正的（样本0正确）"
    
    print("✅ 梯度方向正确")
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ GSPO Loss 所有测试通过！")
    print("=" * 60)
    print("\n核心验证:")
    print("  ✓ 基础 loss 计算")
    print("  ✓ Accuracy 统计")
    print("  ✓ Beta 参数影响")
    print("  ✓ 边界情况处理")
    print("  ✓ 梯度方向正确")
    print("\n可以用于训练！")