"""
序列 Log Probability 计算工具

用于 PPO 训练中计算生成序列的概率
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

IGNORE_INDEX = -100

def compute_sequence_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    return_per_token: bool = False
) -> torch.Tensor:
    """
    计算序列的 log probability
    
    对于生成的序列 [t1, t2, ..., tn]，计算：
    log P(sequence) = sum_i log P(ti | t<i, context)
    
    Args:
        logits: [batch, seq_len, vocab_size] 模型输出
        labels: [batch, seq_len] 目标 tokens
                IGNORE_INDEX 的位置会被忽略（通常是 prompt 部分）
        return_per_token: 是否返回每个 token 的 log prob
        
    Returns:
        如果 return_per_token=False:
            sequence_log_probs: [batch] 每个序列的总 log prob
        如果 return_per_token=True:
            (sequence_log_probs, token_log_probs): [batch], [batch, seq_len-1]
    
    示例:
        输入序列: [INST] What to do? [/INST] <thinking>Move forward</thinking><action>9</action>
        Labels:   [-100, -100, ..., -100, <thinking>, Move, forward, </thinking>, <action>, 9, </action>]
                  ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     prompt (忽略)                        生成部分（计算）
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # ========== 1. Shift (标准语言模型) ==========
    # logits[:, :-1] 用于预测 labels[:, 1:]
    # 即：用前 n-1 个 token 预测第 n 个 token
    
    shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq_len-1, vocab]
    shift_labels = labels[:, 1:].contiguous()       # [batch, seq_len-1]
    
    # ========== 2. 计算 log probabilities ==========
    
    # Softmax -> log (数值稳定)
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch, seq_len-1, vocab]
    
    # ========== 3. Masking (在 gather 之前) ==========
    # 忽略 IGNORE_INDEX（prompt 部分）
    
    mask = (shift_labels != IGNORE_INDEX).float()  # [batch, seq_len-1]
    
    # 替换 IGNORE_INDEX 为 0（避免 gather 越界）
    # 这些位置会被 mask 清零，所以替换成什么都无所谓
    valid_labels = shift_labels.clone()
    valid_labels[shift_labels == IGNORE_INDEX] = 0
    
    # 收集对应 label 的 log prob
    # gather: 从 vocab 维度选择 label 对应的概率
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=valid_labels.unsqueeze(2)  # [batch, seq_len-1, 1]
    ).squeeze(2)  # [batch, seq_len-1]
    
    # 应用 mask（清零 IGNORE_INDEX 位置）
    masked_token_log_probs = token_log_probs * mask
    
    # ========== 4. Sum over sequence ==========
    
    sequence_log_probs = masked_token_log_probs.sum(dim=1)  # [batch]
    
    if return_per_token:
        return sequence_log_probs, masked_token_log_probs
    else:
        return sequence_log_probs


def compute_sequence_logprob_from_model(
    model,
    pixel_values: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_per_token: bool = False
) -> torch.Tensor:
    """
    从模型直接计算序列 log prob（便捷函数）
    
    Args:
        model: VLA 模型
        pixel_values: 图像输入
        input_ids: token ids
        labels: 标签（包含 IGNORE_INDEX）
        return_per_token: 是否返回每个 token 的 log prob
        
    Returns:
        sequence_log_probs 或 (sequence_log_probs, token_log_probs)
    """
    # Forward
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids
    )
    
    logits = outputs.logits
    
    return compute_sequence_logprob(logits, labels, return_per_token)


def compute_kl_divergence(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    计算策略和参考模型之间的 KL 散度
    
    KL(π_policy || π_ref) = sum_i π_policy(i) * log(π_policy(i) / π_ref(i))
    
    Args:
        policy_logits: [batch, seq_len, vocab] 当前策略的 logits
        ref_logits: [batch, seq_len, vocab] 参考模型的 logits
        labels: [batch, seq_len] 标签（用于 masking）
        
    Returns:
        kl_div: [batch] 每个序列的 KL 散度
    """
    # Shift
    shift_policy_logits = policy_logits[:, :-1, :].contiguous()
    shift_ref_logits = ref_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Log probabilities
    policy_log_probs = F.log_softmax(shift_policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)
    
    # Probabilities (policy)
    policy_probs = F.softmax(shift_policy_logits, dim=-1)
    
    # KL = sum(p * (log p - log q))
    kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)  # [batch, seq_len-1]
    
    # Mask
    mask = (shift_labels != IGNORE_INDEX).float()
    masked_kl = kl_per_token * mask
    
    # Average over valid tokens
    kl_div = masked_kl.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [batch]
    
    return kl_div


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """测试 Sequence LogProb 计算"""
    
    print("="*60)
    print("Sequence LogProb 工具测试")
    print("="*60)
    
    # ========== 测试 1: 基础 log prob 计算 ==========
    
    print("\n[测试 1] 基础 log prob 计算")
    print("-"*60)
    
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    
    # 模拟数据
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Labels: 前 5 个是 prompt (IGNORE_INDEX)，后 5 个是生成部分
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, :5] = IGNORE_INDEX  # Mask prompt
    
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels (sample 0): {labels[0].tolist()}")
    
    # 计算 log prob
    seq_log_probs, token_log_probs = compute_sequence_logprob(
        logits, labels, return_per_token=True
    )
    
    print(f"\n序列 log prob: {seq_log_probs}")
    print(f"Token log probs shape: {token_log_probs.shape}")
    print(f"Token log probs (sample 0, last 6): {token_log_probs[0, -6:].tolist()}")
    
    # 验证：前 5 个应该是 0 (masked)
    assert torch.allclose(token_log_probs[:, :4], torch.zeros_like(token_log_probs[:, :4])), \
        "前 4 个 token 应该被 mask"
    
    # 验证：后面的不是 0
    assert not torch.allclose(token_log_probs[:, 5:], torch.zeros_like(token_log_probs[:, 5:])), \
        "生成部分的 log prob 不应该是 0"
    
    print("✅ 基础测试通过")
    
    # ========== 测试 2: Log prob 的数值范围 ==========
    
    print("\n[测试 2] Log prob 数值范围")
    print("-"*60)
    
    # Log prob 应该是负数（概率 <= 1，log <= 0）
    print(f"Log prob 范围: [{seq_log_probs.min():.2f}, {seq_log_probs.max():.2f}]")
    assert seq_log_probs.max() <= 0, "Log prob 应该 <= 0"
    
    # 对于随机初始化的模型，log prob 应该接近 -log(vocab_size)
    expected_per_token = -torch.log(torch.tensor(vocab_size, dtype=torch.float))
    num_valid_tokens = (labels != IGNORE_INDEX).sum(dim=1).float()
    expected_seq = expected_per_token * num_valid_tokens
    
    print(f"期望 log prob: {expected_seq}")
    print(f"实际 log prob: {seq_log_probs}")
    print(f"差异: {(seq_log_probs - expected_seq).abs()}")
    
    # 应该在合理范围内
    assert (seq_log_probs - expected_seq).abs().max() < 10, \
        "随机模型的 log prob 应该接近理论值"
    
    print("✅ 数值范围测试通过")
    
    # ========== 测试 3: KL 散度计算 ==========
    
    print("\n[测试 3] KL 散度计算")
    print("-"*60)
    
    # 两个不同的 logits
    policy_logits = torch.randn(batch_size, seq_len, vocab_size)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    kl_div = compute_kl_divergence(policy_logits, ref_logits, labels)
    
    print(f"KL 散度: {kl_div}")
    
    # KL 散度应该 >= 0
    assert (kl_div >= 0).all(), "KL 散度应该非负"
    
    # 如果两个 logits 相同，KL 应该接近 0
    kl_same = compute_kl_divergence(policy_logits, policy_logits, labels)
    print(f"相同 logits 的 KL: {kl_same}")
    assert kl_same.abs().max() < 1e-5, "相同分布的 KL 应该接近 0"
    
    print("✅ KL 散度测试通过")
    
    # ========== 测试 4: 梯度传播 ==========
    
    print("\n[测试 4] 梯度传播")
    print("-"*60)
    
    # 创建需要梯度的 logits
    logits_grad = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    
    seq_log_probs_grad = compute_sequence_logprob(logits_grad, labels)
    
    # 反向传播
    loss = -seq_log_probs_grad.mean()  # 最大化 log prob
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits 梯度范数: {logits_grad.grad.norm().item():.4f}")
    
    assert logits_grad.grad is not None, "应该有梯度"
    assert logits_grad.grad.norm() > 0, "梯度应该非零"
    
    print("✅ 梯度传播测试通过")
    
    # ========== 总结 ==========
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！")
    print("="*60)
    print("\n核心功能：")
    print("  ✓ 正确计算序列 log prob")
    print("  ✓ 正确处理 IGNORE_INDEX masking")
    print("  ✓ 数值范围合理")
    print("  ✓ KL 散度计算正确")
    print("  ✓ 支持梯度传播")
    print("\n可以用于 PPO 训练！")
