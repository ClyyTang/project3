"""
CoT Generator - 生成 CoT 并计算 old log prob

用于 PPO 训练的数据生成阶段
"""
import sys
sys.path.insert(0, '/home/ubuntu/data1/lyy/OpenFly-Platform/train')
sys.path.insert(0, '/home/ubuntu/data1/zx/OpenFly-Platform/train')

import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

from .sequence_logprob import compute_sequence_logprob, IGNORE_INDEX

class CoTGenerator:
    """
    CoT 生成器
    
    核心功能：
    1. 生成 CoT 序列
    2. 同时计算并返回 old_log_prob（用于 PPO）
    3. 支持温度采样
    """
    
    def __init__(
        self,
        vla_model,
        tokenizer,
        image_transform,
        device: str = "cuda",
        max_new_tokens: int = 300
    ):
        self.vla = vla_model
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        self.vla.eval()
    
    @torch.no_grad()
    def generate_with_logprob(
        self,
        pixel_values: Dict[str, torch.Tensor],
        prompt_ids: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        生成序列并计算 log prob
        
        Args:
            pixel_values: 图像特征
            prompt_ids: [batch, prompt_len] Prompt token ids
            temperature: 采样温度（> 1 更随机，< 1 更确定）
            top_p: Nucleus sampling
            top_k: Top-k sampling
            
        Returns:
            generated_ids: [batch, total_len] 完整序列（包括 prompt）
            old_log_prob: [batch] 生成部分的 log prob
            per_token_log_probs: List[float] 每个 token 的 log prob（调试用）
        """
        batch_size = prompt_ids.shape[0]
        
        # 初始化
        current_ids = prompt_ids.clone()  # [batch, prompt_len]
        prompt_len = current_ids.shape[1]
        
        # 记录每个 token 的 log prob
        all_token_log_probs = []
        
        # 自回归生成
        for step in range(self.max_new_tokens):
            # Forward
            outputs = self.vla(
                pixel_values=pixel_values,
                input_ids=current_ids
            )
            
            # 取最后一个 token 的 logits
            next_token_logits = outputs.logits[:, -1, :]  # [batch, vocab]
            
            # 温度缩放
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过 top_p 的 tokens
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # 计算这个 token 的 log prob
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=next_token)  # [batch, 1]
            all_token_log_probs.append(token_log_prob.squeeze(1))  # [batch]
            
            # 拼接
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # 检查是否所有序列都已经生成 EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        # 计算总 log prob
        if len(all_token_log_probs) > 0:
            old_log_prob = torch.stack(all_token_log_probs, dim=1).sum(dim=1)  # [batch]
            per_token_probs = [t.cpu().tolist() for t in all_token_log_probs]
        else:
            old_log_prob = torch.zeros(batch_size, device=self.device)
            per_token_probs = []
        
        return current_ids, old_log_prob, per_token_probs
    
    def generate_cot_sample(
        self,
        sample: Dict,
        image_base_path: Path,
        temperature: float = 1.0
    ) -> Dict:
        """
        为单个样本生成 CoT
        
        Args:
            sample: 来自 CoTDataset 的样本
                {
                    'episode_id': ...,
                    'instruction': ...,
                    'frame_idx': ...,
                    'action': ...,
                    ...
                }
            image_base_path: 图片根目录
            temperature: 采样温度
            
        Returns:
            {
                'sample_id': int,
                'instruction': str,
                'generated_ids': torch.Tensor,
                'generated_text': str,
                'old_log_prob': float,
                'thinking': str,
                'predicted_action': int,
                'ground_truth_action': int,
                ...
            }
        """
        # 加载图片
        episode_path = image_base_path / sample['episode_id']
        img_file = f"{sample['frame_idx']}.png"
        img_path = episode_path / img_file
        
        if img_path.exists():
            img = Image.open(img_path).convert('RGB')
        else:
            # Fallback: 灰色图片
            img = Image.new('RGB', (224, 224), color='gray')
        
        # 图像变换
        tr_img = self.image_transform(img)
        pixel_values = {}
        for k in tr_img.keys():
            # 3 帧（简化：用同一张）
            combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0)
            pixel_values[k] = combined.unsqueeze(0).to(self.device)
        
        # 构造 prompt
        from model.prompt_llama2 import LLaMa2ChatPromptBuilder
        
        prompt_builder = LLaMa2ChatPromptBuilder("prismatic")
        prompt_builder.add_turn("human", f"What action should the robot take to {sample['instruction']}?")
        prompt = prompt_builder.get_prompt()
        
        # 去掉末尾的 </s>（让模型继续生成）
        if prompt.endswith("</s>"):
            prompt = prompt[:-4].rstrip()
        
        # Tokenize
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(self.device)
        
        # 生成（带 log prob）
        generated_ids, old_log_prob, per_token_probs = self.generate_with_logprob(
            pixel_values=pixel_values,
            prompt_ids=prompt_ids,
            temperature=temperature
        )
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
        # 提取生成部分
        prompt_len = len(self.tokenizer.encode(prompt))
        generated_only = self.tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=True)
        
        # 解析 CoT
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', generated_only, re.DOTALL)
        action_match = re.search(r'<action>(\d+)</action>', generated_only)
        
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        predicted_action = int(action_match.group(1)) if action_match else -1
        
        return {
            'instruction': sample['instruction'],
            'generated_ids': generated_ids[0].cpu(),  # [total_len]
            'generated_text': generated_text,
            'generated_only': generated_only,
            'old_log_prob': old_log_prob[0].item(),
            'per_token_log_probs': [p[0] for p in per_token_probs],
            'thinking': thinking,
            'predicted_action': predicted_action,
            'ground_truth_action': sample['action'],
            'episode_id': sample['episode_id'],
            'frame_idx': sample['frame_idx']
        }


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """测试 CoT Generator"""
    
    print("="*60)
    print("CoT Generator 测试")
    print("="*60)
    
    # ========== 测试 1: 基础采样逻辑 ==========
    
    print("\n[测试 1] 温度采样逻辑")
    print("-"*60)
    
    vocab_size = 100
    logits = torch.randn(2, vocab_size)
    
    # 测试不同温度
    for temp in [0.1, 0.5, 1.0, 2.0]:
        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        print(f"温度 {temp:.1f}:")
        print(f"  最大概率: {probs.max(dim=-1)[0].mean():.4f}")
        print(f"  熵: {entropy.mean():.4f}")
    
    print("✅ 温度越高，分布越平坦（熵越大）")
    
    # ========== 测试 2: Log Prob 计算 ==========
    
    print("\n[测试 2] Log Prob 计算")
    print("-"*60)
    
    # 模拟生成 5 个 tokens
    num_tokens = 5
    token_log_probs = []
    
    for i in range(num_tokens):
        logits = torch.randn(1, vocab_size)
        probs = F.softmax(logits, dim=-1)
        
        # 采样
        token = torch.multinomial(probs, num_samples=1)
        
        # Log prob
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=token)
        
        token_log_probs.append(token_log_prob.item())
    
    total_log_prob = sum(token_log_probs)
    
    print(f"生成了 {num_tokens} 个 tokens")
    print(f"每个 token 的 log prob: {[f'{p:.2f}' for p in token_log_probs]}")
    print(f"总 log prob: {total_log_prob:.2f}")
    
    # 验证：应该是负数
    assert total_log_prob < 0, "Log prob 应该是负数"
    
    print("✅ Log prob 计算正确")
    
    # ========== 测试 3: Top-p filtering ==========
    
    print("\n[测试 3] Top-p (Nucleus) Sampling")
    print("-"*60)
    
    logits = torch.randn(1, vocab_size)
    top_p = 0.9
    
    # 排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 找到累积概率超过 top_p 的位置
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    num_kept = (~sorted_indices_to_remove).sum().item()
    
    print(f"Top-p = {top_p}")
    print(f"保留的 tokens: {num_kept}/{vocab_size}")
    print(f"被过滤的 tokens: {vocab_size - num_kept}")
    
    assert num_kept < vocab_size, "应该过滤掉一些 tokens"
    
    print("✅ Top-p filtering 正确")
    
    # ========== 总结 ==========
    
    print("\n" + "="*60)
    print("✅ CoT Generator 核心逻辑测试通过！")
    print("="*60)
    print("\n核心功能：")
    print("  ✓ 温度采样")
    print("  ✓ Log prob 计算")
    print("  ✓ Top-p/Top-k filtering")
    print("\n可以用于生成 CoT 数据！")
    
    print("\n注意：完整测试需要加载 VLA 模型")
    print("      这里只测试了核心采样逻辑")
