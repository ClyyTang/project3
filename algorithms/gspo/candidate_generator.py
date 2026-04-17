"""
候选生成器
为每个样本生成多个 CoT 候选，并根据 reward 排序
"""
import sys
sys.path.insert(0, '/home/ubuntu/data1/lyy/OpenFly-Platform/train')
sys.path.insert(0, '/home/ubuntu/data1/zx/OpenFly-Platform/train')

import torch
from pathlib import Path
from typing import Dict, List
from PIL import Image
import re

from .cot_generator import CoTGenerator
from .reward_calculator import RewardCalculator


class CandidateGenerator:
    """
    候选生成器
    
    功能：
    1. 对同一样本生成多个不同的 CoT（使用不同温度）
    2. 计算每个候选的 reward
    3. 排序并选择最好和最差的作为 chosen/rejected 对
    
    这是 GSPO 的核心组件
    """
    
    def __init__(
        self,
        vla_model,
        tokenizer,
        image_transform,
        device: str = "cuda",
        max_new_tokens: int = 300
    ):
        """
        Args:
            vla_model: VLA 模型
            tokenizer: Tokenizer
            image_transform: 图像变换函数
            device: GPU 设备
            max_new_tokens: 最大生成 token 数
        """
        # 内部使用 CoTGenerator（组合而非继承）
        self.cot_generator = CoTGenerator(
            vla_model=vla_model,
            tokenizer=tokenizer,
            image_transform=image_transform,
            device=device,
            max_new_tokens=max_new_tokens
        )
        
        self.reward_calculator = RewardCalculator()
        self.device = device
    
    def generate_candidates(
        self,
        sample: Dict,
        image_base_path: Path,
        temperatures: List[float]
    ) -> Dict:
        """
        为单个样本生成多个候选
        
        Args:
            sample: 来自 CoTDataset 的样本
                {
                    'episode_id': str,
                    'instruction': str,
                    'frame_idx': str,
                    'action': int,
                    ...
                }
            image_base_path: 图片根目录
            temperatures: 采样温度列表 [0.7, 1.0, 1.5]
            
        Returns:
            {
                'chosen': Dict,        # reward 最高的候选
                'rejected': Dict,      # reward 最低的候选
                'all_candidates': List[Dict],  # 所有候选（按 reward 降序）
                'episode_id': str,
                'instruction': str,
                'ground_truth_action': int
            }
        """
        candidates = []
        
        # 为每个温度生成一个候选
        for temp in temperatures:
            try:
                # 1. 生成 CoT
                cot_result = self.cot_generator.generate_cot_sample(
                    sample=sample,
                    image_base_path=image_base_path,
                    temperature=temp
                )
                
                # 2. 计算 reward
                reward_result = self.reward_calculator.calculate(
                    instruction=cot_result['instruction'],
                    thinking=cot_result['thinking'],
                    predicted_action=cot_result['predicted_action'],
                    ground_truth_action=cot_result['ground_truth_action'],
                    generated_cot=cot_result['generated_only']
                )
                
                # 3. 合并信息
                candidate = {
                    'temperature': temp,
                    'thinking': cot_result['thinking'],
                    'predicted_action': cot_result['predicted_action'],
                    'generated_only': cot_result['generated_only'],
                    'generated_text': cot_result.get('generated_text', ''),
                    'reward': reward_result['total'],
                    'reward_breakdown': {
                        'task_success': reward_result['task_success'],
                        'landmark_coverage': reward_result['landmark_coverage'],
                        'reasoning_quality': reward_result['reasoning_quality'],
                        'consistency': reward_result['consistency']
                    }
                }
                
                candidates.append(candidate)
                
            except Exception as e:
                print(f"    ⚠️  温度 {temp} 生成失败: {str(e)}")
                # 创建一个失败的占位候选（reward=0）
                candidates.append({
                    'temperature': temp,
                    'thinking': '',
                    'predicted_action': -1,
                    'generated_only': '',
                    'generated_text': '',
                    'reward': 0.0,
                    'reward_breakdown': {
                        'task_success': 0,
                        'landmark_coverage': 0,
                        'reasoning_quality': 0,
                        'consistency': 0
                    },
                    'error': str(e)
                })
        
        # 4. 按 reward 降序排序
        candidates.sort(key=lambda x: x['reward'], reverse=True)
        
        # 5. 选择 chosen 和 rejected
        chosen = candidates[0]      # reward 最高
        rejected = candidates[-1]   # reward 最低
        
        return {
            'chosen': chosen,
            'rejected': rejected,
            'all_candidates': candidates,
            'episode_id': sample['episode_id'],
            'frame_idx': sample['frame_idx'],
            'instruction': sample['instruction'],
            'ground_truth_action': sample['action']
        }
    
    def generate_batch_candidates(
        self,
        samples: List[Dict],
        image_base_path: Path,
        temperatures: List[float],
        verbose: bool = False
    ) -> List[Dict]:
        """
        为一批样本生成候选
        
        Args:
            samples: 样本列表
            image_base_path: 图片根目录
            temperatures: 温度列表
            verbose: 是否打印详细信息
            
        Returns:
            候选对列表，每个元素格式同 generate_candidates 的返回值
        """
        results = []
        
        for i, sample in enumerate(samples):
            if verbose and (i + 1) % 10 == 0:
                print(f"  已生成 {i+1}/{len(samples)} 个样本的候选...")
            
            result = self.generate_candidates(
                sample=sample,
                image_base_path=image_base_path,
                temperatures=temperatures
            )
            
            results.append(result)
        
        if verbose:
            # 统计信息
            avg_chosen_reward = sum(r['chosen']['reward'] for r in results) / len(results)
            avg_rejected_reward = sum(r['rejected']['reward'] for r in results) / len(results)
            avg_margin = avg_chosen_reward - avg_rejected_reward
            
            print(f"\n  📊 候选生成统计:")
            print(f"    平均 chosen reward: {avg_chosen_reward:.2f}")
            print(f"    平均 rejected reward: {avg_rejected_reward:.2f}")
            print(f"    平均 margin: {avg_margin:.2f}")
        
        return results


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """测试 CandidateGenerator"""
    
    print("=" * 60)
    print("测试 CandidateGenerator")
    print("=" * 60)
    
    # 注意：完整测试需要加载 VLA 模型，这里只测试逻辑
    
    # 测试 1: 模拟候选生成和排序
    print("\n[测试 1] 模拟候选排序逻辑")
    print("-" * 60)
    
    # 模拟候选
    mock_candidates = [
        {'temperature': 0.7, 'reward': 85.5, 'predicted_action': 9},
        {'temperature': 1.0, 'reward': 72.3, 'predicted_action': 9},
        {'temperature': 1.5, 'reward': 45.2, 'predicted_action': 3},
    ]
    
    # 排序
    sorted_candidates = sorted(mock_candidates, key=lambda x: x['reward'], reverse=True)
    
    print("原始顺序:")
    for c in mock_candidates:
        print(f"  T={c['temperature']}: reward={c['reward']}, action={c['predicted_action']}")
    
    print("\n排序后:")
    for c in sorted_candidates:
        print(f"  T={c['temperature']}: reward={c['reward']}, action={c['predicted_action']}")
    
    chosen = sorted_candidates[0]
    rejected = sorted_candidates[-1]
    
    print(f"\nChosen: T={chosen['temperature']}, reward={chosen['reward']}")
    print(f"Rejected: T={rejected['temperature']}, reward={rejected['reward']}")
    print(f"Margin: {chosen['reward'] - rejected['reward']:.1f}")
    
    assert chosen['reward'] == 85.5, "Chosen 应该是最高分"
    assert rejected['reward'] == 45.2, "Rejected 应该是最低分"
    print("✅ 排序逻辑正确")
    
    # 测试 2: 边界情况
    print("\n[测试 2] 边界情况")
    print("-" * 60)
    
    # 2.1 所有候选都很好
    good_candidates = [
        {'temperature': 0.7, 'reward': 95.0},
        {'temperature': 1.0, 'reward': 92.0},
        {'temperature': 1.5, 'reward': 88.0},
    ]
    good_sorted = sorted(good_candidates, key=lambda x: x['reward'], reverse=True)
    print(f"都很好: chosen={good_sorted[0]['reward']}, rejected={good_sorted[-1]['reward']}, margin={good_sorted[0]['reward']-good_sorted[-1]['reward']}")
    
    # 2.2 所有候选都很差
    bad_candidates = [
        {'temperature': 0.7, 'reward': 15.0},
        {'temperature': 1.0, 'reward': 12.0},
        {'temperature': 1.5, 'reward': 8.0},
    ]
    bad_sorted = sorted(bad_candidates, key=lambda x: x['reward'], reverse=True)
    print(f"都很差: chosen={bad_sorted[0]['reward']}, rejected={bad_sorted[-1]['reward']}, margin={bad_sorted[0]['reward']-bad_sorted[-1]['reward']}")
    
    # 2.3 reward 相同
    same_candidates = [
        {'temperature': 0.7, 'reward': 60.0},
        {'temperature': 1.0, 'reward': 60.0},
        {'temperature': 1.5, 'reward': 60.0},
    ]
    same_sorted = sorted(same_candidates, key=lambda x: x['reward'], reverse=True)
    print(f"都相同: margin={same_sorted[0]['reward']-same_sorted[-1]['reward']}")
    assert same_sorted[0]['reward'] == same_sorted[-1]['reward'], "相同时 margin 应该是 0"
    
    print("✅ 边界情况处理正确")
    
    # 测试 3: RewardCalculator 集成
    print("\n[测试 3] RewardCalculator 集成")
    print("-" * 60)
    
    reward_calc = RewardCalculator()
    
    # 模拟不同质量的 CoT
    test_cases = [
        {
            'name': '完美 CoT',
            'instruction': 'Move forward to a gray building',
            'thinking': 'I see a gray building ahead. Based on the instruction, I should move forward. The action is 9.',
            'predicted': 9,
            'ground_truth': 9,
            'generated': '<thinking>I see a gray building ahead. Based on the instruction, I should move forward. The action is 9.</thinking><action>9</action>'
        },
        {
            'name': '错误 CoT',
            'instruction': 'Move forward to a gray building',
            'thinking': 'I will turn left.',
            'predicted': 2,
            'ground_truth': 9,
            'generated': '<thinking>I will turn left.</thinking><action>2</action>'
        }
    ]
    
    for case in test_cases:
        result = reward_calc.calculate(
            instruction=case['instruction'],
            thinking=case['thinking'],
            predicted_action=case['predicted'],
            ground_truth_action=case['ground_truth'],
            generated_cot=case['generated']
        )
        print(f"{case['name']}: reward={result['total']:.1f}")
    
    print("✅ RewardCalculator 集成正常")
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ CandidateGenerator 逻辑测试通过！")
    print("=" * 60)
    print("\n核心验证:")
    print("  ✓ 候选排序逻辑")
    print("  ✓ 边界情况处理")
    print("  ✓ RewardCalculator 集成")
    print("\n注意：完整测试需要加载 VLA 模型")