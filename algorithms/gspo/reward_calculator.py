"""
奖励函数计算模块
多维度评估 CoT 质量
"""
import re
from typing import Dict, List, Tuple

class RewardCalculator:
    """
    多维度奖励计算器
    
    总分 100:
    - Task Success: 60 分
    - Landmark Coverage: 20 分  
    - Reasoning Quality: 15 分
    - Consistency: 5 分
    """
    
    def __init__(self):
        self.weights = {
            'task_success': 60,
            'landmark_coverage': 20,
            'reasoning_quality': 15,
            'consistency': 5
        }
        
        # 推理质量检测的关键词
        self.logic_words = [
            'because', 'since', 'therefore', 'so', 'thus', 
            'hence', 'as', 'given', 'based on', 'according to'
        ]
    
    def calculate(
        self, 
        instruction: str,
        thinking: str,
        predicted_action: int,
        ground_truth_action: int,
        generated_cot: str
    ) -> Dict[str, float]:
        """
        计算总奖励和各维度得分
        
        Args:
            instruction: 原始指令
            thinking: 生成的思考过程
            predicted_action: 预测的动作
            ground_truth_action: 真实动作
            generated_cot: 完整的 CoT 输出
            
        Returns:
            {
                'total': 总分,
                'task_success': 任务成功得分,
                'landmark_coverage': 地标覆盖得分,
                'reasoning_quality': 推理质量得分,
                'consistency': 一致性得分,
                'breakdown': 详细分解
            }
        """
        scores = {}
        breakdown = {}
        
        # 1. Task Success (60分)
        task_score, task_breakdown = self._calculate_task_success(
            predicted_action, ground_truth_action
        )
        scores['task_success'] = task_score
        breakdown['task_success'] = task_breakdown
        
        # 2. Landmark Coverage (20分)
        landmark_score, landmark_breakdown = self._calculate_landmark_coverage(
            instruction, thinking
        )
        scores['landmark_coverage'] = landmark_score
        breakdown['landmark_coverage'] = landmark_breakdown
        
        # 3. Reasoning Quality (15分)
        reasoning_score, reasoning_breakdown = self._calculate_reasoning_quality(
            thinking
        )
        scores['reasoning_quality'] = reasoning_score
        breakdown['reasoning_quality'] = reasoning_breakdown
        
        # 4. Consistency (5分)
        consistency_score, consistency_breakdown = self._calculate_consistency(
            thinking, predicted_action, generated_cot
        )
        scores['consistency'] = consistency_score
        breakdown['consistency'] = consistency_breakdown
        
        # 计算总分
        total = sum(scores.values())
        
        
        return {
            'total': total,
            'task_success': scores['task_success'],
            'task_score': scores['task_success'],  # 兼容旧字段名
            'landmark_coverage': scores['landmark_coverage'],
            'landmark_score': scores['landmark_coverage'],  # 兼容旧字段名
            'reasoning_quality': scores['reasoning_quality'],
            'reasoning_score': scores['reasoning_quality'],  # 兼容旧字段名
            'consistency': scores['consistency'],
            'consistency_score': scores['consistency'],  # 兼容旧字段名
            'breakdown': breakdown
        }
    
    def _calculate_task_success(
        self, 
        predicted_action: int, 
        ground_truth_action: int
    ) -> Tuple[float, Dict]:
        """
        任务成功评分
        
        Returns:
            (score, breakdown)
        """
        max_score = self.weights['task_success']
        
        if predicted_action == ground_truth_action:
            # 完全正确
            score = max_score
            breakdown = {
                'correct': True,
                'predicted': predicted_action,
                'ground_truth': ground_truth_action
            }
        else:
            # 错误，但给少量分以提供梯度信号
            # 这对 RL 训练很重要
            score = max_score * 0.1  # 给 10% (6分)
            breakdown = {
                'correct': False,
                'predicted': predicted_action,
                'ground_truth': ground_truth_action
            }
        
        return score, breakdown
    
    def _calculate_landmark_coverage(
        self, 
        instruction: str, 
        thinking: str
    ) -> Tuple[float, Dict]:
        """
        地标覆盖评分
        
        检查 CoT 是否提到指令中的关键地标
        """
        # 提取指令中的地标
        landmarks = self._extract_landmarks(instruction)
        
        if len(landmarks) == 0:
            # 没有明确地标，给满分
            return self.weights['landmark_coverage'], {
                'landmarks': [],
                'mentioned': [],
                'coverage': 1.0
            }
        
        # 检查 thinking 中提到了哪些
        thinking_lower = thinking.lower()
        mentioned = []
        
        for landmark in landmarks:
            if landmark.lower() in thinking_lower:
                mentioned.append(landmark)
        
        # 计算覆盖率
        coverage = len(mentioned) / len(landmarks)
        score = coverage * self.weights['landmark_coverage']
        
        breakdown = {
            'landmarks': landmarks,
            'mentioned': mentioned,
            'coverage': coverage
        }
        
        return score, breakdown
    
    def _extract_landmarks(self, instruction: str) -> List[str]:
        """
        从指令中提取关键地标
        
        例如: "gray building", "beige building with antenna"
        """
        landmarks = []
        
        # 匹配模式: [颜色/大小] + building/structure
        patterns = [
            r'(gray|grey|beige|brown|white|black|light|dark)\s+(?:multi-story\s+)?building',
            r'(tall|large|small|big)\s+building',
            r'building\s+with\s+(windows|antenna|rooftop|shutters)',
            r'(light\s+brown|light\s+beige|orange-tinted)\s+(?:windows|building)'
        ]
        
        instruction_lower = instruction.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, instruction_lower)
            for match in matches:
                if isinstance(match, tuple):
                    # 如果有捕获组，取第一个
                    landmark = match[0] if match[0] else match
                else:
                    landmark = match
                
                # 构造完整描述
                # 找到匹配的完整短语
                full_match = re.search(r'\b\w+\s+' + re.escape(landmark) + r'\s+\w+', instruction_lower)
                if full_match:
                    landmarks.append(full_match.group().strip())
                elif landmark:
                    landmarks.append(landmark.strip())
        
        # 去重
        landmarks = list(set(landmarks))
        
        return landmarks
    
    def _calculate_reasoning_quality(self, thinking: str) -> Tuple[float, Dict]:
        """
        推理质量评分
        
        考虑:
        1. 长度适中 (5分)
        2. 有逻辑词 (5分)
        3. 无重复 (5分)
        """
        max_score = self.weights['reasoning_quality']
        score = 0
        breakdown = {}
        
        thinking_len = len(thinking)
        
        # 1. 长度适中 (100-400 字符) - 5分
        if 100 <= thinking_len <= 400:
            length_score = 5
        elif 50 <= thinking_len < 100:
            # 稍短，部分分
            length_score = 3
        elif 400 < thinking_len <= 600:
            # 稍长，部分分
            length_score = 3
        elif thinking_len < 50:
            # 太短
            length_score = 0
        else:
            # 太长
            length_score = 1
        
        score += length_score
        breakdown['length'] = {
            'value': thinking_len,
            'score': length_score,
            'optimal_range': [100, 400]
        }
        
        # 2. 逻辑词 (至少1个) - 5分
        thinking_lower = thinking.lower()
        logic_words_found = [
            word for word in self.logic_words 
            if word in thinking_lower
        ]
        
        if len(logic_words_found) >= 2:
            logic_score = 5
        elif len(logic_words_found) == 1:
            logic_score = 3
        else:
            logic_score = 0
        
        score += logic_score
        breakdown['logic_words'] = {
            'found': logic_words_found,
            'count': len(logic_words_found),
            'score': logic_score
        }
        
        # 3. 无明显重复 - 5分
        has_rep = self._has_repetition(thinking)
        repetition_score = 0 if has_rep else 5
        
        score += repetition_score
        breakdown['repetition'] = {
            'detected': has_rep,
            'score': repetition_score
        }
        
        return score, breakdown
    
    def _has_repetition(self, text: str) -> bool:
        """
        检测是否有明显的连续重复
        
        例如: "the building the building the building"
        """
        words = text.lower().split()
        
        # 检查连续 3 个相同词
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True
        
        # 检查连续 2 个相同短语 (3 词组)
        for i in range(len(words) - 5):
            phrase1 = ' '.join(words[i:i+3])
            phrase2 = ' '.join(words[i+3:i+6])
            if phrase1 == phrase2:
                return True
        
        return False
    
    def _calculate_consistency(
        self, 
        thinking: str, 
        predicted_action: int,
        generated_cot: str
    ) -> Tuple[float, Dict]:
        """
        一致性评分
        
        检查:
        1. CoT 中提到的 action 和最终 action 一致
        2. 格式完整（有 <thinking> 和 <action> 标签）
        """
        max_score = self.weights['consistency']
        score = 0
        breakdown = {}
        
        # 1. 格式完整性 (2.5分)
        has_thinking_tag = '<thinking>' in generated_cot and '</thinking>' in generated_cot
        has_action_tag = '<action>' in generated_cot and '</action>' in generated_cot
        
        if has_thinking_tag and has_action_tag:
            format_score = 2.5
        elif has_thinking_tag or has_action_tag:
            format_score = 1.0
        else:
            format_score = 0
        
        score += format_score
        breakdown['format'] = {
            'has_thinking_tag': has_thinking_tag,
            'has_action_tag': has_action_tag,
            'score': format_score
        }
        
        # 2. 内部一致性 (2.5分)
        # 检查 thinking 中提到的 action 数字
        thinking_lower = thinking.lower()
        mentioned_actions = re.findall(r'(?:action|move|go)\s*(?:is|:|=)?\s*(\d+)', thinking_lower)
        
        if mentioned_actions:
            # 取最后一个提到的 action
            last_mentioned = int(mentioned_actions[-1])
            if last_mentioned == predicted_action:
                internal_score = 2.5
            else:
                internal_score = 0
            
            breakdown['internal_consistency'] = {
                'mentioned_actions': [int(a) for a in mentioned_actions],
                'last_mentioned': last_mentioned,
                'predicted': predicted_action,
                'match': (last_mentioned == predicted_action),
                'score': internal_score
            }
        else:
            # 没有明确提到 action，给部分分
            internal_score = 1.0
            breakdown['internal_consistency'] = {
                'mentioned_actions': [],
                'score': internal_score
            }
        
        score += internal_score
        
        return score, breakdown


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """测试 RewardCalculator"""
    
    calculator = RewardCalculator()
    
    # 测试案例 1: 完美的 CoT
    print("="*60)
    print("测试案例 1: 完美的 CoT")
    print("="*60)
    
    instruction1 = "Move forward to a gray building, then turn right to a beige building"
    thinking1 = "I can see a gray building ahead. Based on the instruction, I should move forward toward it. The action is 9 (move forward)."
    predicted_action1 = 9
    ground_truth_action1 = 9
    generated_cot1 = f"<thinking>{thinking1}</thinking><action>9</action>"
    
    result1 = calculator.calculate(
        instruction1, thinking1, predicted_action1, ground_truth_action1, generated_cot1
    )
    
    print(f"\n总分: {result1['total']:.1f}/100")
    print(f"  - Task Success: {result1['task_success']:.1f}/{calculator.weights['task_success']}")
    print(f"  - Landmark Coverage: {result1['landmark_coverage']:.1f}/{calculator.weights['landmark_coverage']}")
    print(f"  - Reasoning Quality: {result1['reasoning_quality']:.1f}/{calculator.weights['reasoning_quality']}")
    print(f"  - Consistency: {result1['consistency']:.1f}/{calculator.weights['consistency']}")
    
    # 测试案例 2: 错误的 action
    print("\n" + "="*60)
    print("测试案例 2: 错误的 action")
    print("="*60)
    
    thinking2 = "I see something. I will turn left."
    predicted_action2 = 3
    ground_truth_action2 = 9
    generated_cot2 = f"<thinking>{thinking2}</thinking><action>3</action>"
    
    result2 = calculator.calculate(
        instruction1, thinking2, predicted_action2, ground_truth_action2, generated_cot2
    )
    
    print(f"\n总分: {result2['total']:.1f}/100")
    print(f"  - Task Success: {result2['task_success']:.1f}")
    print(f"  - Landmark Coverage: {result2['landmark_coverage']:.1f}")
    print(f"  - Reasoning Quality: {result2['reasoning_quality']:.1f}")
    print(f"  - Consistency: {result2['consistency']:.1f}")
    
    print("\n✅ RewardCalculator 测试完成")
