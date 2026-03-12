"""
诊断式评分器 (Diagnosis Scorer) - 修复版

修复内容：
1. 修正 locate → locate_root_cause
2. 修正参数格式，适配 RootCauseLocator.locate_root_cause 的输入
3. 在返回结果中正确包含诊断信息（error_type, error_step, confidence）
4. 改进 fallback 逻辑

功能：评估GSPO生成的候选CoT质量
用途：在Stage 2 GSPO训练中，为候选CoT打分，选出chosen/rejected
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 导入已有的诊断模块
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from cot_parser import CoTParser
    from root_cause_locator import RootCauseLocator
except ImportError:
    print("⚠️  警告: 无法导入诊断模块，请确保cot_parser.py和root_cause_locator.py在同一目录")
    CoTParser = None
    RootCauseLocator = None


class DiagnosisScorer:
    """
    诊断式评分器
    
    基于诊断框架为候选CoT打分，用于GSPO训练中的候选选择
    """
    
    def __init__(
        self,
        cot_parser: Optional['CoTParser'] = None,
        root_cause_locator: Optional['RootCauseLocator'] = None,
        error_penalties: Optional[Dict[int, float]] = None
    ):
        """
        初始化评分器
        
        Args:
            cot_parser: CoT分解器实例（可选，用于独立分析）
            root_cause_locator: 根因定位器实例（None则使用简单规则）
            error_penalties: 错误扣分规则（None则使用默认）
        """
        print("🚀 初始化诊断式评分器...")
        
        self.cot_parser = cot_parser
        self.root_cause_locator = root_cause_locator
        
        if self.cot_parser is None:
            print("   ⚠️  未传入CoT分解器（将使用简单规则评分）")
        else:
            print("   ✓ CoT分解器已配置")
        
        if self.root_cause_locator is None:
            print("   ⚠️  未传入根因定位器（将使用简单规则评分）")
        else:
            print("   ✓ 根因定位器已配置")
        
        # 错误扣分规则（按错误步骤）
        if error_penalties is None:
            self.error_penalties = {
                0: 0.4,   # step 0错误（perception）- 最严重，影响后续所有步骤
                1: 0.3,   # step 1错误（comprehension）
                2: 0.2,   # step 2错误（reasoning）
                3: 0.1,   # step 3+错误（decision）
            }
        else:
            self.error_penalties = error_penalties
        
        # 错误类型到扣分的映射（备用）
        self.error_type_penalties = {
            'perception': 0.4,
            'comprehension': 0.3,
            'reasoning': 0.2,
            'decision': 0.1,
            'unknown': 0.15
        }
        
        # 评分参数
        self.base_score = 0.9          # 无错误的基础分
        self.keyword_bonus_weight = 0.1  # 关键词加成权重
        
        print(f"   ✓ 错误扣分规则: {self.error_penalties}")
        print(f"   ✓ 基础分数: {self.base_score}")
        print("✅ 诊断式评分器初始化完成\n")
    
    def _extract_keywords(self, text: str) -> set:
        """从文本中提取关键词"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        stopwords = {
            'a', 'an', 'the', 'to', 'and', 'or', 'in', 'on', 'at', 'of', 'for',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that',
            'see', 'need', 'want', 'think', 'know', 'move', 'go', 'get'
        }
        
        keywords = set(w for w in words if w not in stopwords and len(w) > 2)
        return keywords
    
    def _compute_keyword_coverage(
        self, 
        candidate_cot: str, 
        gt_cot: str,
        instruction: str
    ) -> float:
        """计算候选CoT的关键词覆盖度"""
        instruction_keywords = self._extract_keywords(instruction)
        gt_keywords = self._extract_keywords(gt_cot)
        candidate_keywords = self._extract_keywords(candidate_cot)
        
        important_keywords = instruction_keywords | gt_keywords
        
        if len(important_keywords) == 0:
            return 0.5
        
        covered = candidate_keywords & important_keywords
        coverage = len(covered) / len(important_keywords)
        
        return coverage
    
    def _compute_error_penalty(self, diagnosis: Dict) -> float:
        """根据诊断结果计算扣分"""
        error_step = diagnosis.get('error_step', -1)
        error_type = diagnosis.get('error_type', 'unknown')
        
        # 无错误
        if error_step < 0 and error_type == 'unknown':
            return 0.0
        
        # 优先使用 error_step
        if error_step >= 0:
            if error_step <= 2:
                penalty = self.error_penalties.get(error_step, 0.1)
            else:
                penalty = self.error_penalties.get(3, 0.1)
            return penalty
        
        # 其次使用 error_type
        if error_type in self.error_type_penalties:
            return self.error_type_penalties[error_type]
        
        return 0.15  # 默认扣分
    
    def _simple_rule_diagnosis(
        self,
        candidate_cot: str,
        gt_cot: str,
        instruction: str
    ) -> Dict:
        """
        简单规则诊断（fallback方案）
        
        当 RootCauseLocator 不可用时使用
        """
        diagnosis = {
            'error_step': -1,
            'error_type': 'unknown',
            'confidence': 'low',
            'method': 'simple_rule'
        }
        
        candidate_lower = candidate_cot.lower()
        instruction_lower = instruction.lower()
        gt_lower = gt_cot.lower()
        
        # 1. 检查感知错误（关键物体缺失）
        key_objects = []
        object_words = ['building', 'tower', 'house', 'tree', 'road', 'wall', 'area']
        color_words = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'grey']
        
        for word in object_words + color_words:
            if word in instruction_lower:
                key_objects.append(word)
        
        missing_objects = [obj for obj in key_objects if obj not in candidate_lower]
        
        if len(missing_objects) > len(key_objects) * 0.5:
            diagnosis['error_step'] = 0
            diagnosis['error_type'] = 'perception'
            diagnosis['confidence'] = 'medium'
            diagnosis['issue'] = f"Missing key objects: {missing_objects}"
            return diagnosis
        
        # 2. 检查方向理解错误
        direction_pairs = [
            ('left', 'right'),
            ('forward', 'backward'),
            ('up', 'down')
        ]
        
        for dir1, dir2 in direction_pairs:
            if dir1 in instruction_lower and dir2 in candidate_lower and dir1 not in candidate_lower:
                diagnosis['error_step'] = 1
                diagnosis['error_type'] = 'comprehension'
                diagnosis['confidence'] = 'high'
                diagnosis['issue'] = f"Direction confusion: {dir1} vs {dir2}"
                return diagnosis
            if dir2 in instruction_lower and dir1 in candidate_lower and dir2 not in candidate_lower:
                diagnosis['error_step'] = 1
                diagnosis['error_type'] = 'comprehension'
                diagnosis['confidence'] = 'high'
                diagnosis['issue'] = f"Direction confusion: {dir2} vs {dir1}"
                return diagnosis
        
        # 3. 检查与GT的相似度（推理质量）
        candidate_words = set(candidate_lower.split())
        gt_words = set(gt_lower.split())
        
        if len(gt_words) > 0:
            overlap = len(candidate_words & gt_words)
            similarity = overlap / len(gt_words)
            
            if similarity < 0.3:
                diagnosis['error_step'] = 2
                diagnosis['error_type'] = 'reasoning'
                diagnosis['confidence'] = 'medium'
                diagnosis['issue'] = f"Low similarity with GT: {similarity:.2f}"
                return diagnosis
        
        # 没有明显错误
        return diagnosis
    


    def _compute_text_similarity(self, text_a: str, text_b: str) -> float:
        """计算两段文本的词级相似度（Jaccard）"""
        words_a = set(re.findall(r'\b\w+\b', text_a.lower()))
        words_b = set(re.findall(r'\b\w+\b', text_b.lower()))
        
        if len(words_a) == 0 and len(words_b) == 0:
            return 1.0
        if len(words_a) == 0 or len(words_b) == 0:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0

    def _compute_ngram_similarity(self, text_a: str, text_b: str, n: int = 2) -> float:
        """计算2-gram相似度（捕捉短语级匹配）"""
        def get_ngrams(text, n):
            words = re.findall(r'\b\w+\b', text.lower())
            if len(words) < n:
                return set()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
        
        ngrams_a = get_ngrams(text_a, n)
        ngrams_b = get_ngrams(text_b, n)
        
        if len(ngrams_a) == 0 and len(ngrams_b) == 0:
            return 1.0
        if len(ngrams_a) == 0 or len(ngrams_b) == 0:
            return 0.0
        
        intersection = len(ngrams_a & ngrams_b)
        union = len(ngrams_a | ngrams_b)
        return intersection / union if union > 0 else 0.0

    def score_single(
        self,
        candidate_cot: str,
        gt_cot: str,
        instruction: str,
        candidate_actions: Optional[List[int]] = None,
        gt_actions: Optional[List[int]] = None,
        return_details: bool = False
    ):
        details = {
            'gt_similarity': 0.0,
            'keyword_coverage': 0.0,
            'action_score': 0.0,
            'error_penalty': 0.0,
            'error_type': 'unknown',
            'error_step': -1,
            'confidence': 'low',
            'diagnosis_method': 'none',
            'final_score': 0.0
        }

        # 1. GT文本相似度（主要区分维度）
        word_sim = self._compute_text_similarity(candidate_cot, gt_cot)
        ngram_sim = self._compute_ngram_similarity(candidate_cot, gt_cot, n=2)
        gt_similarity = 0.6 * word_sim + 0.4 * ngram_sim
        details['gt_similarity'] = gt_similarity

        # 2. 关键词覆盖度
        keyword_coverage = self._compute_keyword_coverage(candidate_cot, gt_cot, instruction)
        details['keyword_coverage'] = keyword_coverage

        # 3. 动作匹配度
        action_score = 0.5  # 默认中性
        if candidate_actions and gt_actions:
            if len(candidate_actions) > 0 and len(gt_actions) > 0:
                if candidate_actions[0] == gt_actions[0]:
                    action_score = 1.0
                else:
                    action_score = 0.2
        details['action_score'] = action_score

        # 4. 错误诊断（保留原逻辑，但降权）
        diagnosis = None
        if self.root_cause_locator is not None:
            try:
                failure_case = {
                    'sample_id': 'candidate_evaluation',
                    'instruction': instruction,
                    'trajectory': {
                        'cot_list': [candidate_cot] if candidate_cot else [],
                        'actions': candidate_actions if candidate_actions else []
                    }
                }
                diagnosis = self.root_cause_locator.locate_root_cause(failure_case)
                details['diagnosis_method'] = 'root_cause_locator'
            except Exception as e:
                diagnosis = None

        if diagnosis is None or diagnosis.get('error') is not None:
            diagnosis = self._simple_rule_diagnosis(candidate_cot, gt_cot, instruction)
            details['diagnosis_method'] = 'simple_rule'

        error_penalty = self._compute_error_penalty(diagnosis)
        normalized_penalty = min(error_penalty / 0.4, 1.0)
        details['error_penalty'] = error_penalty
        details['error_step'] = diagnosis.get('error_step', -1)
        details['error_type'] = diagnosis.get('error_type', 'unknown')
        details['confidence'] = diagnosis.get('confidence', 'low')

        # 5. 加权组合
        final_score = (
            0.35 * gt_similarity +
            0.25 * keyword_coverage +
            0.25 * action_score +
            0.15 * (1.0 - normalized_penalty)
        )
        final_score = max(0.0, min(1.0, final_score))
        details['final_score'] = final_score

        if return_details:
            return final_score, details
        else:
            return final_score

            
    
    def rank_candidates(
        self,
        candidates: List[Dict],
        gt: Dict,
        sample: Dict,
        return_all_scores: bool = False
    ) -> Dict:
        """
        排序多个候选CoT，选出最好和最差的
        
        Args:
            candidates: 候选列表，每个候选包含 {'cot': str, 'actions': list, 'original': dict}
            gt: Ground-Truth，包含 {'cot': str, 'actions': list}
            sample: 原始样本，包含 {'instruction': str, ...}
            return_all_scores: 是否返回所有候选的详细评分
            
        Returns:
            {
                'chosen': Dict,           # 最高分候选（包含 diagnosis 信息）
                'rejected': Dict,         # 最低分候选（包含 diagnosis 信息）
                'scores': List[float],    # 所有候选的分数
                'chosen_idx': int,
                'rejected_idx': int,
                'all_candidates': List[Dict],  # 所有候选（包含诊断信息）
                'details': List[Dict]     # 详细评分信息（如果 return_all_scores=True）
            }
        """
        if len(candidates) == 0:
            raise ValueError("候选列表不能为空")
        
        instruction = sample.get('instruction', '')
        gt_cot = gt.get('cot', '')
        gt_actions = gt.get('actions', [])
        
        scores = []
        all_details = []
        all_candidates_with_diagnosis = []
        
        for idx, candidate in enumerate(candidates):
            candidate_cot = candidate.get('cot', '')
            candidate_actions = candidate.get('actions', [])
            
            # 评分
            score, details = self.score_single(
                candidate_cot=candidate_cot,
                gt_cot=gt_cot,
                instruction=instruction,
                candidate_actions=candidate_actions,
                gt_actions=gt_actions,
                return_details=True
            )
            
            scores.append(score)
            all_details.append(details)
            
            # ⭐ 关键修复：将诊断信息添加到候选中
            candidate_with_diagnosis = candidate.copy()
            candidate_with_diagnosis['score'] = score
            candidate_with_diagnosis['diagnosis'] = {
                'error_type': details.get('error_type', 'unknown'),
                'error_step': details.get('error_step', -1),
                'confidence': details.get('confidence', 'low'),
                'diagnosis_method': details.get('diagnosis_method', 'none')
            }
            all_candidates_with_diagnosis.append(candidate_with_diagnosis)
        
        # 找最高分和最低分
        max_idx = scores.index(max(scores))
        min_idx = scores.index(min(scores))
        
        # 如果最高分和最低分相同（所有候选分数一样），选择不同的
        if max_idx == min_idx and len(candidates) > 1:
            min_idx = (max_idx + 1) % len(candidates)
        
        chosen = all_candidates_with_diagnosis[max_idx]
        rejected = all_candidates_with_diagnosis[min_idx]
        
        result = {
            'chosen': chosen,
            'rejected': rejected,
            'scores': scores,
            'chosen_idx': max_idx,
            'rejected_idx': min_idx,
            'all_candidates': all_candidates_with_diagnosis
        }
        
        if return_all_scores:
            result['details'] = all_details
        
        return result
    
    def batch_rank(
        self,
        samples_with_candidates: List[Dict],
        output_file: Optional[str] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """批量排序多个样本的候选"""
        results = []
        
        if verbose:
            print(f"\n{'='*60}")
            print("批量候选排序开始")
            print(f"{'='*60}")
            print(f"样本数: {len(samples_with_candidates)}")
        
        for idx, item in enumerate(samples_with_candidates):
            sample = item.get('sample', {})
            gt = item.get('gt', {})
            candidates = item.get('candidates', [])
            
            ranking_result = self.rank_candidates(
                candidates=candidates,
                gt=gt,
                sample=sample,
                return_all_scores=True
            )
            
            ranking_result['sample_id'] = item.get('sample_id', idx)
            ranking_result['sample'] = sample
            ranking_result['gt'] = gt
            
            results.append(ranking_result)
            
            if verbose and (idx + 1) % 50 == 0:
                print(f"进度: {idx + 1}/{len(samples_with_candidates)} "
                      f"({(idx+1)/len(samples_with_candidates)*100:.1f}%)")
        
        if output_file:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"\n✅ 排序完成，结果已保存至: {output_file}")
        
        return results


# ===== 测试代码 =====
if __name__ == '__main__':
    print("【测试修复后的诊断式评分器】\n")
    
    # 1. 测试无诊断工具的情况（简单规则）
    print("=" * 60)
    print("测试1: 简单规则评分（无诊断工具）")
    print("=" * 60)
    
    scorer_simple = DiagnosisScorer(
        cot_parser=None,
        root_cause_locator=None
    )
    
    instruction = "Turn left to the red building"
    gt = {
        'cot': "I see a RED building on the LEFT. I should turn right to face it.",
        'actions': [3, 9, 9, 0]
    }
    
    candidates = [
        {
            'cot': "I see a red building on the left. I should turn right to approach it.",
            'actions': [3, 9, 9, 0],
            'original': {'temperature': 0.7}
        },
        {
            'cot': "I see a building ahead. I should move forward.",
            'actions': [9, 9, 0],
            'original': {'temperature': 1.0}
        },
        {
            'cot': "I see a gray building on the right. I should turn left.",
            'actions': [4, 9, 0],
            'original': {'temperature': 1.5}
        }
    ]
    
    result = scorer_simple.rank_candidates(
        candidates=candidates,
        gt=gt,
        sample={'instruction': instruction},
        return_all_scores=True
    )
    
    print(f"\n结果:")
    print(f"  Chosen (idx={result['chosen_idx']}): score={result['chosen']['score']:.3f}")
    print(f"    error_type: {result['chosen']['diagnosis']['error_type']}")
    print(f"    confidence: {result['chosen']['diagnosis']['confidence']}")
    
    print(f"  Rejected (idx={result['rejected_idx']}): score={result['rejected']['score']:.3f}")
    print(f"    error_type: {result['rejected']['diagnosis']['error_type']}")
    print(f"    confidence: {result['rejected']['diagnosis']['confidence']}")
    
    print(f"\n所有分数: {[f'{s:.3f}' for s in result['scores']]}")
    
    # 2. 验证诊断信息正确传递
    print("\n" + "=" * 60)
    print("测试2: 验证诊断信息传递")
    print("=" * 60)
    
    for i, cand in enumerate(result['all_candidates']):
        print(f"\n候选 {i}:")
        print(f"  score: {cand['score']:.3f}")
        print(f"  diagnosis: {cand['diagnosis']}")
    
    print("\n✅ 所有测试完成！")