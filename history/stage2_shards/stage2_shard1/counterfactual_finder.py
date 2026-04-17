"""
反事实发现器 (Counterfactual Finder) - 重写版

职责：从250K训练集里找相似成功样本，统计动作分布，
     辅助 RootCauseLocator 判断 error_type。

核心输出（root_cause_locator.py 真正使用的字段）：
    statistics = {
        'is_action_unusual': bool,
        'failure_first_action': int,
        'most_common_first_action': int,
        'total_similar_samples': int,
        'success_rate': float,
        'first_action_distribution': dict,
    }
"""

import json
import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

TRAIN_DATA_PATH = (
    "/home/ubuntu/data1/zx/1OpenFly-Platform/"
    "OpenFly-Platform/dataset/Annotation/train.json"
)
INDICES_FILE = "/home/ubuntu/data1/lyy/full_rlds_project-3/stage2/cf_indices.pkl"


class CounterfactualFinder:
    """
    反事实样本发现器

    主要功能：
    1. _compute_scene_statistics()  ← 核心，驱动诊断结论
    2. _find_similar_success_samples()  ← 辅助，找相似成功样本
    3. find_counterfactuals()  ← 对外接口，返回统一格式
    """

    def __init__(
        self,
        train_data_path: str = TRAIN_DATA_PATH,
        indices_file: str = INDICES_FILE,
        verbose: bool = False
    ):
        self.verbose = verbose

        print(f"📂 加载训练数据: {train_data_path}")
        print("⏳ 约需 30-60 秒...")
        with open(train_data_path, 'r') as f:
            self.train_data = json.load(f)
        print(f"✅ 加载完成，共 {len(self.train_data)} 个样本")

        # 构建或加载索引
        if not self._load_indices(indices_file):
            self._build_indices()
            self._save_indices(indices_file)

    # ==================== 索引 ====================

    def _build_indices(self):
        print("🔨 构建索引...")
        self.env_index = defaultdict(list)     # env_name → [idx, ...]
        self.spatial_index = defaultdict(list) # (env, gx, gy) → [idx, ...]
        GRID = 50

        for idx, sample in enumerate(self.train_data):
            image_path = sample.get('image_path', '')
            env = image_path.split('/')[0] if '/' in image_path else 'unknown'
            self.env_index[env].append(idx)

            pos_list = sample.get('pos', [])
            if pos_list and len(pos_list[0]) >= 2:
                gx = int(pos_list[0][0] / GRID)
                gy = int(pos_list[0][1] / GRID)
                self.spatial_index[(env, gx, gy)].append(idx)

        print(f"✅ 索引完成: {len(self.env_index)} 个环境, "
              f"{len(self.spatial_index)} 个空间网格")

    def _save_indices(self, filepath: str):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'env_index': dict(self.env_index),
                    'spatial_index': dict(self.spatial_index),
                    'num_samples': len(self.train_data)
                }, f)
            print(f"💾 索引已保存: {filepath}")
        except Exception as e:
            print(f"⚠️  索引保存失败（不影响运行）: {e}")

    def _load_indices(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if data.get('num_samples') != len(self.train_data):
                print("⚠️  索引与数据不匹配，重新构建")
                return False
            self.env_index = defaultdict(list, data['env_index'])
            self.spatial_index = defaultdict(list, data['spatial_index'])
            print(f"✅ 索引加载成功: {len(self.env_index)} 个环境")
            return True
        except Exception as e:
            print(f"⚠️  索引加载失败: {e}")
            return False

    # ==================== 工具函数 ====================

    def _get_env_name(self, sample: Dict) -> str:
        image_path = sample.get('image_path', '')
        return image_path.split('/')[0] if '/' in image_path else 'unknown'

    def _pos_dist(self, p1: List[float], p2: List[float]) -> float:
        if not p1 or not p2 or len(p1) < 2 or len(p2) < 2:
            return float('inf')
        return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(p1[:3], p2[:3]))))

    def _inst_sim(self, a: str, b: str) -> float:
        """Jaccard 词袋相似度"""
        w1 = set(a.lower().split())
        w2 = set(b.lower().split())
        union = len(w1 | w2)
        return len(w1 & w2) / union if union > 0 else 0.0

    def _is_successful(self, sample: Dict) -> bool:
        """判断训练样本是否成功"""
        positions = sample.get('pos', [])
        actions = sample.get('action', [])
        if len(positions) < 2 or not actions:
            return False
        dist = self._pos_dist(positions[0], positions[-1])
        if dist < 10:
            return False
        if all(a == 0 for a in actions):
            return False
        return True

    def _get_candidate_pool(self, failure_case: Dict) -> List[int]:
        """用空间索引获取候选池，降级到环境索引"""
        env = failure_case.get('env_name', '')
        start_pos = failure_case.get('start_pos', [])
        GRID = 50

        if len(start_pos) >= 2:
            gx = int(start_pos[0] / GRID)
            gy = int(start_pos[1] / GRID)
            candidates = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    candidates.extend(
                        self.spatial_index.get((env, gx + dx, gy + dy), []))
            if candidates:
                return candidates

        # 降级：整个环境
        return self.env_index.get(env, [])

    # ==================== 核心：统计分析 ====================

    def _compute_scene_statistics(self, failure_case: Dict) -> Dict:
        """
        统计分析（核心方法）

        找同环境、指令相似的样本，统计第一个动作分布，
        判断失败案例的动作是否异常。

        Returns:
            {
                'is_action_unusual': bool,
                'failure_first_action': int,
                'most_common_first_action': int,
                'total_similar_samples': int,
                'success_samples': int,
                'success_rate': float,
                'first_action_distribution': {action_id: count},
            }
        """
        fail_inst = failure_case.get('instruction', '')
        fail_actions = failure_case.get('trajectory', {}).get('actions', [])
        fail_first_action = fail_actions[0] if fail_actions else -1

        candidate_pool = self._get_candidate_pool(failure_case)

        similar_samples = []
        for idx in candidate_pool:
            sample = self.train_data[idx]
            cand_inst = sample.get('gpt_instruction', '')
            if self._inst_sim(fail_inst, cand_inst) >= 0.15:
                similar_samples.append(sample)

        if not similar_samples:
            return {
                'is_action_unusual': False,
                'failure_first_action': fail_first_action,
                'most_common_first_action': fail_first_action,
                'total_similar_samples': 0,
                'success_samples': 0,
                'success_rate': 0.0,
                'first_action_distribution': {},
            }

        # 统计动作分布
        action_counts = defaultdict(int)
        success_count = 0
        for sample in similar_samples:
            actions = sample.get('action', [])
            if actions:
                action_counts[actions[0]] += 1
            if self._is_successful(sample):
                success_count += 1

        most_common = max(action_counts, key=action_counts.get) \
            if action_counts else fail_first_action

        return {
            'is_action_unusual': (fail_first_action != most_common
                                  and most_common != -1),
            'failure_first_action': fail_first_action,
            'most_common_first_action': most_common,
            'total_similar_samples': len(similar_samples),
            'success_samples': success_count,
            'success_rate': success_count / len(similar_samples),
            'first_action_distribution': dict(action_counts),
        }

    # ==================== 反事实样本检索 ====================

    def _find_similar_success_samples(
        self,
        failure_case: Dict,
        top_k: int = 5
    ) -> List[Dict]:
        """
        找相似且成功的样本（用于佐证诊断结论）

        Returns: List of {sample, sample_idx, similarity_score, differences}
        """
        fail_inst = failure_case.get('instruction', '')
        fail_start = failure_case.get('start_pos', [])
        fail_actions = failure_case.get('trajectory', {}).get('actions', [])
        fail_first = fail_actions[0] if fail_actions else -1

        candidate_pool = self._get_candidate_pool(failure_case)
        results = []

        for idx in candidate_pool:
            sample = self.train_data[idx]
            if not self._is_successful(sample):
                continue

            cand_inst = sample.get('gpt_instruction', '')
            inst_sim = self._inst_sim(fail_inst, cand_inst)
            if inst_sim < 0.15:
                continue

            cand_pos = sample.get('pos', [[]])[0] if sample.get('pos') else []
            pos_dist = self._pos_dist(fail_start, cand_pos)
            pos_sim = max(0.0, 1.0 - pos_dist / 50.0)

            score = inst_sim * 0.6 + pos_sim * 0.4

            cand_actions = sample.get('action', [])
            cand_first = cand_actions[0] if cand_actions else -1

            results.append({
                'sample': sample,
                'sample_idx': idx,
                'similarity_score': score,
                'differences': {
                    'position_diff': pos_dist,
                    'inst_similarity': inst_sim,
                    'fail_first_action': fail_first,
                    'cand_first_action': cand_first,
                    'action_differs': (fail_first != cand_first),
                    'fail_steps': len(fail_actions),
                    'cand_steps': len(cand_actions),
                }
            })

        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]

    # ==================== 对外接口 ====================

    def find_counterfactuals(
        self,
        failure_case: Dict,
        top_k: int = 5,
        verbose: bool = False
    ) -> Dict:
        """
        主接口，供 RootCauseLocator 调用

        Args:
            failure_case: {
                'env_name': str,
                'start_pos': [x, y, z],
                'instruction': str,
                'trajectory': {'actions': [...]}
            }
            top_k: 返回的反事实样本数量
            verbose: 是否打印过程

        Returns:
            {
                'counterfactuals': [...],
                'statistics': {...},
                'type_counts': {...}
            }
        """
        if verbose:
            print(f"🔍 查找反事实 (env={failure_case.get('env_name', '?')})")

        statistics = self._compute_scene_statistics(failure_case)
        counterfactuals = self._find_similar_success_samples(failure_case, top_k)

        if verbose:
            print(f"   相似样本: {statistics['total_similar_samples']}")
            print(f"   成功率: {statistics['success_rate']:.1%}")
            print(f"   失败动作: {statistics['failure_first_action']} | "
                  f"最常见: {statistics['most_common_first_action']} | "
                  f"异常: {statistics['is_action_unusual']}")
            print(f"   反事实样本: {len(counterfactuals)}")

        return {
            'counterfactuals': counterfactuals,
            'statistics': statistics,
            'type_counts': {
                'scene': len(counterfactuals),
                'action': 0,
                'instruction': 0,
            }
        }

    def find_batch(
        self,
        failures: List[Dict],
        output_file: str,
        top_k: int = 5,
        verbose: bool = True
    ) -> List[Dict]:
        """批量查找，结果写入文件"""
        print(f"🚀 批量查找 ({len(failures)} 个失败案例)")
        results = []

        for i, failure in enumerate(failures):
            if verbose and i % 50 == 0:
                print(f"  进度: {i}/{len(failures)}")
            result = self.find_counterfactuals(failure, top_k=top_k)
            results.append({
                'failure_case': failure,
                'counterfactuals': result['counterfactuals'],
                'statistics': result['statistics'],
                'type_counts': result['type_counts'],
                'num_found': len(result['counterfactuals'])
            })

        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        avg = sum(r['num_found'] for r in results) / len(results) if results else 0
        print(f"✅ 完成，平均每个案例找到 {avg:.1f} 个反事实 → {output_file}")
        return results
