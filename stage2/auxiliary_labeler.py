"""

✅ auxiliary_labeler.py 已完成
核心功能

AuxiliaryLabeler类 - 辅助任务标注器

自动从train_1500提取关键词（60-80维）
预计算TF-IDF矩阵（快速相似度查询）
保存关键词列表供人工审核


_extract_keywords_from_dataset() - 自动关键词提取

统计1500个指令的高频词
过滤停用词
分类为：colors, directions, objects, actions, attributes, distances
每类保留高频词


_precompute_tfidf() - TF-IDF预计算

使用sklearn的TfidfVectorizer
训练1500个指令
存储矩阵供快速查询


label_keywords() - 关键词标注

输入：instruction
输出：二值向量[1,0,1,...]
标记出现的关键词


label_direction() - 方向标注

输入：instruction
输出：0=left, 1=right, 2=forward, 3=backward, -1=none
优先级匹配


label_cot_quality() - CoT质量评分（混合方案）

简单规则：关键词覆盖(30%) + 长度相似(20%) + 基础分(50%)
0.3-0.7时调用千问
取平均值


label_action_validity() - 动作合理性

TF-IDF找top-50相似指令
统计第一个action分布
计算成功率
频率<5时降低置信度


batch_label() - 批量标注

根据error_type调用对应方法
进度显示
统计千问调用次数
保存完整结果



关键特性

✅ 自动关键词提取（数据驱动）
✅ 保存关键词供审核
✅ TF-IDF预计算（快速查询）
✅ 混合评分策略（简单规则 + 千问）
✅ 完整的进度显示
✅ 详细的统计信息
✅ 按错误类型分别标注

"""

"""
辅助任务标注器 (Auxiliary Labeler)

功能：为错误样本标注辅助任务label
输入：qwen_screener筛选后的错误样本（约600个）
输出：带有辅助任务label的训练数据

四种辅助任务：
1. Keywords - 关键词识别（perception错误）
2. Direction - 方向分类（comprehension错误）
3. CoT Quality - CoT质量评分（reasoning错误）
4. Action Validity - 动作合理性（decision错误）
"""

"""
辅助任务标注器 (Auxiliary Labeler)

功能：为错误样本标注辅助任务label
输入：qwen_screener筛选后的错误样本（约600个）
输出：带有辅助任务label的训练数据

四种辅助任务：
1. Keywords - 关键词识别（perception错误）
2. Direction - 方向分类（comprehension错误）
3. CoT Quality - CoT质量评分（reasoning错误）
4. Action Validity - 动作合理性（decision错误）
"""

"""
辅助任务标注器 (Auxiliary Labeler)

功能：为错误样本标注辅助任务label
输入：qwen_screener筛选后的错误样本（约600个）
输出：带有辅助任务label的训练数据

四种辅助任务：
1. Keywords - 关键词识别（perception错误）
2. Direction - 方向分类（comprehension错误）
3. CoT Quality - CoT质量评分（reasoning错误）
4. Action Validity - 动作合理性（decision错误）
"""

import json
import os
import re
import time
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AuxiliaryLabeler:
    """
    辅助任务标注器
    
    自动为错误样本标注辅助任务label
    """
    
    def __init__(
        self,
        train_dataset_path: str,
        qwen_api_base: str = "http://localhost:8000",
        qwen_model_name: str = "Qwen3-VL-32B-Instruct",
        extract_keywords: bool = True,
        keywords_output_path: Optional[str] = None
    ):
        """
        初始化标注器
        
        Args:
            train_dataset_path: train_1500.json路径
            qwen_api_base: 千问API地址（用于CoT质量评分）
            qwen_model_name: 千问模型名称
            extract_keywords: 是否自动提取关键词
            keywords_output_path: 关键词保存路径（供审核）
        """
        print("🚀 初始化辅助任务标注器...")
        
        self.qwen_api_base = qwen_api_base
        self.qwen_model_name = qwen_model_name
        
        # 1. 加载训练数据
        print("   加载训练数据集...")
        with open(train_dataset_path, 'r') as f:
            self.train_data = json.load(f)
        print(f"   ✓ 已加载 {len(self.train_data)} 个训练样本")
        
        # 2. 提取或加载关键词
        if extract_keywords:
            print("   自动提取关键词...")
            self.keyword_dict = self._extract_keywords_from_dataset()
        else:
            # 使用默认关键词
            print("   使用默认关键词...")
            self.keyword_dict = self._get_default_keywords()
        
        # 3. 关键词索引（固定顺序）- 必须在保存之前建立
        self.keyword_index = self._build_keyword_index()
        print(f"   ✓ 关键词维度: {len(self.keyword_index)}")
        
        # 4. 保存关键词供人工审核（如果需要）
        if extract_keywords and keywords_output_path:
            self._save_keywords(keywords_output_path)
            print(f"   ✓ 关键词已保存至: {keywords_output_path}")
        
        # 5. 预计算TF-IDF
        print("   预计算TF-IDF矩阵...")
        self._precompute_tfidf()
        print(f"   ✓ TF-IDF矩阵: {self.tfidf_matrix.shape}")
        
        # 6. 统计信息
        self.stats = {
            'total_labeled': 0,
            'by_type': defaultdict(int),
            'qwen_calls': 0,
            'start_time': None,
            'end_time': None
        }
        
        print("✅ 辅助任务标注器初始化完成\n")
    
    def _get_default_keywords(self) -> Dict[str, List[str]]:
        """返回默认关键词字典"""
        return {
            'colors': ['red', 'blue', 'green', 'yellow', 'white', 'black', 
                      'gray', 'grey', 'brown', 'orange', 'purple', 'pink'],
            'directions': ['left', 'right', 'forward', 'backward', 'ahead',
                          'front', 'back', 'north', 'south', 'east', 'west'],
            'objects': ['building', 'tower', 'house', 'structure', 'wall',
                       'tree', 'road', 'path', 'area', 'zone', 'region']
        }
    
    def _extract_keywords_from_dataset(self) -> Dict[str, List[str]]:
        """
        从训练数据集中自动提取关键词
        
        Returns:
            {
                'colors': [...],
                'directions': [...],
                'objects': [...],
                'actions': [...],
                'attributes': [...]
            }
        """
        # 停用词
        stopwords = {
            'a', 'an', 'the', 'to', 'and', 'or', 'in', 'on', 'at', 'of', 'for',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
            'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that'
        }
        
        # 统计词频
        word_freq = Counter()
        
        for sample in self.train_data:
            instruction = sample.get('instruction', '')
            words = re.findall(r'\b\w+\b', instruction.lower())
            
            # 过滤停用词和数字
            words = [w for w in words if w not in stopwords and not w.isdigit()]
            word_freq.update(words)
        
        # 获取高频词
        top_words = [word for word, freq in word_freq.most_common(200)]
        
        # 分类关键词
        classified = {
            'colors': [],
            'directions': [],
            'objects': [],
            'actions': [],
            'attributes': [],
            'distances': []
        }
        
        # 预定义的分类规则
        color_words = {
            'red', 'blue', 'green', 'yellow', 'white', 'black', 'gray', 'grey',
            'brown', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'violet'
        }
        
        direction_words = {
            'left', 'right', 'forward', 'backward', 'ahead', 'front', 'back',
            'north', 'south', 'east', 'west', 'northeast', 'northwest',
            'southeast', 'southwest', 'up', 'down', 'upward', 'downward'
        }
        
        object_words = {
            'building', 'tower', 'house', 'structure', 'wall', 'tree', 'road',
            'path', 'area', 'zone', 'region', 'door', 'window', 'gate', 'fence',
            'roof', 'floor', 'ground', 'sky', 'mountain', 'hill', 'river', 'lake'
        }
        
        action_words = {
            'turn', 'move', 'go', 'fly', 'navigate', 'walk', 'run', 'approach',
            'reach', 'arrive', 'head', 'proceed', 'travel', 'advance'
        }
        
        attribute_words = {
            'large', 'small', 'big', 'little', 'tall', 'short', 'high', 'low',
            'wide', 'narrow', 'long', 'near', 'far', 'close', 'distant'
        }
        
        distance_words = {
            'near', 'far', 'close', 'distant', 'nearby', 'away', 'adjacent'
        }
        
        # 分类
        for word in top_words:
            if word in color_words:
                classified['colors'].append(word)
            elif word in direction_words:
                classified['directions'].append(word)
            elif word in object_words:
                classified['objects'].append(word)
            elif word in action_words:
                classified['actions'].append(word)
            elif word in attribute_words:
                classified['attributes'].append(word)
            elif word in distance_words:
                classified['distances'].append(word)
        
        # 确保每个类别至少有一些词
        for category, words in classified.items():
            if not words:
                # 使用默认词
                default = self._get_default_keywords()
                if category in default:
                    classified[category] = default[category]
        
        return classified
    
    def _classify_keyword(self, word: str) -> str:
        """
        分类单个关键词
        
        Args:
            word: 单词
            
        Returns:
            类别名称
        """
        for category, words in self.keyword_dict.items():
            if word in words:
                return category
        return 'unknown'
    
    def _build_keyword_index(self) -> Dict[str, int]:
        """
        构建关键词索引（固定顺序）
        
        Returns:
            {'red': 0, 'blue': 1, 'left': 12, ...}
        """
        index = {}
        idx = 0
        
        # 按类别顺序
        for category in ['colors', 'directions', 'objects', 'actions', 'attributes', 'distances']:
            if category in self.keyword_dict:
                for word in sorted(self.keyword_dict[category]):  # 排序确保顺序固定
                    index[word] = idx
                    idx += 1
        
        return index
    
    def _save_keywords(self, output_path: str):
        """保存关键词列表供审核"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'keyword_dict': self.keyword_dict,
                'keyword_index': self.keyword_index,
                'total_dimensions': len(self.keyword_index)
            }, f, indent=2, ensure_ascii=False)
    
    def _precompute_tfidf(self):
        """预计算TF-IDF矩阵"""
        # 提取所有指令
        all_instructions = []
        for sample in self.train_data:
            instruction = sample.get('instruction', '')
            if not instruction:
                instruction = sample.get('task', '')  # 尝试其他字段
            if not instruction:
                instruction = "empty instruction"  # 防止完全为空
            all_instructions.append(instruction)
        
        # 过滤空字符串
        all_instructions = [inst if inst.strip() else "empty" for inst in all_instructions]
        
        # 训练TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # 最多500个特征
            stop_words='english',
            lowercase=True,
            min_df=1,  # 至少出现1次即可
            max_df=0.95  # 最多出现在95%的文档中
        )
        
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(all_instructions)
            # shape: (1500, vocab_size)
        except ValueError as e:
            # 如果还是失败，使用更宽松的设置
            print(f"   ⚠️  TF-IDF训练失败: {e}")
            print("   尝试使用更宽松的设置...")
            
            self.vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words=None,  # 不使用停用词
                lowercase=True,
                min_df=1
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(all_instructions)
    
    def label_keywords(self, instruction: str) -> List[int]:
        """
        标注关键词向量
        
        Args:
            instruction: 指令文本
            
        Returns:
            keywords_vector: [1, 0, 1, 0, ...]（二值向量）
        """
        # 初始化向量（全0）
        vector = [0] * len(self.keyword_index)
        
        # 提取指令中的词
        words = re.findall(r'\b\w+\b', instruction.lower())
        
        # 标记出现的关键词
        for word in words:
            if word in self.keyword_index:
                idx = self.keyword_index[word]
                vector[idx] = 1
        
        return vector
    
    def label_direction(self, instruction: str) -> int:
        """
        标注主要方向
        
        Args:
            instruction: 指令文本
            
        Returns:
            direction: 0=left, 1=right, 2=forward, 3=backward, -1=无方向
        """
        instruction_lower = instruction.lower()
        
        # 方向映射
        direction_map = {
            'left': 0,
            'right': 1,
            'forward': 2,
            'ahead': 2,
            'front': 2,
            'backward': 3,
            'back': 3
        }
        
        # 按优先级查找
        priority_order = ['left', 'right', 'forward', 'ahead', 'backward', 'back']
        
        for direction_word in priority_order:
            pattern = r'\b' + re.escape(direction_word) + r'\b'
            if re.search(pattern, instruction_lower):
                return direction_map[direction_word]
        
        return -1  # 无明确方向
    
    def _simple_cot_score(
        self, 
        model_cot: str, 
        gt_cot: str, 
        instruction: str
    ) -> float:
        """
        简单规则评估CoT质量
        
        Args:
            model_cot: 模型的CoT
            gt_cot: Ground-Truth的CoT
            instruction: 指令
            
        Returns:
            score: 0.0-1.0
        """
        # 1. 关键词覆盖度
        # 从指令中提取关键词
        inst_words = set(re.findall(r'\b\w+\b', instruction.lower()))
        model_words = set(re.findall(r'\b\w+\b', model_cot.lower()))
        gt_words = set(re.findall(r'\b\w+\b', gt_cot.lower()))
        
        # GT包含的关键指令词
        gt_keywords = inst_words & gt_words
        
        if len(gt_keywords) == 0:
            coverage = 0.5  # 无法判断
        else:
            # model覆盖了多少GT的关键词
            model_coverage = len(model_words & gt_keywords) / len(gt_keywords)
            coverage = model_coverage
        
        # 2. 长度相似度
        if len(model_cot) == 0 or len(gt_cot) == 0:
            length_sim = 0.0
        else:
            length_ratio = min(len(model_cot), len(gt_cot)) / max(len(model_cot), len(gt_cot))
            length_sim = length_ratio
        
        # 3. 综合评分
        # 基础分0.5，关键词覆盖占30%，长度占20%
        score = 0.5 + 0.3 * coverage + 0.2 * length_sim
        
        # 限制在[0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _qwen_cot_score(
        self, 
        model_cot: str, 
        gt_cot: str, 
        instruction: str
    ) -> Optional[float]:
        """
        使用千问评估CoT质量
        
        Args:
            model_cot: 模型的CoT
            gt_cot: Ground-Truth的CoT
            instruction: 指令
            
        Returns:
            score: 0.0-1.0，失败返回None
        """
        prompt = f"""评估以下CoT（Chain of Thought）的质量，给出0到1之间的分数。

【指令】
{instruction}

【正确的CoT】
{gt_cot}

【待评估的CoT】
{model_cot}

【评分标准】
- 1.0: 完美，与正确CoT质量相当
- 0.7-0.9: 很好，主要推理正确
- 0.5-0.7: 一般，部分正确
- 0.3-0.5: 较差，有明显错误
- 0.0-0.3: 很差，完全错误

请只输出一个0到1之间的数字，不要有其他文字。"""
        
        try:
            response = requests.post(
                f"{self.qwen_api_base}/v1/chat/completions",
                json={
                    "model": self.qwen_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 50
                },
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content'].strip()
                
                # 提取数字
                match = re.search(r'0?\.\d+|[01]\.?\d*', content)
                if match:
                    score = float(match.group())
                    return max(0.0, min(1.0, score))
            
            return None
            
        except Exception as e:
            print(f"      ⚠️  千问评分失败: {e}")
            return None
    
    def label_cot_quality(
        self, 
        model_cot: str, 
        gt_cot: str, 
        instruction: str
    ) -> float:
        """
        标注CoT质量（混合方案）
        
        策略：
        1. 先用简单规则评分
        2. 如果分数在0.3-0.7（不确定），调用千问
        3. 取平均值
        
        Args:
            model_cot: 模型的CoT
            gt_cot: Ground-Truth的CoT
            instruction: 指令
            
        Returns:
            quality_score: 0.0-1.0
        """
        # 简单评分
        simple_score = self._simple_cot_score(model_cot, gt_cot, instruction)
        
        # 如果明确好或明确差，直接返回
        if simple_score < 0.3 or simple_score > 0.7:
            return simple_score
        
        # 不确定区间，调用千问
        qwen_score = self._qwen_cot_score(model_cot, gt_cot, instruction)
        self.stats['qwen_calls'] += 1
        
        if qwen_score is not None:
            # 取平均
            final_score = (simple_score + qwen_score) / 2
            return final_score
        else:
            # 千问失败，使用简单评分
            return simple_score
    
    def find_similar_instructions(
        self, 
        target_instruction: str, 
        top_k: int = 50
    ) -> List[Dict]:
        """
        使用TF-IDF查找相似指令
        
        Args:
            target_instruction: 目标指令
            top_k: 返回top-k个最相似的
            
        Returns:
            相似样本列表
        """
        # 转换为TF-IDF向量
        target_vec = self.vectorizer.transform([target_instruction])
        
        # 计算与所有训练样本的相似度
        similarities = cosine_similarity(target_vec, self.tfidf_matrix)[0]
        
        # 找top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_samples = [self.train_data[i] for i in top_indices]
        
        return similar_samples
    
    def label_action_validity(
        self, 
        actions: List[int], 
        instruction: str
    ) -> float:
        """
        标注动作合理性（基于统计）
        
        策略：
        1. 在train_1500中找top-50相似指令
        2. 统计这些样本的第一个action分布
        3. 计算目标action的"成功率"
        
        Args:
            actions: 动作序列
            instruction: 指令
            
        Returns:
            validity: 0.0-1.0
        """
        if not actions:
            return 0.0
        
        target_action = actions[0]  # 第一个action最重要
        
        # 找相似样本
        similar_samples = self.find_similar_instructions(instruction, top_k=50)
        
        # 统计第一个action的分布
        action_counts = defaultdict(int)
        
        for sample in similar_samples:
            sample_actions = sample.get('action', [])
            if sample_actions:
                first_action = sample_actions[0]
                action_counts[first_action] += 1
        
        # 计算validity
        total_samples = len(similar_samples)
        target_count = action_counts.get(target_action, 0)
        
        if target_count == 0:
            # 这个action从未在相似场景出现
            return 0.1
        
        # 基础validity = 出现频率
        validity = target_count / total_samples
        
        # 频率调整：如果样本数太少，降低置信度
        if target_count < 5:
            validity *= 0.7
        
        return validity
    
    def batch_label(
        self,
        screened_samples: List[Dict],
        output_file: str,
        verbose: bool = True
    ) -> Dict:
        """
        批量标注所有错误样本
        
        Args:
            screened_samples: 从qwen_screener来的样本
            output_file: 输出文件路径
            verbose: 是否打印详细信息
            
        Returns:
            {
                'total_samples': int,
                'labeled_by_type': dict,
                'samples': List[Dict]
            }
        """
        print(f"\n{'='*60}")
        print("辅助任务批量标注开始")
        print(f"{'='*60}")
        print(f"待标注样本数: {len(screened_samples)}")
        
        self.stats['start_time'] = datetime.now()
        labeled_samples = []
        
        for idx, sample in enumerate(screened_samples):
            sample_id = sample.get('sample_id', idx)
            qwen_result = sample.get('qwen_screening', sample)  # 兼容不同格式
            error_type = qwen_result.get('error_type', 'unknown')
            
            # 提取信息
            sample_data = sample.get('sample', {})
            model_output = sample.get('model_output', {})
            gt = sample.get('gt', {})
            
            instruction = sample_data.get('instruction', '')
            model_cot = model_output.get('cot', '')
            model_actions = model_output.get('actions', [])
            gt_cot = gt.get('cot', '')
            
            # 根据错误类型标注对应的辅助任务
            aux_labels = {}
            
            if error_type == 'perception':
                # 标注关键词
                aux_labels['keywords'] = self.label_keywords(instruction)
                self.stats['by_type']['perception'] += 1
                
            elif error_type == 'comprehension':
                # 标注方向
                aux_labels['direction'] = self.label_direction(instruction)
                self.stats['by_type']['comprehension'] += 1
                
            elif error_type == 'reasoning':
                # 标注CoT质量
                aux_labels['cot_quality'] = self.label_cot_quality(
                    model_cot, gt_cot, instruction
                )
                self.stats['by_type']['reasoning'] += 1
                
            elif error_type == 'decision':
                # 标注动作合理性
                aux_labels['action_validity'] = self.label_action_validity(
                    model_actions, instruction
                )
                self.stats['by_type']['decision'] += 1
            
            # 添加aux_labels到sample
            labeled_sample = sample.copy()
            labeled_sample['aux_labels'] = aux_labels
            labeled_samples.append(labeled_sample)
            
            self.stats['total_labeled'] += 1
            
            # 进度显示
            if verbose and (idx + 1) % 50 == 0:
                elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                avg_time = elapsed / (idx + 1)
                eta = avg_time * (len(screened_samples) - idx - 1)
                
                print(f"进度: {idx + 1}/{len(screened_samples)} "
                      f"({(idx+1)/len(screened_samples)*100:.1f}%) | "
                      f"千问调用: {self.stats['qwen_calls']} | "
                      f"预计剩余: {eta/60:.1f}分钟")
        
        self.stats['end_time'] = datetime.now()
        
        # 保存结果
        output_data = {
            'total_samples': len(screened_samples),
            'labeled_by_type': dict(self.stats['by_type']),
            'samples': labeled_samples,
            'stats': {
                'total_labeled': self.stats['total_labeled'],
                'qwen_calls': self.stats['qwen_calls'],
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': self.stats['end_time'].isoformat()
            }
        }
        
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # 统计信息
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print(f"\n{'='*60}")
        print("标注完成！")
        print(f"{'='*60}")
        print(f"总样本数: {output_data['total_samples']}")
        print(f"标注分布:")
        for error_type, count in output_data['labeled_by_type'].items():
            print(f"  {error_type}: {count}")
        print(f"千问调用次数: {self.stats['qwen_calls']}")
        print(f"总耗时: {int(duration//60)}分{int(duration%60)}秒")
        print(f"输出文件: {output_file}")
        
        return output_data


# 使用示例
if __name__ == '__main__':
    # 测试关键词提取
    print("【测试关键词提取】")
    
    # 创建测试数据
    test_train_data = [
        {"instruction": "Turn left to the red building"},
        {"instruction": "Move forward to the blue tower"},
        {"instruction": "Go right near the green house"}
    ]
    
    # 保存测试数据
    with open('/tmp/test_train.json', 'w') as f:
        json.dump(test_train_data, f)
    
    # 初始化标注器
    labeler = AuxiliaryLabeler(
        train_dataset_path='/tmp/test_train.json',
        extract_keywords=True,
        keywords_output_path='/tmp/test_keywords.json'
    )
    
    # 测试各项功能
    print("\n【测试关键词标注】")
    test_inst = "Turn left to the red building"
    keywords_vec = labeler.label_keywords(test_inst)
    print(f"指令: {test_inst}")
    print(f"关键词向量维度: {len(keywords_vec)}")
    print(f"激活的关键词: {sum(keywords_vec)}")
    
    print("\n【测试方向标注】")
    direction = labeler.label_direction(test_inst)
    direction_names = {0: 'left', 1: 'right', 2: 'forward', 3: 'backward', -1: 'none'}
    print(f"方向: {direction_names.get(direction)}")
    
    print("\n【测试CoT质量评分】")
    model_cot = "I see a building ahead"
    gt_cot = "I see a red building on the left"
    score = labeler._simple_cot_score(model_cot, gt_cot, test_inst)
    print(f"Model CoT: {model_cot}")
    print(f"GT CoT: {gt_cot}")
    print(f"质量分数: {score:.3f}")
    
    print("\n【测试相似指令查找】")
    similar = labeler.find_similar_instructions(test_inst, top_k=2)
    print(f"找到 {len(similar)} 个相似指令")
    
    print("\n✅ 所有功能测试完成！")