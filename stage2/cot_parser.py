"""
CoT分解器 (CoT Parser)

功能：将CogFly生成的CoT推理链分解为细粒度的步骤
创新点：首次在VLA中做推理链的细粒度分解，支持逐步验证

输入示例：
    "<thinking>我看到前方有红色建筑。指令要求飞向它。应该向前飞。</thinking><action>9</action>"

输出示例：
    [
        {'step_id': 0, 'type': 'perception', 'content': '我看到前方有红色建筑'},
        {'step_id': 1, 'type': 'comprehension', 'content': '指令要求飞向它'},
        {'step_id': 2, 'type': 'reasoning', 'content': '应该向前飞'}
    ]
"""

import re
import json
from typing import List, Dict, Optional
import os  # 第6行附近添加

class CoTParser:
    """
    CoT推理链解析器
    
    将完整的CoT文本分解为：
    - 感知步骤 (perception)
    - 理解步骤 (comprehension) 
    - 推理步骤 (reasoning)
    - 决策步骤 (decision)
    """
    
    def __init__(self):
        """初始化分类规则"""

                # 感知关键词（观察环境）
        self.perception_keywords = [
            'see', 'observe', 'notice', 'detect', 'identify', 'spot',
            'view', 'visible', 'ahead', 'front', 'left', 'right', 'around',
            'building', 'obstacle', 'landmark', 'target', 'object', 'environment',
            'current location', 'position', 'scene'
        ]

        # 理解关键词（解释指令）
        self.comprehension_keywords = [
            'instruction', 'task', 'goal', 'objective', 'target', 'destination',
            'required', 'need to', 'should reach', 'navigate to', 'fly to',
            'towards', 'direction', 'path'
        ]

        # 推理关键词（决策逻辑）
        self.reasoning_keywords = [
            'should', 'need', 'therefore', 'thus', 'so', 'judge', 'infer',
            'can', 'must', 'best', 'suitable', 'reasonable', 'plan',
            'analyze', 'consider', 'evaluate', 'choose', 'decide'
        ]

        # 决策关键词（最终动作）
        self.decision_keywords = [
            'decide', 'choose', 'execute', 'take', 'perform',
            'move forward', 'turn left', 'turn right', 'ascend', 'descend', 'stop',
            'action', 'movement'
        ]
    
    def parse(self, cot_text: str, verbose: bool = False) -> List[Dict]:
        """
        解析完整CoT文本
        
        Args:
            cot_text: CoT文本（包含<thinking>标签）
            verbose: 是否打印详细信息
            
        Returns:
            步骤列表，每个步骤包含：
            - step_id: 步骤ID
            - type: 步骤类型
            - content: 步骤内容
            - raw_text: 原始文本
        """
        # 1. 提取<thinking>内容
        thinking_content = self._extract_thinking_content(cot_text)
        
        if not thinking_content:
            if verbose:
                print("⚠️ 未找到<thinking>标签内容")
            return []
        
        # 2. 分句
        sentences = self._split_into_sentences(thinking_content)
        
        if verbose:
            print(f"📝 分解为 {len(sentences)} 个句子")
        
        # 3. 分类每个句子
        steps = []
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            step_type = self._classify_step(sentence)
            
            step = {
                'step_id': idx,
                'type': step_type,
                'content': sentence.strip(),
                'raw_text': sentence.strip()
            }
            steps.append(step)
            
            if verbose:
                print(f"  [{idx}] {step_type:15s} | {sentence[:60]}...")
        
        # 4. 后处理（合并连续相同类型、去除空步骤等）
        steps = self._post_process(steps)
        
        return steps
    
    def _extract_thinking_content(self, cot_text: str) -> str:
        """
        提取<thinking>标签内的内容
        
        Args:
            cot_text: 完整CoT文本
            
        Returns:
            <thinking>内的文本，如果没有则返回原文
        """
        # 尝试匹配<thinking>...</thinking>
        match = re.search(r'<thinking>\s*(.*?)\s*</thinking>', cot_text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # 如果没有标签，尝试提取<action>之前的内容
        action_match = re.search(r'(.*?)<action>', cot_text, re.DOTALL)
        if action_match:
            return action_match.group(1).strip()
        
        # 都没有，返回原文
        return cot_text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分句
        
        规则：
        - 按句号、问号、感叹号分割
        - 按换行符分割
        - 保留语义完整性
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 先按换行符分割
        lines = text.split('\n')
        
        sentences = []
        for line in lines:
            # 再按标点符号分割
            # 支持中英文标点
            parts = re.split(r'[。！？.!?]+', line)
            
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    sentences.append(cleaned)
        
        return sentences
    
    def _classify_step(self, sentence: str) -> str:
        """
        分类单个句子
        
        优先级：perception > comprehension > reasoning > decision
        （因为感知最基础，决策最高层）
        
        Args:
            sentence: 输入句子
            
        Returns:
            步骤类型：'perception' | 'comprehension' | 'reasoning' | 'decision' | 'unknown'
        """
        sentence_lower = sentence.lower()
        
        # 记录每种类型的匹配分数
        scores = {
            'perception': 0,
            'comprehension': 0,
            'reasoning': 0,
            'decision': 0
        }
        
        # 计算感知分数
        for keyword in self.perception_keywords:
            if keyword in sentence_lower:
                scores['perception'] += 1
        
        # 计算理解分数
        for keyword in self.comprehension_keywords:
            if keyword in sentence_lower:
                scores['comprehension'] += 1
        
        # 计算推理分数
        for keyword in self.reasoning_keywords:
            if keyword in sentence_lower:
                scores['reasoning'] += 1
        
        # 计算决策分数
        for keyword in self.decision_keywords:
            if keyword in sentence_lower:
                scores['decision'] += 1
        
        # 找最高分
        max_score = max(scores.values())
        
        if max_score == 0:
            # 没有匹配到任何关键词
            return 'unknown'
        
        # 返回最高分的类型（如果有并列，按优先级返回）
        priority = ['perception', 'comprehension', 'reasoning', 'decision']
        for step_type in priority:
            if scores[step_type] == max_score:
                return step_type
        
        return 'unknown'
    
    def _post_process(self, steps: List[Dict]) -> List[Dict]:
        """
        后处理步骤列表
        
        - 重新编号step_id
        - 移除unknown类型（可选）
        
        Args:
            steps: 原始步骤列表
            
        Returns:
            处理后的步骤列表
        """
        # 过滤掉unknown（可选，这里保留）
        # steps = [s for s in steps if s['type'] != 'unknown']
        
        # 重新编号
        for idx, step in enumerate(steps):
            step['step_id'] = idx
        
        return steps
    
    def parse_batch(self, cot_list: List[str], verbose: bool = False) -> List[List[Dict]]:
        """
        批量解析多个CoT
        
        Args:
            cot_list: CoT文本列表
            verbose: 是否打印详细信息
            
        Returns:
            步骤列表的列表
        """
        results = []
        
        for idx, cot_text in enumerate(cot_list):
            if verbose:
                print(f"\n{'='*60}")
                print(f"解析 CoT #{idx+1}")
                print('='*60)
            
            steps = self.parse(cot_text, verbose=verbose)
            results.append(steps)
        
        return results
    
    
    def export_to_json(self, steps: List[Dict], output_file: str):
        """
        导出步骤到JSON文件
        
        Args:
            steps: 步骤列表
            output_file: 输出文件路径（自动创建目录）
        """
        # 自动创建输出目录
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(steps, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已导出到: {output_file}")


def test_parser():
    """测试函数"""
    print("🧪 测试CoT分解器\n")
    
    # 测试案例1：标准格式
    cot1 = """
    <thinking>
    我看到前方有一座红色建筑。
    指令要求飞向红色建筑。
    这座建筑就是目标。
    因此应该向前飞行。
    </thinking>
    <action>9</action>
    """
    
    # 测试案例2：复杂格式
    cot2 = """
    <thinking>
    观察周围环境，前方视野中出现大型灰色建筑，左侧有树木遮挡。
    任务目标是到达灰色建筑左侧。
    分析当前位置与目标的相对关系，判断需要先向前接近，再向左转。
    决定采取向前加速动作。
    </thinking>
    <action>8</action>
    """
    
    # 测试案例3：无标签格式
    cot3 = "看到目标建筑在右前方。应该右转然后前进。<action>3</action>"
    
    parser = CoTParser()
    
    # 测试1
    print("="*60)
    print("测试案例1: 标准格式")
    print("="*60)
    steps1 = parser.parse(cot1, verbose=True)
    print(f"\n✅ 共分解 {len(steps1)} 个步骤\n")
    
    # 测试2
    print("="*60)
    print("测试案例2: 复杂格式")
    print("="*60)
    steps2 = parser.parse(cot2, verbose=True)
    print(f"\n✅ 共分解 {len(steps2)} 个步骤\n")
    
    # 测试3
    print("="*60)
    print("测试案例3: 无标签格式")
    print("="*60)
    steps3 = parser.parse(cot3, verbose=True)
    print(f"\n✅ 共分解 {len(steps3)} 个步骤\n")
    
    # 导出示例
    parser.export_to_json(steps1, '/home/claude/diagnosis_module/test_cot_steps.json')


if __name__ == '__main__':
    test_parser()




"""

这个没什么说的这个比较简单  
"""