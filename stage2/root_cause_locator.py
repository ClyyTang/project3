"""
根因定位算法 (Root Cause Locator)

功能：综合CoT分解 + 反事实验证 + 统计分析，定位失败根因
创新点：首次在VLA中实现推理链的逐步诊断

核心思想：
    1. 分解CoT为步骤
    2. 从前往后逐步验证
    3. 找到最早出错的步骤 = 根因
    4. 结合反事实和统计分析确认
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from cot_parser import CoTParser
from counterfactual_finder import CounterfactualFinder


class RootCauseLocator:
    """
    根因定位器
    
    综合多种方法定位失败的根本原因
    """
    
    def __init__(
        self,
        train_data_path: str,
        verbose: bool = False
    ):
        """
        初始化
        
        Args:
            train_data_path: 训练数据路径
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
        
        # 初始化子模块
        print("🔨 初始化根因定位器...")
        
        self.cot_parser = CoTParser()
        print("  ✅ CoT分解器就绪")
        
        self.cf_finder = CounterfactualFinder(train_data_path)
        print("  ✅ 反事实发现器就绪")
        
        print("✅ 根因定位器初始化完成\n")
    
    def locate_root_cause(self, failure_case: Dict) -> Dict:
        """
        定位失败案例的根因
        
        核心算法：
        1. CoT分解为步骤
        2. 逐步验证（从前往后）
        3. 找反事实对比
        4. 统计分析辅助
        5. 确定根因
        
        Args:
            failure_case: 失败案例
            
        Returns:
            诊断报告
        """
        sample_id = failure_case.get('sample_id', 'unknown')
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"诊断样本 #{sample_id}")
            print('='*60)
        
        # 1. 提取CoT
        cot_list = failure_case.get('trajectory', {}).get('cot_list', [])
        
        if not cot_list:
            return {
                'sample_id': sample_id,
                'error': 'No CoT available',
                'diagnosis': None
            }
        
        # 合并所有CoT（如果是多步的）
        full_cot = "\n".join(cot_list)
        
        # 2. CoT分解
        steps = self.cot_parser.parse(full_cot, verbose=self.verbose)
        
        if self.verbose:
            print(f"\n📋 CoT分解为 {len(steps)} 个步骤")
        
        if len(steps) == 0:
            # 降级方案：无法分解，用统计分析
            return self._statistical_diagnosis(failure_case)
        
        # 3. 逐步验证
        for step_idx, step in enumerate(steps):
            if self.verbose:
                print(f"\n🔍 验证步骤 {step_idx}: {step['type']}")
                print(f"   内容: {step['content'][:60]}...")
            
            # 验证这一步
            is_error, evidence = self._validate_step(
                step, 
                failure_case, 
                step_idx
            )
            
            if is_error:
                # 找到根因！
                if self.verbose:
                    print(f"❌ 发现错误！")
                
                # 详细诊断
                diagnosis = self._diagnose_step_error(
                    step,
                    step_idx,
                    failure_case,
                    evidence
                )
                
                return {
                    'sample_id': sample_id,
                    'error_step': step_idx,
                    'error_type': step['type'],
                    'root_cause': diagnosis['cause'],
                    'confidence': diagnosis['confidence'],
                    'evidence': evidence,
                    'all_steps': steps
                }
        
        # 4. 如果逐步验证没找到，用统计分析
        if self.verbose:
            print(f"\n⚠️  逐步验证未找到明确错误，使用统计分析")
        
        return self._statistical_diagnosis(failure_case, steps)
    
    def _validate_step(
        self, 
        step: Dict, 
        failure_case: Dict,
        step_idx: int
    ) -> Tuple[bool, Dict]:
        """
        验证单个步骤是否错误
        
        方法：
        1. 根据步骤类型选择验证方式
        2. 结合反事实和统计分析
        3. 返回是否错误 + 证据
        
        Args:
            step: CoT步骤
            failure_case: 失败案例
            step_idx: 步骤索引
            
        Returns:
            (is_error, evidence)
        """
        step_type = step['type']
        evidence = {
            'type': step_type,
            'step_idx': step_idx,
            'validation_method': []
        }
        
        # 根据步骤类型选择验证方法
        if step_type == 'perception':
            return self._validate_perception(step, failure_case, evidence)
        
        elif step_type == 'comprehension':
            return self._validate_comprehension(step, failure_case, evidence)
        
        elif step_type == 'reasoning':
            return self._validate_reasoning(step, failure_case, evidence)
        
        elif step_type == 'decision':
            return self._validate_decision(step, failure_case, evidence)
        
        else:
            # unknown类型，跳过
            return False, evidence
    
    def _validate_perception(
        self, 
        step: Dict, 
        failure_case: Dict,
        evidence: Dict
    ) -> Tuple[bool, Dict]:
        """
        验证感知步骤
        
        方法：统计分析（看是否识别了关键目标）
        """
        # 简化版：检查是否提到指令中的关键物体
        instruction = failure_case.get('instruction', '').lower()
        step_content = step['content'].lower()
        
        # 提取关键词
        key_objects = []
        if 'building' in instruction:
            key_objects.append('building')
        if 'red' in instruction:
            key_objects.append('red')
        if 'gray' in instruction or 'grey' in instruction:
            key_objects.append('gray')
        if 'left' in instruction:
            key_objects.append('left')
        if 'right' in instruction:
            key_objects.append('right')
        
        # 检查是否都提到了
        missing_objects = [obj for obj in key_objects if obj not in step_content]
        
        evidence['validation_method'].append('keyword_check')
        evidence['key_objects'] = key_objects
        evidence['missing_objects'] = missing_objects
        
        # 如果缺失关键物体 → 可能是感知错误
        if len(missing_objects) > 0:
            evidence['issue'] = f"Missing key objects: {missing_objects}"
            return True, evidence
        
        return False, evidence
    
    def _validate_comprehension(
        self, 
        step: Dict, 
        failure_case: Dict,
        evidence: Dict
    ) -> Tuple[bool, Dict]:
        """
        验证理解步骤
        
        方法：检查是否正确理解指令
        """
        instruction = failure_case.get('instruction', '').lower()
        step_content = step['content'].lower()
        
        # 检查方向理解
        direction_errors = []
        
        if 'left' in instruction and 'right' in step_content:
            direction_errors.append("Confused left/right")
        
        if 'right' in instruction and 'left' in step_content:
            direction_errors.append("Confused right/left")
        
        if 'forward' in instruction and 'backward' in step_content:
            direction_errors.append("Confused forward/backward")
        
        evidence['validation_method'].append('direction_check')
        evidence['direction_errors'] = direction_errors
        
        if len(direction_errors) > 0:
            evidence['issue'] = f"Direction confusion: {direction_errors}"
            return True, evidence
        
        return False, evidence
    
    def _validate_reasoning(
        self, 
        step: Dict, 
        failure_case: Dict,
        evidence: Dict
    ) -> Tuple[bool, Dict]:
        """
        验证推理步骤
        
        方法：结合反事实和统计分析
        """
        # 查找反事实（动作类型）
        result = self.cf_finder.find_counterfactuals(
            failure_case=failure_case,
            top_k=5,
            verbose=False
        )
        
        counterfactuals = result['counterfactuals']
        statistics = result['statistics']
        
        evidence['validation_method'].append('counterfactual_check')
        evidence['num_counterfactuals'] = len(counterfactuals)
        evidence['statistics'] = statistics
        
        # 检查动作是否异常
        if statistics.get('is_action_unusual', False):
            evidence['issue'] = (
                f"Action choice unusual: used {statistics['failure_first_action']}, "
                f"but {statistics['most_common_first_action']} is most common"
            )
            return True, evidence
        
        return False, evidence
    
    def _validate_decision(
        self, 
        step: Dict, 
        failure_case: Dict,
        evidence: Dict
    ) -> Tuple[bool, Dict]:
        """
        验证决策步骤
        
        方法：检查动作选择
        """
        # 获取实际动作
        actions = failure_case.get('trajectory', {}).get('actions', [])
        
        if not actions:
            return False, evidence
        
        first_action = actions[0]
        
        # 查找统计信息
        result = self.cf_finder.find_counterfactuals(
            failure_case=failure_case,
            top_k=3,
            verbose=False
        )
        
        statistics = result['statistics']
        
        evidence['validation_method'].append('action_statistics')
        evidence['chosen_action'] = first_action
        evidence['common_action'] = statistics.get('most_common_first_action')
        evidence['is_unusual'] = statistics.get('is_action_unusual', False)
        
        if statistics.get('is_action_unusual', False):
            evidence['issue'] = f"Chose action {first_action}, but {statistics['most_common_first_action']} is more common"
            return True, evidence
        
        return False, evidence
    
    def _diagnose_step_error(
        self,
        step: Dict,
        step_idx: int,
        failure_case: Dict,
        evidence: Dict
    ) -> Dict:
        """
        详细诊断步骤错误
        
        生成：根本原因描述 + 置信度
        """
        step_type = step['type']
        issue = evidence.get('issue', 'Unknown error')
        
        # 生成诊断描述
        if step_type == 'perception':
            cause = f"Perception error at step {step_idx}: {issue}"
            confidence = 'medium'
            
        elif step_type == 'comprehension':
            cause = f"Comprehension error at step {step_idx}: {issue}"
            confidence = 'high'
            
        elif step_type == 'reasoning':
            cause = f"Reasoning error at step {step_idx}: {issue}"
            confidence = 'high' if evidence.get('statistics', {}).get('is_action_unusual') else 'medium'
            
        elif step_type == 'decision':
            cause = f"Decision error at step {step_idx}: {issue}"
            confidence = 'high'
            
        else:
            cause = f"Error at step {step_idx}: {issue}"
            confidence = 'low'
        
        return {
            'cause': cause,
            'confidence': confidence
        }
    
    def _statistical_diagnosis(
        self, 
        failure_case: Dict,
        steps: Optional[List[Dict]] = None
    ) -> Dict:
        """
        统计分析诊断（降级方案）
        
        当逐步验证失败时使用
        """
        sample_id = failure_case.get('sample_id', 'unknown')
        
        # 查找统计信息
        result = self.cf_finder.find_counterfactuals(
            failure_case=failure_case,
            top_k=5,
            verbose=False
        )
        
        statistics = result['statistics']
        
        # 基于统计的诊断
        if statistics.get('is_action_unusual', False):
            error_type = 'reasoning'
            cause = (
                f"Statistical analysis suggests action choice error: "
                f"used {statistics['failure_first_action']}, "
                f"but {statistics['most_common_first_action']} is more common "
                f"(success rate: {statistics.get('success_rate', 0)*100:.1f}%)"
            )
            confidence = 'medium'
        else:
            error_type = 'unknown'
            cause = "Unable to determine root cause through statistical analysis"
            confidence = 'low'
        
        return {
            'sample_id': sample_id,
            'error_step': -1,
            'error_type': error_type,
            'root_cause': cause,
            'confidence': confidence,
            'evidence': {
                'method': 'statistical_analysis',
                'statistics': statistics
            },
            'all_steps': steps if steps else []
        }
    
    def diagnose_batch(
        self,
        failures: List[Dict],
        output_file: str,
        checkpoint_interval: int = 10
    ) -> List[Dict]:
        """
        批量诊断
        
        Args:
            failures: 失败案例列表
            output_file: 输出文件
            checkpoint_interval: checkpoint间隔
            
        Returns:
            诊断结果列表
        """
        print(f"\n🚀 批量诊断 ({len(failures)} 个失败案例)")
        
        # 检查checkpoint
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
        results = []
        start_idx = 0
        
        if os.path.exists(checkpoint_file):
            print(f"📂 加载checkpoint...")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                start_idx = len(results)
            print(f"   已完成 {start_idx} 个，继续诊断...")
        
        # 诊断剩余样本
        for idx in range(start_idx, len(failures)):
            failure = failures[idx]
            
            if idx % 5 == 0:
                print(f"进度: {idx}/{len(failures)}")
            
            # 诊断
            diagnosis = self.locate_root_cause(failure)
            results.append(diagnosis)
            
            # 保存checkpoint
            if (idx + 1) % checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'results': results}, f, indent=2)
                print(f"💾 Checkpoint ({idx + 1}/{len(failures)})")
        
        # 保存最终结果
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ 诊断完成！输出: {output_file}")
        
        # 删除checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        # 统计
        self._print_diagnosis_statistics(results)
        
        return results
    
    def _print_diagnosis_statistics(self, results: List[Dict]):
        """打印诊断统计"""
        print("\n" + "="*60)
        print("诊断统计")
        print("="*60)
        
        total = len(results)
        
        # 错误类型分布
        error_types = {}
        for r in results:
            error_type = r.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print(f"总样本数: {total}")
        print("\n错误类型分布:")
        for error_type, count in error_types.items():
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {error_type:15s}: {count:3d} ({percentage:5.1f}%)")
        
        # 置信度分布
        confidence_dist = {}
        for r in results:
            conf = r.get('confidence', 'unknown')
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
        
        print("\n置信度分布:")
        for conf, count in confidence_dist.items():
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {conf:10s}: {count:3d} ({percentage:5.1f}%)")
        
        # 平均错误步骤
        error_steps = [r.get('error_step', -1) for r in results if r.get('error_step', -1) >= 0]
        if error_steps:
            avg_step = sum(error_steps) / len(error_steps)
            print(f"\n平均错误步骤: {avg_step:.1f}")
        
        print("="*60)


# 使用示例
if __name__ == '__main__':
    # 初始化
    locator = RootCauseLocator(
        train_data_path='/home/ubuntu/data1/zx/1OpenFly-Platform/OpenFly-Platform/dataset/Annotation/train.json',
        verbose=True
    )
    
    # 加载失败案例
    failures_file = '../5_failure_collection/outputs/failures.json'
    
    if not os.path.exists(failures_file):
        print(f"❌ 失败案例文件不存在: {failures_file}")
        exit(1)
    
    with open(failures_file, 'r') as f:
        failures = json.load(f)
    
    print(f"📂 加载 {len(failures)} 个失败案例")
    
    # 批量诊断
    results = locator.diagnose_batch(
        failures=failures[:100],  # 先测试100个
        output_file='outputs/diagnosis_results.json',
        checkpoint_interval=10
    )



"""
输入：失败案例（含CoT）
输出：诊断报告
  - 错误步骤（第几步出错）
  - 错误类型（perception/comprehension/reasoning/decision）
  - 根本原因（一句话描述）
  - 置信度（high/medium/low）
  - 证据（反事实、统计等）



  1. CoT分解
   失败案例的CoT → [感知, 理解, 推理, 决策]
   
2. 逐步验证（从前往后）
   For each step:
     - 根据类型选择验证方法
     - 感知：检查关键物体
     - 理解：检查方向理解
     - 推理：查找反事实+统计
     - 决策：检查动作选择
     
     If 发现错误:
       → 这就是根因！（最早错误）
       → 停止验证
       → 生成诊断报告
   
3. 降级方案
   If 逐步验证没找到:
     → 使用纯统计分析
     → 基于动作分布诊断


步骤类型        验证方法            示例
感知           关键词检查        指令说"red building"，CoT提到了吗？
理解            方向检查          指令说"left"，CoT理解成"right"了吗？
推理           反事实+统计           选择的动作在统计上是否异常？
决策              动作统计         第一个动作是否是常见选择？


{
  "sample_id": 999,
  "error_step": 0,
  "error_type": "perception",
  "root_cause": "Perception error at step 0: Missing key objects: ['red', 'left']",
  "confidence": "medium",
  "evidence": {
    "type": "perception",
    "step_idx": 0,
    "validation_method": ["keyword_check"],
    "key_objects": ["building", "red", "left"],
    "missing_objects": ["red", "left"],
    "issue": "Missing key objects: ['red', 'left']"
  },
  "all_steps": [
    {
      "step_id": 0,
      "type": "perception",
      "content": "I see a gray building ahead"
    },
    {
      "step_id": 1,
      "type": "reasoning",
      "content": "I should move forward"
    }
  ]
}

RootCauseLocator (根因定位器)
├── __init__()              - 初始化
├── locate_root_cause()     - 主诊断函数 ⭐⭐⭐⭐⭐
├── _validate_step()        - 验证单步 ⭐⭐⭐⭐
├── _validate_perception()  - 验证感知
├── _validate_comprehension() - 验证理解
├── _validate_reasoning()   - 验证推理
├── _validate_decision()    - 验证决策
├── _diagnose_step_error()  - 生成诊断
├── _statistical_diagnosis() - 统计诊断（降级）
├── diagnose_batch()        - 批量诊断
└── _print_diagnosis_statistics() - 打印统计




# 🔄 完整执行流程示例

**输入**：
```
失败案例：
  指令: "Fly to the red building on the left"
  CoT: "I see a gray building ahead. I should move forward."
  动作: [9, 9, 0]
```

**执行过程**：
```
1. locate_root_cause() 被调用
   ↓
2. CoT分解
   → [perception: "I see a gray building ahead",
      reasoning: "I should move forward"]
   ↓
3. 验证步骤0 (perception)
   → _validate_step() → _validate_perception()
   → 检查关键词：building✓, red✗, left✗
   → 返回：(True, evidence)  ← 错误！
   ↓
4. 发现错误，停止验证
   ↓
5. _diagnose_step_error()
   → 生成诊断："Perception error at step 0: Missing ['red', 'left']"
   → 置信度：medium
   ↓
6. 返回诊断报告
```

---

## 💡 关键设计思想

### **1. 最早错误原则**
```
为什么从前往后验证？

因为：错误会传播
步骤0错 → 步骤1可能也错（基于错误的感知）
步骤1错 → 步骤2可能也错（基于错误的理解）

找最早的错误 = 找根因
```

### **2. 多重证据**
```
不只用一种方法验证：

感知：关键词
理解：方向检查
推理：反事实 + 统计
决策：动作统计

证据越多 → 置信度越高
```

### **3. 降级保护**
```
如果逐步验证都没找到错误：
→ 不是返回"找不到"
→ 而是用统计分析给一个低置信度的诊断

保证：每个失败案例都有诊断
```

---

## ✅ 总结

**核心算法**：
```
逐步验证 + 多重证据 + 降级保护
```

**关键创新**：
```
1. 首次在VLA中做推理链逐步诊断
2. 结合反事实和统计（方案B+C）
3. 最早错误定位（根因）
4. 多层次置信度评估
```

**实际效果**（从测试看）：
```
✅ 准确定位错误步骤（步骤0）
✅ 准确识别错误类型（perception）
✅ 清晰解释原因（缺失red, left）
✅ 提供完整证据（关键词检查）




"""