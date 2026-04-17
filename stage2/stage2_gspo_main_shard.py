#!/home/ubuntu/miniconda3/envs/lyy-openfly/bin/python
"""
Stage 2 Multi-task GSPO 主训练脚本 - 修复版

修复内容：
1. 修正 _rank_candidates_with_scorer 中的诊断信息处理
2. 添加断点恢复机制
3. 添加原子性保存（防止文件损坏）
4. 更频繁的检查点保存（每50个样本）

使用方法：
    python stage2_gspo_main.py
"""

import os
import sys
import json
import torch
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

# ===== 调试信息 =====
print("=" * 60)
print("调试信息 - 模块导入")
print("=" * 60)
print(f"当前工作目录: {Path.cwd()}")
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")

# ===== 首先添加所有必要路径 =====
PROJECT_ROOT = Path("/home/ubuntu/data1/lyy/full_rlds_project-3")
OPENFLY_PATH = '/home/ubuntu/data1/lyy/OpenFly-Platform/train'
DIAGNOSIS_DIR = PROJECT_ROOT / '6_diagnosis'

print(f"\n项目路径:")
print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
print(f"  DIAGNOSIS_DIR: {DIAGNOSIS_DIR}")
print(f"  OPENFLY_PATH: {OPENFLY_PATH}")

if str(DIAGNOSIS_DIR) not in sys.path:
    sys.path.insert(0, str(DIAGNOSIS_DIR))
    print(f"\n✓ 添加路径到sys.path: {DIAGNOSIS_DIR}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"✓ 添加路径到sys.path: {PROJECT_ROOT}")

if OPENFLY_PATH not in sys.path:
    sys.path.insert(0, OPENFLY_PATH)
    print(f"✓ 添加路径到sys.path: {OPENFLY_PATH}")

print(f"\nsys.path前5项:")
for i, p in enumerate(sys.path[:5]):
    print(f"  {i}: {p}")

print(f"\n开始导入模块...")
print("=" * 60)

# ===== 导入诊断模块 =====
try:
    from multitask_model import MultiTaskVLA
    print("✓ 导入 multitask_model.MultiTaskVLA")
except Exception as e:
    print(f"✗ 导入 multitask_model 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from dynamic_loss import DynamicLoss
    print("✓ 导入 dynamic_loss.DynamicLoss")
except Exception as e:
    print(f"✗ 导入 dynamic_loss 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from multitask_gspo_trainer import MultiTaskGSPOTrainer
    print("✓ 导入 multitask_gspo_trainer.MultiTaskGSPOTrainer")
except Exception as e:
    print(f"✗ 导入 multitask_gspo_trainer 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from diagnosis_scorer import DiagnosisScorer
    print("✓ 导入 diagnosis_scorer.DiagnosisScorer")
except Exception as e:
    print(f"✗ 导入 diagnosis_scorer 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from stage2_config import Stage2Config
    print("✓ 导入 stage2_config.Stage2Config")
except Exception as e:
    print(f"✗ 导入 Stage2Config 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("✓ 所有模块导入成功")
print("=" * 60)
print()


class Stage2Trainer:
    """Stage 2 Multi-task GSPO训练管理器"""
    
    @staticmethod
    def _parse_generated_cot(generated_text: str) -> tuple:
        import re

        if generated_text is None:
            return "", None

        text = str(generated_text)
        if "[/INST]" in text:
            text = text.split("[/INST]", 1)[-1]

        thinking_text = ""
        m = re.search(r"<thinking>\s*(.*?)\s*</thinking>", text, re.IGNORECASE | re.DOTALL)
        if m:
            thinking_text = m.group(1).strip()
        else:
            m2 = re.search(r"<thinking>\s*(.*)", text, re.IGNORECASE | re.DOTALL)
            if m2:
                thinking_text = m2.group(1).strip()
            else:
                thinking_text = text.strip()

        thinking_text = re.sub(r"</?thinking>", "", thinking_text, flags=re.IGNORECASE).strip()
        thinking_text = re.sub(
            r"<(?:next_action|action)>\s*.*?\s*</(?:next_action|action)>",
            "",
            thinking_text,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()

        action_number = None
        patterns = [
            r"<action>\s*(-?\d+)\s*</action>",
            r"<next_action>\s*(-?\d+)\s*</next_action>",
            r"\bnext[_\s-]*action\s*[:=]\s*(-?\d+)\b",
            r"\baction\s*[:=]\s*(-?\d+)\b",
        ]
        for ptn in patterns:
            am = re.search(ptn, text, re.IGNORECASE)
            if am:
                try:
                    v = int(am.group(1))
                    action_number = v if 0 <= v <= 9 else None
                except Exception:
                    action_number = None
                break

        return thinking_text, action_number

    @staticmethod
    def _extract_ground_truth(sample: Dict, frame_idx: str) -> Dict:
        import re

        def _norm(x):
            return str(x).strip()

        def _to_int(x):
            try:
                return int(str(x).strip())
            except Exception:
                return None

        gt_cot = ""
        gt_actions = []

        index_list_raw = (
            sample.get("index_list")
            or sample.get("frame_index_list")
            or sample.get("frame_indices")
            or []
        )
        if not isinstance(index_list_raw, list):
            index_list_raw = list(index_list_raw) if index_list_raw else []

        index_list = [_norm(x) for x in index_list_raw]
        cur = _norm(frame_idx)

        current_idx = None
        if cur in index_list:
            current_idx = index_list.index(cur)
        else:
            cur_i = _to_int(cur)
            if cur_i is not None:
                for i, v in enumerate(index_list):
                    if _to_int(v) == cur_i:
                        current_idx = i
                        break

        if current_idx is None:
            return {"cot": "", "actions": []}

        cot_dict = (
            sample.get("cot")
            or sample.get("cot_dict")
            or sample.get("cot_map")
            or sample.get("cot_pairs")
            or {}
        )
        if not isinstance(cot_dict, dict):
            cot_dict = {}

        raw_cot = None
        if cot_dict and current_idx < len(index_list) - 1:
            nxt = index_list[current_idx + 1]
            cur_i = _to_int(cur)
            nxt_i = _to_int(nxt)

            key_candidates = [
                f"{cur}-{nxt}",
                f"{cur}_{nxt}",
                f"{cur},{nxt}",
                f"{cur}->{nxt}",
                f"{cur} {nxt}",
            ]
            if cur_i is not None and nxt_i is not None:
                key_candidates.extend([
                    f"{cur_i}-{nxt_i}",
                    f"{cur_i}_{nxt_i}",
                    f"{cur_i},{nxt_i}",
                    f"{cur_i}->{nxt_i}",
                    f"{cur_i} {nxt_i}",
                ])

            norm_key_map = {_norm(k): k for k in cot_dict.keys()}

            for k in key_candidates:
                if k in cot_dict:
                    raw_cot = cot_dict[k]
                    break
                nk = _norm(k)
                if nk in norm_key_map:
                    raw_cot = cot_dict[norm_key_map[nk]]
                    break

            if raw_cot is None:
                for k, v in cot_dict.items():
                    ks = _norm(k)
                    if cur in ks and nxt in ks:
                        raw_cot = v
                        break

            if raw_cot is None and cur_i is not None and nxt_i is not None:
                for k, v in cot_dict.items():
                    nums = [int(x) for x in re.findall(r"-?\d+", _norm(k))]
                    if len(nums) >= 2 and nums[0] == cur_i and nums[1] == nxt_i:
                        raw_cot = v
                        break

        if isinstance(raw_cot, dict):
            raw_cot = (
                raw_cot.get("cot")
                or raw_cot.get("thinking")
                or raw_cot.get("text")
                or raw_cot.get("output")
                or ""
            )
        elif isinstance(raw_cot, list):
            raw_cot = " ".join(str(x) for x in raw_cot if x is not None)
        elif raw_cot is None:
            raw_cot = ""

        gt_cot, _ = Stage2Trainer._parse_generated_cot(raw_cot)
        if not gt_cot:
            txt2 = str(raw_cot).strip()
            txt2 = re.sub(r"</?thinking>", "", txt2, flags=re.IGNORECASE)
            txt2 = re.sub(r"</?action>", "", txt2, flags=re.IGNORECASE)
            txt2 = re.sub(r"</?next_action>", "", txt2, flags=re.IGNORECASE)
            gt_cot = txt2.strip()

        action_src = sample.get("action")
        if action_src is None:
            action_src = sample.get("actions")
        if action_src is None:
            action_src = sample.get("action_list")

        if isinstance(action_src, list):
            if len(action_src) > 0 and isinstance(action_src[0], dict):
                rows = []
                for row in action_src:
                    if not isinstance(row, dict):
                        continue
                    f = row.get("frame_idx", row.get("frame_id", row.get("idx", row.get("index"))))
                    a = row.get("action", row.get("action_id", row.get("next_action")))
                    fi = _to_int(f)
                    ai = _to_int(a)
                    if fi is not None and ai is not None:
                        rows.append((fi, ai))
                rows.sort(key=lambda x: x[0])

                cur_i = _to_int(cur)
                if cur_i is not None:
                    gt_actions = [a for fi, a in rows if fi >= cur_i][:10]
            else:
                start = min(current_idx, len(action_src))
                for a in action_src[start:start + 10]:
                    ai = _to_int(a)
                    if ai is not None:
                        gt_actions.append(ai)

        elif isinstance(action_src, dict):
            items = []
            for k, v in action_src.items():
                fi = _to_int(k)
                if isinstance(v, dict):
                    v = v.get("action", v.get("action_id"))
                ai = _to_int(v)
                if fi is not None and ai is not None:
                    items.append((fi, ai))
            items.sort(key=lambda x: x[0])

            cur_i = _to_int(cur)
            if cur_i is None:
                gt_actions = [a for _, a in items[:10]]
            else:
                gt_actions = [a for fi, a in items if fi >= cur_i][:10]

        return {"cot": gt_cot, "actions": gt_actions}

    def _rank_candidates_with_scorer(
        self,
        candidates: List[Dict],
        sample: Dict,
        scorer,
        frame_idx: str,
        sample_idx: int = -1
    ) -> tuple:
        """
        使用DiagnosisScorer对候选进行完整评分和排序
        
        【修复版】：正确处理诊断信息（error_type, error_step, confidence）
        """
        # 1. 提取Ground Truth
        gt = self._extract_ground_truth(sample, frame_idx)
        
        # 2. 转换候选格式
        formatted_candidates = []
        for cand in candidates:
            raw_text = (cand.get('generated_only') or cand.get('generated_text') or '').strip()
            thinking, action = self._parse_generated_cot(raw_text)
            
            formatted_candidates.append({
                'cot': thinking,
                'actions': [action] if action is not None else [],
                'original': cand
            })
        
        # 3. 调用DiagnosisScorer的rank_candidates
        try:
            result = scorer.rank_candidates(
                candidates=formatted_candidates,
                gt=gt,
                sample={
                    'instruction': sample.get('gpt_instruction') or sample.get('instruction', '')
                },
                return_all_scores=True
            )
            
            # 4. 从result中提取chosen和rejected
            chosen = result['chosen']['original'].copy()
            rejected = result['rejected']['original'].copy()
            
            chosen['score'] = result['chosen'].get('score', 0.0)
            rejected['score'] = result['rejected'].get('score', 0.0)
            
            # 5. 构建完整诊断信息
            diagnosis_info = {
                'chosen_score': chosen['score'],
                'rejected_score': rejected['score'],
                'score_gap': chosen['score'] - rejected['score'],
                'gt_cot': gt.get('cot', ''),
                'all_candidates': []
            }
            
            for cand_result in result.get('all_candidates', []):
                cand_info = {
                    'temperature': cand_result['original']['temperature'],
                    'generated_cot': cand_result.get('cot', ''),
                    'score': cand_result.get('score', 0.0),
                    'diagnosis': cand_result.get('diagnosis', {})
                }
                diagnosis_info['all_candidates'].append(cand_info)
            
            # ⭐ 关键修复：正确提取chosen的诊断信息
            chosen_diagnosis = result['chosen'].get('diagnosis', {})
            if chosen_diagnosis:
                diagnosis_info['error_type'] = chosen_diagnosis.get('error_type', 'unknown')
                diagnosis_info['error_step'] = chosen_diagnosis.get('error_step', -1)
                diagnosis_info['confidence'] = chosen_diagnosis.get('confidence', 'low')
            else:
                diagnosis_info['error_type'] = 'unknown'
                diagnosis_info['error_step'] = -1
                diagnosis_info['confidence'] = 'low'
            
            # 前5个样本打印评分信息
            if sample_idx >= 0 and sample_idx < 5:
                print(f"\n  ✓ 样本 {sample_idx} 评分完成:")
                print(f"     Chosen (temp={chosen['temperature']}): score={chosen['score']:.3f}")
                print(f"     Rejected (temp={rejected['temperature']}): score={rejected['score']:.3f}")
                print(f"     Gap: {diagnosis_info['score_gap']:.3f}")
                print(f"     Error Type: {diagnosis_info['error_type']}")
            
        except Exception as e:
            print(f"   ⚠️  DiagnosisScorer评分失败: {e}")
            print(f"   使用fallback：基于温度排序")
            
            import traceback
            traceback.print_exc()
            
            sorted_candidates = sorted(candidates, key=lambda x: x['temperature'])
            chosen = sorted_candidates[0].copy()
            rejected = sorted_candidates[-1].copy()
            
            chosen['score'] = 0.6
            rejected['score'] = 0.3
            
            diagnosis_info = {
                'chosen_score': 0.6,
                'rejected_score': 0.3,
                'score_gap': 0.3,
                'gt_cot': gt.get('cot', ''),
                'error_type': 'unknown',
                'error_step': -1,
                'confidence': 'low',
                'all_candidates': [],
                'fallback': True
            }
        
        return chosen, rejected, diagnosis_info
    
    def __init__(self, config: Stage2Config, args):
        """初始化训练管理器"""
        self.config = config
        self.args = args
        self.device = config.device
        
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_log = []
        self.log_file = self.save_dir / "training_log.json"
        
        print(f"\n{'='*60}")
        print("Stage 2 Multi-task GSPO 训练初始化")
        print(f"{'='*60}")
        print(f"保存目录: {self.save_dir}")
        print(f"设备: {self.device}")
        print(f"轮数: {config.num_rounds}")
        print(f"每轮步数: {config.steps_per_round}")
        print(f"Batch size: {config.batch_size}")
        print(f"{'='*60}\n")
    
    def _atomic_save(self, data, filepath: Path):
        """原子性保存JSON文件"""
        temp_path = filepath.with_suffix('.tmp')
        
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            shutil.move(str(temp_path), str(filepath))
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _save_checkpoint_with_progress(
        self,
        pairs: List[Dict],
        diagnosis_records: List[Dict],
        completed: int,
        candidates_path: Path,
        diagnosis_path: Path,
        progress_path: Path
    ):
        """保存检查点和进度（原子性）"""
        try:
            if len(pairs) == 0:
                print(f"\n  ⚠️ 没有有效候选对，跳过保存，保留原文件")
                return

            # 保存候选对（不包含pixel_values，太大了）
            pairs_to_save = []
            for p in pairs:
                p_copy = {k: v for k, v in p.items() if k != 'pixel_values'}
                pairs_to_save.append(p_copy)
            
            self._atomic_save(pairs_to_save, candidates_path)
            self._atomic_save(diagnosis_records, diagnosis_path)
            
            progress_data = {
                'completed_samples': completed,
                'total_pairs': len(pairs),
                'total_diagnosis': len(diagnosis_records),
                'timestamp': datetime.now().isoformat()
            }
            self._atomic_save(progress_data, progress_path)
            
            print(f"\n  💾 检查点已保存: {completed} 个样本完成")
            
        except Exception as e:
            print(f"\n  ⚠️  检查点保存失败: {e}")
    
    def load_data(self, round_num: int = 0) -> List[Dict]:
        """加载训练数据"""
        print(f"\n{'='*60}")
        print("加载训练数据")
        print(f"{'='*60}")

        data_path = Path(self.config.data_path)
        print(f"加载数据: {data_path}")

        with open(data_path, 'r') as f:
            train_data = json.load(f)

        print(f"  ✓ 加载了 {len(train_data)} 个样本")

        aux_labels_dict = {}
        if round_num == 0:
            print("  Round 0: no aux_labels by design")
        else:
            aux_labels_path = Path(self.config.get_aux_labels_path(round_num))
            if aux_labels_path.exists():
                print(f"加载辅助标签: {aux_labels_path}")
                with open(aux_labels_path, 'r') as f:
                    aux_payload = json.load(f)

                aux_labels_list = aux_payload if isinstance(aux_payload, list) else aux_payload.get('samples', [])
                for item in aux_labels_list:
                    sample_id = str(item.get('sample_id') if item.get('sample_id') is not None else item.get('sample_idx', ''))
                    if sample_id and item.get('aux_labels') is not None:
                        aux_labels_dict[sample_id] = item.get('aux_labels')

                print(f"  ✓ 加载了 {len(aux_labels_dict)} 个样本的辅助标签")
            else:
                print(f"  ⚠️  未找到辅助标签文件: {aux_labels_path}")
                print("      将只使用GSPO loss训练")

        dataset = []
        for idx, sample in enumerate(train_data):
            sample = dict(sample)
            sample_id = str(idx)
            sample['aux_labels'] = aux_labels_dict.get(sample_id)
            dataset.append(sample)

        if self.args.test:
            original_size = len(dataset)
            dataset = dataset[:10]
            print(f"\n⚠️  测试模式：数据集从 {original_size} 减少到 {len(dataset)} 个样本")

        samples_with_labels = sum(1 for s in dataset if s.get('aux_labels') is not None)

        print(f"\n数据统计:")
        print(f"  总样本数: {len(dataset)}")
        print(f"  有辅助标签: {samples_with_labels}")
        print(f"  无辅助标签: {len(dataset) - samples_with_labels}")
        print(f"{'='*60}\n")

        return dataset

    def initialize_components(self) -> Dict:
        """初始化所有训练组件"""
        print(f"\n{'='*60}")
        print("初始化训练组件")
        print(f"{'='*60}")
        
        print("\n[1/6] 初始化MultiTaskVLA...")
        multitask_vla = MultiTaskVLA(
            stage1_checkpoint_dir=self.config.stage1_model_path,
            num_unfrozen_layers=4,
            num_keywords=34,
            hidden_dim=4096,
            dropout=0.1,
            verbose=True
        )
        multitask_vla = multitask_vla.to(self.device)
        
        print("\n[2/6] 初始化DynamicLoss...")
        dynamic_loss = DynamicLoss(
            lambda_keyword=0.15,
            lambda_direction=0.1,
            lambda_quality=0.1,
            lambda_validity=0.1,
            device=self.device,
            verbose=True
        )
        
        print("\n[3/6] 初始化MultiTaskGSPOTrainer...")
        trainer = MultiTaskGSPOTrainer(
            multitask_vla=multitask_vla,
            dynamic_loss=dynamic_loss,
            config=self.config,
            device=self.device
        )
        
        print("\n[4/6] 初始化完整诊断框架...")
        
        from root_cause_locator import RootCauseLocator
        
        print("  初始化根因定位器...")
        root_cause_locator = RootCauseLocator(
            train_data_path="/home/ubuntu/data1/zx/1OpenFly-Platform/OpenFly-Platform/dataset/Annotation/train.json",
            verbose=False
        )
        
        print("  初始化诊断评分器...")
        diagnosis_scorer = DiagnosisScorer(
            cot_parser=root_cause_locator.cot_parser,
            root_cause_locator=root_cause_locator,
            error_penalties={
                0: 0.4,
                1: 0.3,
                2: 0.2,
                3: 0.1,
            }
        )
        
        print("✅ 完整诊断框架初始化完成")
        
        print("\n[5/6] 获取Tokenizer...")
        tokenizer = multitask_vla.base_vla.llm_backbone.tokenizer
        
        print(f"{'='*60}")
        print("✅ 所有组件初始化完成")
        print(f"{'='*60}\n")
        
        return {
            'multitask_vla': multitask_vla,
            'dynamic_loss': dynamic_loss,
            'trainer': trainer,
            'diagnosis_scorer': diagnosis_scorer,
            'tokenizer': tokenizer
        }
    
    def generate_candidates_simple(
        self,
        vla,
        sample: Dict,
        tokenizer,
        num_candidates: int = 3
    ) -> List[Dict]:
        """简化版候选生成"""
        instruction = sample.get('gpt_instruction') or sample.get('instruction') or ''
        
        base_model = vla.base_vla if hasattr(vla, 'base_vla') else vla
        
        if hasattr(base_model, 'get_prompt_builder'):
            prompt_builder = base_model.get_prompt_builder()
        else:
            from model.prompt_llama2 import LLaMa2ChatPromptBuilder
            prompt_builder = LLaMa2ChatPromptBuilder("prismatic")

        image_transform = base_model.vision_backbone.image_transform

        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?"
        )
        prompt_text = prompt_builder.get_prompt()
        
        pixel_values = None
        episode_folder = sample.get('image_path') or sample.get('episode_id')
        frame_name = sample.get('frame_idx')
        
        if not frame_name and 'index_list' in sample and len(sample['index_list']) > 0:
            frame_name = sample['index_list'][0]

        if episode_folder:
            try:
                from PIL import Image
                
                base_root = Path(self.config.image_base_path)
                folder_path = base_root / episode_folder
                
                target_file = None
                
                if frame_name:
                    for ext in ['.jpg', '.png', '.jpeg', '']:
                        f_path = folder_path / f"{frame_name}{ext}" if ext else folder_path / frame_name
                        if f_path.exists():
                            target_file = f_path
                            break
                
                if target_file is None and folder_path.exists():
                    all_imgs = sorted(list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")))
                    if all_imgs:
                        target_file = all_imgs[0]

                if target_file and target_file.exists():
                    img = Image.open(target_file).convert('RGB')
                    
                    if img.size[0] == 0 or img.size[1] == 0:
                        raise ValueError(f"图片尺寸无效: {img.size}")
                    
                    tr_img = image_transform(img)
                    
                    temp_pixel_values = {}
                    for k in tr_img.keys():
                        combined = torch.cat(
                            (tr_img[k], tr_img[k], tr_img[k]), 
                            dim=0
                        )
                        temp_pixel_values[k] = combined.unsqueeze(0).to(self.device)
                    
                    pixel_values = temp_pixel_values
                    
            except Exception as e:
                if not self.args.test:
                    print(f"❌ 图片处理异常: {e}")
                pixel_values = None



        if pixel_values is None:
            return []





        candidates = []
        temperatures = [0.7, 1.0][:num_candidates]
        
        prompt_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        vocab_size = tokenizer.vocab_size  # ⭐ 提前获取词表大小
        
        for temp in temperatures:
            current_ids = prompt_ids.clone()
            try:
                with torch.no_grad():
                    for _ in range(self.config.max_new_tokens):
                        outputs = base_model(pixel_values=pixel_values, input_ids=current_ids)
                        next_token_logits = outputs.logits[:, -1, :]
                        
                        # ⭐ 防护1：限制logits在词表范围内
                        effective_vocab = min(vocab_size, next_token_logits.shape[-1])
                        next_token_logits = next_token_logits[:, :effective_vocab]
                        
                        if temp > 0:
                            probs = torch.softmax(next_token_logits / temp, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        # ⭐ 防护2：二次检查，无效则结束
                        if next_token.item() >= vocab_size or next_token.item() < 0:
                            break
                        
                        current_ids = torch.cat([current_ids, next_token], dim=1)
                        if next_token.item() == tokenizer.eos_token_id: 
                            break
                
                # ⭐ 防护3：decode前过滤无效token
                valid_ids = [t for t in current_ids[0].tolist() if 0 <= t < vocab_size]
                generated_text = tokenizer.decode(valid_ids, skip_special_tokens=True)
                
                if prompt_text in generated_text:
                    generated_only = generated_text.split(prompt_text)[-1].strip()
                else:
                    generated_only = generated_text[len(prompt_text):].strip()
                if "ASSISTANT:" in generated_only:
                    generated_only = generated_only.split("ASSISTANT:")[-1].strip()
                
                candidates.append({
                    'generated_text': generated_text,
                    'generated_only': generated_only,
                    'temperature': temp
                })
                
            except Exception as e:
                # ⭐ 防护4：单个温度失败不影响其他
                print(f"      ⚠️ 温度{temp}生成失败: {e}")
                candidates.append({
                    'generated_text': prompt_text,
                    'generated_only': '',
                    'temperature': temp
                })
        
        # ⭐ 防护5：确保至少有一个候选
        if len(candidates) == 0:
            candidates.append({
                'generated_text': prompt_text,
                'generated_only': '',
                'temperature': 0.7
            })
        
        return candidates







    
    def _early_analysis(self, records: List[Dict], round_num: int):
        """前100个样本的早期分析"""
        import numpy as np
        from collections import Counter
        
        scores = [r['chosen_score'] for r in records]
        rejected_scores = [r['rejected_score'] for r in records]
        gaps = [r['score_gap'] for r in records]
        
        avg_chosen = np.mean(scores)
        avg_rejected = np.mean(rejected_scores)
        avg_gap = np.mean(gaps)
        
        weak_count = len([r for r in records if r['chosen_score'] < 0.6])
        weak_ratio = weak_count / len(records)
        
        zero_count = len([r for r in records if r['chosen_score'] == 0])
        nan_count = len([r for r in records if np.isnan(r['chosen_score'])])
        fallback_count = len([r for r in records if r.get('fallback', False)])
        
        # ⭐ 统计 error_type 分布
        error_types = [r.get('error_type', 'unknown') for r in records]
        error_type_counter = Counter(error_types)
        
        print("\n" + "="*60)
        print("【前100个样本快速分析】")
        print("="*60)
        print(f"平均Chosen分数:   {avg_chosen:.3f}")
        print(f"平均Rejected分数: {avg_rejected:.3f}")
        print(f"平均分数差距:     {avg_gap:.3f}")
        print(f"弱样本数量:       {weak_count}/100 ({weak_ratio:.0%})")
        
        # ⭐ 打印 error_type 分布
        print(f"\n错误类型分布:")
        for error_type, count in error_type_counter.most_common():
            print(f"  {error_type}: {count}")
        
        # 检查诊断是否正常工作
        unknown_ratio = error_type_counter.get('unknown', 0) / len(records)
        if unknown_ratio > 0.8:
            print(f"\n⚠️  警告：{unknown_ratio*100:.0f}% 的样本 error_type 为 unknown")
            print(f"   诊断框架可能未正常工作，请检查 diagnosis_scorer.py")
        else:
            print(f"\n✅ 诊断框架正常工作（unknown比例: {unknown_ratio*100:.0f}%）")
        
        if zero_count > 50 or nan_count > 10 or fallback_count > 20:
            print("\n❌❌❌ 严重错误：检测到技术问题！")
            print(f"   零分样本: {zero_count}/100")
            print(f"   异常值: {nan_count}/100")
            print(f"   Fallback样本: {fallback_count}/100")
            
            response = input("\n是否继续？(y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        elif avg_chosen < 0.4:
            print("\n📊 情况说明：当前模型表现较弱（这是正常的Stage1问题）")
            print("   → 继续训练，无需担心")
        
        elif avg_chosen < 0.6:
            print("\n📊 情况说明：当前模型表现中等（符合预期）")
            print("   → 继续训练")
        
        else:
            print("\n📊 情况说明：当前模型表现良好")
            print("   → 继续训练")
        
        print("="*60 + "\n")
    
    def _final_analysis(self, records: List[Dict], round_num: int, dataset: List[Dict], pairs: List[Dict]):
        """完整分析并生成报告"""
        import numpy as np
        from collections import Counter
        
        print(f"\n{'='*60}")
        print(f"Round {round_num} 完整诊断分析")
        print(f"{'='*60}")
        
        scores = [r['chosen_score'] for r in records]
        rejected_scores = [r['rejected_score'] for r in records]
        gaps = [r['score_gap'] for r in records]
        
        avg_chosen = np.mean(scores)
        avg_rejected = np.mean(rejected_scores)
        avg_gap = np.mean(gaps)
        
        sorted_records = sorted(records, key=lambda x: x['chosen_score'])
        target_count = int(len(records) * 0.35)
        weak_samples = sorted_records[:target_count]
        
        error_types = [r['error_type'] for r in weak_samples if r['error_type'] != 'unknown']
        error_type_counter = Counter(error_types)
        
        score_bins = {
            '[0.8, 1.0]': len([s for s in scores if 0.8 <= s <= 1.0]),
            '[0.6, 0.8)': len([s for s in scores if 0.6 <= s < 0.8]),
            '[0.4, 0.6)': len([s for s in scores if 0.4 <= s < 0.6]),
            '[0.0, 0.4)': len([s for s in scores if 0.0 <= s < 0.4])
        }
        
        report_path = self.save_dir / f"round_{round_num}_diagnosis_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"Round {round_num} 完整诊断总结\n")
            f.write("="*80 + "\n\n")
            
            f.write("【整体统计】\n")
            f.write("-"*80 + "\n")
            f.write(f"总样本数: {len(records)}\n")
            f.write(f"正常样本: {len(records) - len(weak_samples)} ({(len(records)-len(weak_samples))/len(records)*100:.1f}%)\n")
            f.write(f"弱样本: {len(weak_samples)} ({len(weak_samples)/len(records)*100:.1f}%)\n\n")
            
            f.write("分数统计:\n")
            f.write(f"  Chosen平均分: {avg_chosen:.3f}\n")
            f.write(f"  Rejected平均分: {avg_rejected:.3f}\n")
            f.write(f"  平均Gap: {avg_gap:.3f}\n\n")
            
            f.write("分数区间分布:\n")
            for bin_range, count in score_bins.items():
                f.write(f"  {bin_range}: {count} 样本 ({count/len(records)*100:.1f}%)\n")
            f.write("\n")
            
            f.write("【弱样本分析】\n")
            f.write("-"*80 + "\n")
            f.write(f"弱样本定义: Chosen分数最低的35% ({target_count}个)\n\n")
            
            f.write("错误类型分布:\n")
            for error_type, count in error_type_counter.most_common():
                f.write(f"  {error_type}: {count} ({count/len(weak_samples)*100:.1f}%)\n")
            f.write("\n")
            
            f.write("【详细弱样本列表】\n")
            f.write("-"*80 + "\n")
            for idx, weak in enumerate(weak_samples[:50]):
                f.write(f"\n#{idx+1}. 样本ID: {weak['sample_id']} | Chosen: {weak['chosen_score']:.3f} | ")
                f.write(f"Gap: {weak['score_gap']:.3f} | 错误: {weak['error_type']}\n")
                f.write(f"指令: {weak['instruction'][:80]}...\n")
                f.write("-"*80 + "\n")
            
            f.write("\n【建议】\n")
            f.write("-"*80 + "\n")
            f.write(f"1. 为{len(weak_samples)}个弱样本生成辅助标签\n")
            f.write(f"2. Round {round_num+1}训练时启用辅助loss\n")
            f.write("="*80 + "\n")
        
        print(f"✅ 诊断报告已保存: {report_path}")
        print(f"   弱样本数量: {len(weak_samples)} ({len(weak_samples)/len(records)*100:.1f}%)")
        print(f"   错误类型: {dict(error_type_counter)}")
        
        weak_samples_path = self.save_dir / f"round_{round_num}_weak_samples.json"
        weak_samples_data = []
        
        for w in weak_samples:
            idx = w['sample_idx']
            
            sample_data = dataset[idx] if idx < len(dataset) else {}
            pair_data = pairs[idx] if idx < len(pairs) else {}
            chosen = pair_data.get('chosen', {})
            
            chosen_text = chosen.get('generated_only', chosen.get('generated_text', ''))
            chosen_cot, chosen_action = self._parse_generated_cot(chosen_text)
            chosen_actions = [chosen_action] if chosen_action is not None else []
            
            gt_cot = sample_data.get('cot', sample_data.get('thinking', ''))
            gt_actions = sample_data.get('action', [])
            if not isinstance(gt_actions, list):
                gt_actions = [gt_actions] if gt_actions is not None else []
            
            weak_sample = {
                'sample_idx': idx,
                'sample_id': w['sample_id'],
                'sample': {
                    'instruction': w['instruction'],
                    'image_path': sample_data.get('image_path', ''),
                    'frame_idx': sample_data.get('frame_idx', ''),
                    'episode_id': sample_data.get('episode_id', '')
                },
                'model_output': {
                    'cot': chosen_cot,
                    'actions': chosen_actions
                },
                'gt': {
                    'cot': gt_cot,
                    'actions': gt_actions
                },
                'chosen_score': w['chosen_score'],
                'error_type': w['error_type'],
                'score_gap': w.get('score_gap', 0.0),
                'confidence': records[idx].get('confidence', 'low') if idx < len(records) else 'low'
            }
            
            weak_samples_data.append(weak_sample)
        
        with open(weak_samples_path, 'w') as f:
            json.dump(weak_samples_data, f, indent=2)
        
        print(f"✅ 弱样本列表已保存: {weak_samples_path}")
        print(f"   格式: 兼容qwen_screener和auxiliary_labeler")
        print(f"{'='*60}\n")
    

    def _prepare_pairs_for_training(
            self,
            candidates_path: Path,
            diagnosis_path: Path,
            dataset: List[Dict],
            components: Dict,
            round_num: int
        ) -> List[Dict]:
            """
            从已保存的候选数据准备训练pairs（重新加载图片）
            
            当候选生成已完成但需要训练时调用
            """
            print(f"\n{'='*60}")
            print(f"从已保存的候选数据准备训练...")
            print(f"{'='*60}")
            
            # 1. 加载已保存的候选对和诊断记录
            with open(candidates_path, 'r') as f:
                saved_pairs = json.load(f)
            
            with open(diagnosis_path, 'r') as f:
                diagnosis_records = json.load(f)
            
            print(f"加载了 {len(saved_pairs)} 个候选对")
            print(f"加载了 {len(diagnosis_records)} 个诊断记录")
            print(f"dataset 样本数: {len(dataset)}")
            
            # 2. 构建 sample_id -> sample_idx 的映射（从 diagnosis_records）
            id_to_idx = {}
            for record in diagnosis_records:
                sample_id = record.get('sample_id')
                sample_idx = record.get('sample_idx')
                if sample_id is not None and sample_idx is not None:
                    id_to_idx[str(sample_id)] = sample_idx
            
            print(f"构建 sample_id->sample_idx 映射: {len(id_to_idx)} 个")
            
            # 调试信息
            if len(saved_pairs) > 0:
                print(f"\n调试 - 前3个 saved_pair 的 sample_id:")
                for i, p in enumerate(saved_pairs[:3]):
                    sid = p.get('sample_id')
                    idx = id_to_idx.get(str(sid))
                    print(f"  [{i}] sample_id={sid}, 映射到 sample_idx={idx}")
            
            # 3. 为每个 pair 重新加载图片
            vla = components['multitask_vla']
            image_transform = vla.base_vla.vision_backbone.image_transform
            
            pairs = []
            skipped = 0
            skip_reasons = {
                'generation_failed': 0,
                'sample_not_found': 0,
                'image_not_found': 0,
                'image_load_error': 0
            }
            
            from PIL import Image
            
            for pair in tqdm(saved_pairs, desc="加载图片"):
                # 跳过生成失败的
                if pair.get('generation_failed', False):
                    skip_reasons['generation_failed'] += 1
                    skipped += 1
                    continue
                
                # ⭐ 修改：通过 diagnosis_records 的 sample_idx 获取 dataset 样本
                sample_id = pair.get('sample_id', '')
                sample_idx = id_to_idx.get(str(sample_id))
                
                if sample_idx is None or sample_idx >= len(dataset):
                    skip_reasons['sample_not_found'] += 1
                    skipped += 1
                    continue
                
                sample = dataset[sample_idx]
                
                # 加载图片
                try:
                    episode_folder = sample.get('image_path') or sample.get('episode_id')
                    frame_name = sample.get('frame_idx')
                    if not frame_name and 'index_list' in sample:
                        frame_name = sample['index_list'][0] if sample['index_list'] else None
                    
                    if not episode_folder or not frame_name:
                        skip_reasons['image_not_found'] += 1
                        skipped += 1
                        continue
                    
                    base_root = Path(self.config.image_base_path)
                    folder_path = base_root / episode_folder
                    
                    img_path = None
                    for ext in ['.png', '.jpg', '.jpeg']:
                        test_path = folder_path / f"{frame_name}{ext}"
                        if test_path.exists():
                            img_path = test_path
                            break
                    
                    # 也尝试不带扩展名的
                    if img_path is None:
                        test_path = folder_path / frame_name
                        if test_path.exists():
                            img_path = test_path
                    
                    if img_path is None:
                        skip_reasons['image_not_found'] += 1
                        skipped += 1
                        continue
                    
                    img = Image.open(img_path).convert('RGB')
                    tr_img = image_transform(img)
                    
                    pixel_values = {}
                    for k in tr_img.keys():
                        combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0)
                        pixel_values[k] = combined.unsqueeze(0).to(self.device)
                    
                    # 构建完整的 pair
                    complete_pair = {
                        **pair,
                        'pixel_values': pixel_values
                    }
                    pairs.append(complete_pair)
                    
                except Exception as e:
                    skip_reasons['image_load_error'] += 1
                    skipped += 1
                    continue
            
            print(f"\n{'='*60}")
            print(f"✅ 训练数据准备完成:")
            print(f"   有效 pairs: {len(pairs)}")
            print(f"   跳过: {skipped}")
            print(f"   跳过原因:")
            for reason, count in skip_reasons.items():
                if count > 0:
                    print(f"      - {reason}: {count}")
            print(f"{'='*60}\n")
            
            if len(pairs) == 0:
                raise RuntimeError("❌ 没有有效的训练数据！请检查图片路径")
            
            return pairs

    
    def generate_candidates_for_round(
        self,
        components: Dict,
        dataset: List[Dict],
        round_num: int
    ) -> List[Dict]:
        """
        为一轮训练生成所有候选对
        
        【修复版】：添加断点恢复机制
        """
        if self.args.candidate_num_shards > 1:
            total_before = len(dataset)
            dataset = [s for i, s in enumerate(dataset)
                       if i % self.args.candidate_num_shards == self.args.candidate_shard_id]
            print(f"  候选生成分片: shard {self.args.candidate_shard_id}/{self.args.candidate_num_shards}, "
                  f"{len(dataset)}/{total_before}")
        print(f"\n{'='*60}")
        print(f"Round {round_num}: 生成候选CoT")
        print(f"{'='*60}")
        
        vla = components['multitask_vla']
        scorer = components['diagnosis_scorer']
        tokenizer = components['tokenizer']
        
        candidates_save_path = self.save_dir / f"round_{round_num}_candidates.json"
        diagnosis_save_path = self.save_dir / f"round_{round_num}_diagnosis_records.json"
        progress_save_path = self.save_dir / f"round_{round_num}_progress.json"
        
        pairs = []
        diagnosis_records = []
        start_idx = 0
        

        # ⭐ 断点恢复检查
        if progress_save_path.exists():
            print(f"\n🔄 发现进度文件，尝试恢复...")
            
            try:
                with open(progress_save_path, 'r') as f:
                    progress = json.load(f)
                
                saved_idx = progress.get('completed_samples', 0)
                
                # ⭐ 关键修改：如果已全部完成，进入"训练准备模式"
                if saved_idx >= len(dataset):
                    print(f"   ✅ 候选生成已完成（{saved_idx}/{len(dataset)}）")
                    print(f"   进入训练准备模式，重新加载图片...")
                    
                    # 调用新函数准备训练数据
                    return self._prepare_pairs_for_training(
                        candidates_path=candidates_save_path,
                        diagnosis_path=diagnosis_save_path,
                        dataset=dataset,
                        components=components,
                        round_num=round_num
                    )
                
                # 否则，继续生成（原来的恢复逻辑）
                if candidates_save_path.exists() and diagnosis_save_path.exists():
                    with open(candidates_save_path, 'r') as f:
                        pairs = json.load(f)
                    
                    with open(diagnosis_save_path, 'r') as f:
                        diagnosis_records = json.load(f)
                    
                    
                    if len(pairs) > 0 and len(diagnosis_records) > 0:
                        start_idx = saved_idx
                        
                        print(f"   ✅ 成功恢复！已完成 {start_idx}/{len(dataset)} 个样本")
                        print(f"   继续从样本 {start_idx} 开始...")
                    else:
                        print(f"   ⚠️  数据不一致，从头开始")
                        pairs = []
                        diagnosis_records = []
                        start_idx = 0
                else:
                    print(f"   ⚠️  数据文件不完整，从头开始")
            
            except Exception as e:
                print(f"   ⚠️  恢复失败: {e}")
                print(f"   从头开始...")
                pairs = []
                diagnosis_records = []
                start_idx = 0

        
        if start_idx >= len(dataset):
            print(f"✅ 所有候选已生成完毕，直接返回")
            return pairs
        
        vla.eval()
        
        for i in tqdm(range(start_idx, len(dataset)), desc="生成候选", initial=start_idx, total=len(dataset)):
            sample = dataset[i]
            
            try:
                episode_folder = sample.get('image_path') or sample.get('episode_id')
                frame_name = sample.get('frame_idx')
                
                if not frame_name and 'index_list' in sample and len(sample['index_list']) > 0:
                    frame_name = sample['index_list'][0]
                
                pixel_values_for_pair = None
                if episode_folder and frame_name:
                    try:
                        from PIL import Image
                        base_root = Path(self.config.image_base_path)
                        folder_path = base_root / episode_folder
                        
                        img_path = None
                        for ext in ['.png', '.jpg', '.jpeg']:
                            test_path = folder_path / f"{frame_name}{ext}"
                            if test_path.exists():
                                img_path = test_path
                                break
                        
                        if img_path and img_path.exists():
                            img = Image.open(img_path).convert('RGB')
                            tr_img = vla.base_vla.vision_backbone.image_transform(img)
                            
                            pixel_values_for_pair = {}
                            for k in tr_img.keys():
                                combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0)
                                pixel_values_for_pair[k] = combined.unsqueeze(0).to(self.device)
                    
                    except Exception as img_e:
                        if i < 5:
                            print(f"\n  ⚠️  样本 {i} 图像加载失败: {img_e}")
                
                candidates = self.generate_candidates_simple(
                    vla,
                    sample,
                    tokenizer,
                    num_candidates=self.config.num_candidates
                )
                if len(candidates) == 0:
                    print(f"\n  ⚠️ 样本 {i} 图片缺失，跳过")
                    continue
                
                instruction = sample.get('gpt_instruction') or sample.get('instruction') or ''
                sample_id = sample.get('sample_id') or sample.get('frame_idx') or str(i)
                
                chosen, rejected, diagnosis_info = self._rank_candidates_with_scorer(
                    candidates=candidates,
                    sample=sample,
                    scorer=scorer,
                    frame_idx=frame_name,
                    sample_idx=i
                )
                
                diagnosis_records.append({
                    'sample_idx': i,
                    'sample_id': sample_id,
                    'instruction': instruction,
                    'chosen_score': diagnosis_info['chosen_score'],
                    'rejected_score': diagnosis_info['rejected_score'],
                    'score_gap': diagnosis_info['score_gap'],
                    'error_type': diagnosis_info.get('error_type', 'unknown'),
                    'error_step': diagnosis_info.get('error_step', -1),
                    'confidence': diagnosis_info.get('confidence', 'low'),
                    'fallback': diagnosis_info.get('fallback', False)
                })
                
                pair = {
                    'chosen': chosen,
                    'rejected': rejected,
                    'pixel_values': pixel_values_for_pair,
                    'instruction': instruction,
                    'prompt_len': len(tokenizer(
                        f"What action should the robot take to {instruction.lower()}?",
                        return_tensors="pt"
                    ).input_ids[0]),
                    'aux_labels': sample.get('aux_labels'),
                    'sample_id': sample_id
                }
                
                pairs.append(pair)
                
                # ⭐ 每50个样本保存一次（更频繁）
                if (i + 1) % 50 == 0:
                    self._save_checkpoint_with_progress(
                        pairs=pairs,
                        diagnosis_records=diagnosis_records,
                        completed=i + 1,
                        candidates_path=candidates_save_path,
                        diagnosis_path=diagnosis_save_path,
                        progress_path=progress_save_path
                    )
                
                if i == 99:
                    self._early_analysis(diagnosis_records[:100], round_num)
                
            except Exception as e:
                print(f"\n  ❌ 样本 {i} 生成失败: {e}")
                import traceback
                traceback.print_exc()
                
                diagnosis_records.append({
                    'sample_idx': i,
                    'sample_id': str(i),
                    'instruction': '',
                    'chosen_score': 0.0,
                    'rejected_score': 0.0,
                    'score_gap': 0.0,
                    'error_type': 'generation_failed',
                    'error_step': -1,
                    'confidence': 'none',
                    'fallback': True,
                    'error_message': str(e)
                })
                
                pairs.append({
                    'chosen': None,
                    'rejected': None,
                    'pixel_values': None,
                    'instruction': '',
                    'prompt_len': 0,
                    'aux_labels': None,
                    'sample_id': str(i),
                    'generation_failed': True
                })
                
                continue
        
        self._save_checkpoint_with_progress(
            pairs=pairs,
            diagnosis_records=diagnosis_records,
            completed=len(dataset),
            candidates_path=candidates_save_path,
            diagnosis_path=diagnosis_save_path,
            progress_path=progress_save_path
        )
        
        print(f"\n✅ 生成完成: {len(pairs)} 个候选对")
        print(f"   保存至: {candidates_save_path}")
        
        failed_count = len([p for p in pairs if p.get('generation_failed', False)])
        if failed_count > 0:
            print(f"   ⚠️  生成失败: {failed_count} 个样本")
        
        print(f"{'='*60}\n")
        
        self._final_analysis(diagnosis_records, round_num, dataset, pairs)
        
        return pairs
    

    def train_one_round(
            self,
            trainer: MultiTaskGSPOTrainer,
            pairs: List[Dict],
            tokenizer,
            round_num: int
        ) -> List[Dict]:
            """训练一轮"""
            print(f"\n{'='*60}")
            print(f"Round {round_num}: GSPO训练")
            print(f"{'='*60}")

            # ⭐ 安全注入：确保 aux_labels 不因断点恢复而丢失
            aux_labels_path = Path(self.config.get_aux_labels_path(round_num))
            if aux_labels_path.exists():
                with open(aux_labels_path, 'r') as f:
                    aux_data = json.load(f)
                samples_list = aux_data if isinstance(aux_data, list) else aux_data.get('samples', [])
                aux_dict = {}
                for item in samples_list:
                    sid = str(item.get('sample_id') or item.get('sample_idx', ''))
                    if sid and item.get('aux_labels'):
                        aux_dict[sid] = item['aux_labels']
                injected = 0
                for pair in pairs:
                    sid = str(pair.get('sample_id', ''))
                    if sid in aux_dict and pair.get('aux_labels') is None:
                        pair['aux_labels'] = aux_dict[sid]
                        injected += 1
                if injected > 0:
                    print(f"  ⭐ 安全注入 aux_labels: {injected} 个 pairs 补充了辅助标签")
            
            # ⭐ 修改：过滤掉生成失败的样本 和 pixel_values为None的样本
            valid_pairs = [
                p for p in pairs 
                if not p.get('generation_failed', False) 
                and p.get('pixel_values') is not None
            ]
            
            print(f"有效训练样本数: {len(valid_pairs)}/{len(pairs)}")
            print(f"训练步数: {self.config.steps_per_round}")
            print(f"Batch size: {self.config.batch_size}")
            
            if len(valid_pairs) == 0:
                raise RuntimeError("❌ 没有有效的训练样本！")
            
            if len(valid_pairs) < self.config.batch_size:
                print(f"⚠️  警告：有效样本数({len(valid_pairs)})小于batch_size({self.config.batch_size})")
                print(f"   将使用所有可用样本")
            
            print(f"{'='*60}\n")
            
            stats_history = []
            empty_batch_count = 0
            max_empty_batches = 10  # 连续10个空batch就报错
            
            for step in tqdm(range(self.config.steps_per_round), desc=f"Round {round_num}"):
                batch_size = min(self.config.batch_size, len(valid_pairs))
                batch_pairs = random.sample(valid_pairs, batch_size)
                
                batch = trainer.prepare_batch(batch_pairs, tokenizer)
                
                # ⭐ 新增：防护空batch
                if batch is None:
                    empty_batch_count += 1
                    if empty_batch_count >= max_empty_batches:
                        raise RuntimeError(f"❌ 连续 {max_empty_batches} 个空batch，训练终止！请检查数据")
                    print(f"  ⚠️ Step {step}: batch为空，跳过 (连续{empty_batch_count}次)")
                    continue
                
                empty_batch_count = 0  # 重置计数
                
                stats = trainer.train_step(batch)
                stats['round'] = round_num
                stats['step'] = step
                
                stats_history.append(stats)
                self.training_log.append(stats)
                
                if step % self.config.log_interval == 0:
                    self._print_stats(stats, round_num, step)
                
                if step > 0 and step % self.args.save_interval == 0:
                    ckpt_path = self.save_dir / f"round_{round_num}_step_{step}"
                    trainer.save_checkpoint(ckpt_path, round_num)
                    self._cleanup_old_checkpoints(round_num)
            
            return stats_history
    
    
    def _print_stats(self, stats: Dict, round_num: int, step: int):
        """打印训练统计"""
        print(f"\n[Round {round_num} | Step {step}]")
        print(f"  Total Loss: {stats['total_loss']:.4f}")
        print(f"    ├─ GSPO Loss: {stats['gspo_loss']:.4f}")
        print(f"    │   ├─ Accuracy: {stats['gspo_accuracy']:.2%}")
        print(f"    │   └─ Margin: {stats['gspo_margin']:.4f}")
        print(f"    └─ Aux Loss: {stats['aux_total_loss']:.4f}")
        
        if stats['keyword_loss'] > 0:
            print(f"        ├─ Keyword: {stats['keyword_loss']:.4f}")
        if stats['direction_loss'] > 0:
            print(f"        ├─ Direction: {stats['direction_loss']:.4f}")
        if stats['quality_loss'] > 0:
            print(f"        ├─ Quality: {stats['quality_loss']:.4f}")
        if stats['validity_loss'] > 0:
            print(f"        └─ Validity: {stats['validity_loss']:.4f}")
        
        print(f"  Grad Norm: {stats['grad_norm']:.4f}")
    
    def _cleanup_old_checkpoints(self, current_round: int):
        """清理旧的中间checkpoint"""
        keep_last = self.args.keep_checkpoints
        
        pattern = f"round_{current_round}_step_*"
        checkpoints = sorted(
            self.save_dir.glob(pattern),
            key=lambda x: int(x.name.split('_')[-1])
        )
        
        if len(checkpoints) > keep_last:
            for old_ckpt in checkpoints[:-keep_last]:
                shutil.rmtree(old_ckpt)
                print(f"  清理旧checkpoint: {old_ckpt.name}")
    
    def save_training_log(self):
        """保存训练日志"""
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        print(f"✓ 训练日志已保存: {self.log_file}")
    
    def run(self):
        """运行完整训练流程"""
        start_time = datetime.now()
        components = self.initialize_components()

        for round_num in range(self.config.num_rounds):
            dataset = self.load_data(round_num=round_num)

            final_ckpt = self.save_dir / f"round_{round_num}_final" / "adapter_model.safetensors"
            if final_ckpt.exists() and (not self.args.test):
                print(f"\n⏭️ Round {round_num} 已完成，跳过")
                continue

            print(f"\n{'#'*60}")
            print(f"# Round {round_num + 1}/{self.config.num_rounds}")
            print(f"{'#'*60}")

            if round_num == 0 or not self.args.reuse_candidates:
                pairs = self.generate_candidates_for_round(components, dataset, round_num)
            else:
                print(f"\n复用上一轮候选对 (Round {round_num-1} -> Round {round_num})")
                prev_round = round_num - 1
                candidates_path = self.save_dir / f"round_{prev_round}_candidates.json"
                diagnosis_path = self.save_dir / f"round_{prev_round}_diagnosis_records.json"
                if not diagnosis_path.exists():
                    diagnosis_path = self.save_dir / f"round_{prev_round}_diagnosis.json"

                if candidates_path.exists() and diagnosis_path.exists():
                    pairs = self._prepare_pairs_for_training(
                        candidates_path=candidates_path,
                        diagnosis_path=diagnosis_path,
                        dataset=dataset,
                        components=components,
                        round_num=round_num
                    )
                    if len(pairs) == 0:
                        raise RuntimeError("复用后有效pairs为0")
                else:
                    raise FileNotFoundError(f"缺少复用文件: {candidates_path} 或 {diagnosis_path}")

            self.train_one_round(
                components['trainer'],
                pairs,
                components['tokenizer'],
                round_num
            )

            final_ckpt_path = self.save_dir / f"round_{round_num}_final"
            components['trainer'].save_checkpoint(final_ckpt_path, round_num)
            self.save_training_log()

        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n{'='*60}")
        print("✅ Stage 2训练完成！")
        print(f"{'='*60}")
        print(f"总耗时: {duration}")
        print(f"保存目录: {self.save_dir}")
        print(f"训练日志: {self.log_file}")
        print(f"{'='*60}\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Stage 2 Multi-task GSPO Training")
    
    parser.add_argument('--test', action='store_true',
                        help='测试模式（少量steps）')
    parser.add_argument('--reuse_candidates', action='store_true',
                        help='复用第一轮的候选（快速测试）')
    
    parser.add_argument('--save_interval', type=int, default=400,
                        help='每N步保存checkpoint')
    parser.add_argument('--keep_checkpoints', type=int, default=3,
                        help='保留最近N个中间checkpoint')
    
    parser.add_argument('--resume_from', type=str, default=None,
                        help='从checkpoint恢复（暂未实现）')
    
    parser.add_argument('--candidate_num_shards', type=int, default=1,
                        help='候选生成分片总数')
    parser.add_argument('--candidate_shard_id', type=int, default=0,
                        help='当前分片编号，从0开始')
    args = parser.parse_args()
    
    config = Stage2Config()
    
    if args.test:
        print("⚠️  测试模式：快速测试")
        config.num_rounds = 1
        config.steps_per_round = 10
        config.num_candidates = 2
        config.temperatures = [0.7, 1.5]
        config.max_new_tokens = 128
    
    config.print_config()
    
    stage2_trainer = Stage2Trainer(config, args)
    stage2_trainer.run()


if __name__ == '__main__':
    main()
