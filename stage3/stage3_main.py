#!/home/ubuntu/miniconda3/envs/lyy-openfly/bin/python
"""
Stage 3 Risk-Aware GSPO 主训练脚本

基于 stage2/stage2_gspo_main.py 扩展，新增：
1. RiskAwareVLA（从stage2权重加载，含risk_head_1和risk_head_2）
2. RiskLoss（含risk_loss_1和risk_loss_2，全局归一化）
3. 组pair时额外存 chosen_score 和 error_type
4. 每轮训练结束后自动重新生成 auxiliary_labels_round{N+1}.json
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

# ===== 路径设置 =====
PROJECT_ROOT = Path('/home/ubuntu/data1/lyy/full_rlds_project-3')
STAGE3_DIR = PROJECT_ROOT / 'stage3'
STAGE2_DIR = PROJECT_ROOT / 'stage2'
OPENFLY_PATH = '/home/ubuntu/data1/lyy/OpenFly-Platform/train'

for p in [str(STAGE3_DIR), str(STAGE2_DIR), str(PROJECT_ROOT), OPENFLY_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("=" * 60)
print("Stage 3 Risk-Aware GSPO 训练")
print("=" * 60)

# ===== 导入模块 =====
try:
    from risk_model import RiskAwareVLA
    print("✓ 导入 risk_model.RiskAwareVLA")
except Exception as e:
    print(f"✗ 导入 risk_model 失败: {e}")
    import traceback; traceback.print_exc(); sys.exit(1)

try:
    from risk_loss import RiskLoss
    print("✓ 导入 risk_loss.RiskLoss")
except Exception as e:
    print(f"✗ 导入 risk_loss 失败: {e}")
    import traceback; traceback.print_exc(); sys.exit(1)

try:
    from risk_trainer import RiskTrainer
    print("✓ 导入 risk_trainer.RiskTrainer")
except Exception as e:
    print(f"✗ 导入 risk_trainer 失败: {e}")
    import traceback; traceback.print_exc(); sys.exit(1)

try:
    from stage3_config import Stage3Config
    print("✓ 导入 stage3_config.Stage3Config")
except Exception as e:
    print(f"✗ 导入 stage3_config 失败: {e}")
    import traceback; traceback.print_exc(); sys.exit(1)

try:
    from diagnosis_scorer import DiagnosisScorer
    print("✓ 导入 diagnosis_scorer.DiagnosisScorer")
except Exception as e:
    print(f"✗ 导入 diagnosis_scorer 失败: {e}")
    import traceback; traceback.print_exc(); sys.exit(1)

try:
    from auxiliary_labeler import AuxiliaryLabeler
    print("✓ 导入 auxiliary_labeler.AuxiliaryLabeler")
except Exception as e:
    print(f"✗ 导入 auxiliary_labeler 失败: {e}")
    import traceback; traceback.print_exc(); sys.exit(1)

print("=" * 60)
print("✓ 所有模块导入成功")
print("=" * 60)


class Stage3Trainer:
    """Stage 3 Risk-Aware GSPO 训练管理器"""

    def __init__(self, config: Stage3Config, args):
        self.config = config
        # 环境变量覆盖 save_dir（分片用）
        import os as _os
        _env_save = _os.environ.get('STAGE3_SAVE_DIR')
        if _env_save:
            self.config.save_dir = _env_save
            print(f"  [分片模式] save_dir 覆盖为: {_env_save}")
        self.args = args
        self.device = config.device

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.training_log = []
        self.log_file = self.save_dir / "training_log.json"

        print(f"\n{'='*60}")
        print("Stage 3 Risk-Aware GSPO 训练初始化")
        print(f"{'='*60}")
        print(f"保存目录: {self.save_dir}")
        print(f"设备: {self.device}")
        print(f"轮数: {config.num_rounds}")
        print(f"每轮步数: {config.steps_per_round}")
        print(f"Batch size: {config.batch_size}")
        print(f"{'='*60}\n")

    # ==================== 工具方法（和stage2一致）====================

    @staticmethod
    def _parse_generated_cot(generated_text: str) -> tuple:
        import re
        thinking_text = ""
        action_number = None
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', generated_text, re.DOTALL)
        if thinking_match:
            thinking_text = thinking_match.group(1).strip()
        action_match = re.search(r'<action>(\d+)</action>', generated_text)
        if action_match:
            action_number = int(action_match.group(1))
        else:
            action_match = re.search(r'<next_action>(\d+)</next_action>', generated_text)
            if action_match:
                action_number = int(action_match.group(1))
        return thinking_text, action_number

    @staticmethod
    def _extract_ground_truth(sample: Dict, frame_idx: str) -> Dict:
        gt_cot = ""
        gt_actions = []
        cot_dict = sample.get('cot', {})
        index_list = sample.get('index_list', [])
        action_list = sample.get('action', [])
        try:
            current_idx = index_list.index(frame_idx)
        except (ValueError, AttributeError):
            return {'cot': gt_cot, 'actions': gt_actions}
        if current_idx < len(index_list) - 1:
            next_frame = index_list[current_idx + 1]
            cot_key = f"{frame_idx}-{next_frame}"
            if cot_key in cot_dict:
                cot_text = cot_dict[cot_key]
                gt_cot, _ = Stage3Trainer._parse_generated_cot(cot_text)
        if current_idx < len(action_list):
            gt_actions = action_list[current_idx:current_idx + 10]
        return {'cot': gt_cot, 'actions': gt_actions}

    def _atomic_save(self, data, filepath: Path):
        temp_path = filepath.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            shutil.move(str(temp_path), str(filepath))
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def _rank_candidates_with_scorer(
        self,
        candidates: List[Dict],
        sample: Dict,
        scorer,
        frame_idx: str,
        sample_idx: int = -1
    ) -> tuple:
        gt = self._extract_ground_truth(sample, frame_idx)

        formatted_candidates = []
        for cand in candidates:
            thinking, action = self._parse_generated_cot(cand.get('generated_only', ''))
            formatted_candidates.append({
                'cot': thinking,
                'actions': [action] if action is not None else [],
                'original': cand
            })

        try:
            result = scorer.rank_candidates(
                candidates=formatted_candidates,
                gt=gt,
                sample={
                    'instruction': sample.get('gpt_instruction') or sample.get('instruction', '')
                },
                return_all_scores=True
            )

            chosen = result['chosen']['original'].copy()
            rejected = result['rejected']['original'].copy()
            chosen['score'] = result['chosen'].get('score', 0.0)
            rejected['score'] = result['rejected'].get('score', 0.0)

            diagnosis_info = {
                'chosen_score': chosen['score'],
                'rejected_score': rejected['score'],
                'score_gap': chosen['score'] - rejected['score'],
                'gt_cot': gt.get('cot', ''),
                'all_candidates': []
            }

            chosen_diagnosis = result['chosen'].get('diagnosis', {})
            if chosen_diagnosis:
                diagnosis_info['error_type'] = chosen_diagnosis.get('error_type', 'unknown')
                diagnosis_info['error_step'] = chosen_diagnosis.get('error_step', -1)
                diagnosis_info['confidence'] = chosen_diagnosis.get('confidence', 'low')
            else:
                diagnosis_info['error_type'] = 'unknown'
                diagnosis_info['error_step'] = -1
                diagnosis_info['confidence'] = 'low'

            if sample_idx >= 0 and sample_idx < 5:
                print(f"\n  ✓ 样本 {sample_idx} 评分完成:")
                print(f"     Chosen: score={chosen['score']:.3f}")
                print(f"     Rejected: score={rejected['score']:.3f}")
                print(f"     Gap: {diagnosis_info['score_gap']:.3f}")
                print(f"     Error Type: {diagnosis_info['error_type']}")

        except Exception as e:
            print(f"   ⚠️  DiagnosisScorer评分失败: {e}")
            import traceback; traceback.print_exc()

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

    # ==================== 数据加载 ====================

    def load_data(self, round_num: int) -> List[Dict]:
        """加载训练数据，读取当前轮次对应的 aux_labels"""
        print(f"\n{'='*60}")
        print(f"加载训练数据 (Round {round_num})")
        print(f"{'='*60}")

        with open(self.config.data_path, 'r') as f:
            train_data = json.load(f)
        print(f"  ✓ 加载了 {len(train_data)} 个样本")

        # ⭐ 使用模板路径读取当前轮次的 aux_labels
        aux_labels_path = self.config.get_aux_labels_path(round_num)
        aux_labels_dict = {}

        if Path(aux_labels_path).exists():
            print(f"加载辅助标签: {aux_labels_path}")
            with open(aux_labels_path, 'r') as f:
                aux_data = json.load(f)

            # 兼容两种格式：list 或 dict with 'samples' key
            samples_list = aux_data if isinstance(aux_data, list) \
                else aux_data.get('samples', [])

            for item in samples_list:
                sample_id = str(item.get('sample_id') or item.get('sample_idx', ''))
                if sample_id:
                    aux_labels_dict[sample_id] = item.get('aux_labels')

            print(f"  ✓ 加载了 {len(aux_labels_dict)} 个样本的辅助标签")
        else:
            print(f"  ⚠️  未找到辅助标签文件: {aux_labels_path}")
            print(f"      将只使用 GSPO loss 训练")

        dataset = []
        samples_with_labels = 0

        for idx, sample in enumerate(train_data):
            sample_id = str(idx)  # ⭐ 用索引号匹配（和stage2修复一致）
            if sample_id in aux_labels_dict:
                sample['aux_labels'] = aux_labels_dict[sample_id]
                samples_with_labels += 1
            else:
                sample['aux_labels'] = None
            dataset.append(sample)

        if self.args.test:
            dataset = dataset[:10]
            print(f"\n⚠️  测试模式：数据集缩减到 {len(dataset)} 个样本")

        print(f"\n数据统计:")
        print(f"  总样本数: {len(dataset)}")
        print(f"  有辅助标签: {samples_with_labels}")
        print(f"  无辅助标签: {len(dataset) - samples_with_labels}")
        print(f"{'='*60}\n")

        return dataset

    # ==================== 组件初始化 ====================

    def initialize_components(self) -> Dict:
        """初始化所有训练组件"""
        print(f"\n{'='*60}")
        print("初始化训练组件")
        print(f"{'='*60}")

        # [1] RiskAwareVLA（从stage2权重加载）
        print("\n[1/5] 初始化 RiskAwareVLA...")
        risk_vla = RiskAwareVLA(
            stage2_checkpoint_dir=self.config.stage2_model_path,
            num_unfrozen_layers=4,
            num_keywords=34,
            hidden_dim=4096,
            dropout=0.1,
            verbose=True
        )
        risk_vla = risk_vla.to(self.device)
        # compat_use_mor_v2_start
        _patched = False
        for path_name in ["base_vla.llm_backbone", "llm_backbone", "vla.llm_backbone", "base_vla.vla.llm_backbone"]:
            try:
                obj = risk_vla
                for part in path_name.split("."):
                    obj = getattr(obj, part)
                if not hasattr(obj, "use_mor"):
                    obj.use_mor = False
                if not hasattr(obj, "use_mora"):
                    obj.use_mora = False
                print(f"[compat] use_mor/use_mora set on risk_vla.{path_name}")
                _patched = True
                break
            except Exception:
                continue

        if not _patched:
            for name, mod in risk_vla.named_modules():
                if "LLaMa2LLMBackbone" in type(mod).__name__:
                    if not hasattr(mod, "use_mor"):
                        mod.use_mor = False
                    if not hasattr(mod, "use_mora"):
                        mod.use_mora = False
                    print(f"[compat] use_mor/use_mora set on module: {name}")
                    _patched = True
                    break

        if not _patched:
            raise RuntimeError("[compat] 未找到 LLaMa2LLMBackbone，无法注入 use_mor/use_mora")
        # compat_use_mor_v2_end

        # codex_dtype_guard_v2
        try:
            if hasattr(risk_vla, "config"):
                risk_vla.config.use_cache = False
        except Exception:
            pass
        total_n, train_n = 0, 0
        for name, param in risk_vla.named_parameters():
            total_n += param.numel()
            if param.requires_grad:
                if param.dtype != torch.float32:
                    param.data = param.data.float()
                train_n += param.numel()
            else:
                if param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)
        if train_n == 0:
            raise RuntimeError("dtype_guard后无可训练参数，请检查冻结策略")
        print(f"[dtype_guard] trainable params: {train_n/1e6:.2f}M / {total_n/1e6:.2f}M")


        # [2] 全局归一化参数（训练开始前统计一次）
        print("\n[2/5] 计算全局 chosen_score 归一化参数...")
        aux_labels_path = self.config.get_aux_labels_path(0)
        if Path(aux_labels_path).exists():
            score_min, score_max = RiskLoss.compute_global_score_stats(aux_labels_path)
        else:
            print(f"  ⚠️  未找到 aux_labels，使用默认归一化范围 [0, 1]")
            score_min, score_max = 0.0, 1.0

        # [3] RiskLoss
        print("\n[3/5] 初始化 RiskLoss...")
        risk_loss = RiskLoss(
            lambda_keyword=0.15,
            lambda_direction=0.1,
            lambda_quality=0.1,
            lambda_validity=0.1,
            alpha=self.config.alpha,
            gamma=self.config.gamma,
            mu=self.config.mu,
            score_global_min=score_min,
            score_global_max=score_max,
            device=self.device,
            verbose=True
        )

        # [4] RiskTrainer
        print("\n[4/5] 初始化 RiskTrainer...")
        trainer = RiskTrainer(
            risk_vla=risk_vla,
            risk_loss=risk_loss,
            config=self.config,
            device=self.device
        )

        # [5] 诊断框架
        print("\n[5/5] 初始化诊断框架...")
        from root_cause_locator import RootCauseLocator
        root_cause_locator = RootCauseLocator(
            train_data_path="/home/ubuntu/data1/zx/1OpenFly-Platform/OpenFly-Platform/dataset/Annotation/train.json",
            verbose=False
        )
        diagnosis_scorer = DiagnosisScorer(
            cot_parser=root_cause_locator.cot_parser,
            root_cause_locator=root_cause_locator,
            error_penalties={'perception': 0.4, 'comprehension': 0.3,
                           'reasoning': 0.2, 'decision': 0.1}
        )

        tokenizer = risk_vla.base_vla.llm_backbone.tokenizer

        print(f"\n{'='*60}")
        print("✅ 所有组件初始化完成")
        print(f"{'='*60}\n")

        return {
            'risk_vla': risk_vla,
            'risk_loss': risk_loss,
            'trainer': trainer,
            'diagnosis_scorer': diagnosis_scorer,
            'tokenizer': tokenizer
        }

    # ==================== 候选生成 ====================

    def generate_candidates_simple(
        self,
        vla,
        sample: Dict,
        tokenizer,
        num_candidates: int = 3
    ) -> List[Dict]:
        """生成候选CoT（和stage2完全一致）"""
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
                        f_path = folder_path / f"{frame_name}{ext}" if ext \
                            else folder_path / frame_name
                        if f_path.exists():
                            target_file = f_path
                            break
                if target_file is None and folder_path.exists():
                    all_imgs = sorted(
                        list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")))
                    if all_imgs:
                        target_file = all_imgs[0]
                if target_file and target_file.exists():
                    img = Image.open(target_file).convert('RGB')
                    tr_img = image_transform(img)
                    temp_pixel_values = {}
                    for k in tr_img.keys():
                        combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0)
                        temp_pixel_values[k] = combined.unsqueeze(0).to(self.device)
                    pixel_values = temp_pixel_values
            except Exception as e:
                pixel_values = None

        if pixel_values is None:
            return []

        candidates = []
        temperatures = self.config.temperatures[:num_candidates]
        prompt_ids = tokenizer(
            prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        vocab_size = tokenizer.vocab_size

        for temp in temperatures:
            current_ids = prompt_ids.clone()
            try:
                with torch.no_grad():
                    for _ in range(self.config.max_new_tokens):
                        outputs = base_model(
                            pixel_values=pixel_values, input_ids=current_ids)
                        next_token_logits = outputs.logits[:, -1, :]
                        effective_vocab = min(vocab_size, next_token_logits.shape[-1])
                        next_token_logits = next_token_logits[:, :effective_vocab]
                        if temp > 0:
                            probs = torch.softmax(next_token_logits / temp, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(
                                next_token_logits, dim=-1, keepdim=True)
                        if next_token.item() >= vocab_size or next_token.item() < 0:
                            break
                        current_ids = torch.cat([current_ids, next_token], dim=1)
                        if next_token.item() == tokenizer.eos_token_id:
                            break

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
                print(f"      ⚠️ 温度{temp}生成失败: {e}")
                candidates.append({
                    'generated_text': prompt_text,
                    'generated_only': '',
                    'temperature': temp
                })

        return candidates

    def generate_candidates_for_round(
        self,
        components: Dict,
        dataset: List[Dict],
        round_num: int
    ) -> List[Dict]:
        """为一轮训练生成所有候选对（含断点恢复）"""
        print(f"\n{'='*60}")
        print(f"Round {round_num}: 生成候选CoT")
        print(f"{'='*60}")

        vla = components['risk_vla']
        scorer = components['diagnosis_scorer']
        tokenizer = components['tokenizer']

        candidates_save_path = self.save_dir / f"round_{round_num}_candidates.json"
        diagnosis_save_path = self.save_dir / f"round_{round_num}_diagnosis_records.json"
        progress_save_path = self.save_dir / f"round_{round_num}_progress.json"

        pairs = []
        diagnosis_records = []
        start_idx = 0

        # 断点恢复
        if progress_save_path.exists():
            print(f"\n🔄 发现进度文件，尝试恢复...")
            try:
                with open(progress_save_path, 'r') as f:
                    progress = json.load(f)
                saved_idx = progress.get('completed_samples', 0)

                if saved_idx >= len(dataset):
                    print(f"   ✅ 候选生成已完成，进入训练准备模式...")
                    return self._prepare_pairs_for_training(
                        candidates_path=candidates_save_path,
                        diagnosis_path=diagnosis_save_path,
                        dataset=dataset,
                        components=components,
                        round_num=round_num
                    )

                if candidates_save_path.exists() and diagnosis_save_path.exists():
                    with open(candidates_save_path, 'r') as f:
                        pairs = json.load(f)
                    with open(diagnosis_save_path, 'r') as f:
                        diagnosis_records = json.load(f)
                    if len(pairs) >= saved_idx:
                        start_idx = saved_idx
                        pairs = pairs[:saved_idx]
                        diagnosis_records = diagnosis_records[:saved_idx]
                        print(f"   ✅ 恢复成功，从样本 {start_idx} 继续...")
            except Exception as e:
                print(f"   ⚠️  恢复失败: {e}，从头开始")
                pairs = []
                diagnosis_records = []
                start_idx = 0

        vla.eval()
        vla.float()  # 推理统一fp32

        for i in tqdm(range(start_idx, len(dataset)),
                      desc="生成候选", initial=start_idx, total=len(dataset)):
            # stage3_shard_skip_v1
            if hasattr(self, 'args') and self.args.candidate_num_shards > 1:
                if i % self.args.candidate_num_shards != self.args.candidate_shard_id:
                    continue
            sample = dataset[i]
            try:
                episode_folder = sample.get('image_path') or sample.get('episode_id')
                frame_name = sample.get('frame_idx')
                if not frame_name and 'index_list' in sample and sample['index_list']:
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
                    except Exception:
                        pass

                candidates = self.generate_candidates_simple(
                    vla, sample, tokenizer,
                    num_candidates=self.config.num_candidates
                )
                if not candidates:
                    print(f"\n  ⚠️ 样本 {i} 图片缺失，跳过")
                    continue

                instruction = sample.get('gpt_instruction') or sample.get('instruction') or ''
                sample_id = str(i)  # ⭐ 用索引号

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
                    'sample_id': sample_id,
                    # ⭐ stage3 新增：存入 chosen_score 和 error_type
                    'chosen_score': diagnosis_info['chosen_score'],
                    'error_type': diagnosis_info.get('error_type', 'unknown'),
                }
                pairs.append(pair)

                if (i + 1) % 50 == 0:
                    self._save_checkpoint_with_progress(
                        pairs=pairs,
                        diagnosis_records=diagnosis_records,
                        completed=i + 1,
                        candidates_path=candidates_save_path,
                        diagnosis_path=diagnosis_save_path,
                        progress_path=progress_save_path
                    )

            except Exception as e:
                print(f"\n  ❌ 样本 {i} 生成失败: {e}")
                import traceback; traceback.print_exc()
                diagnosis_records.append({
                    'sample_idx': i, 'sample_id': str(i), 'instruction': '',
                    'chosen_score': 0.0, 'rejected_score': 0.0, 'score_gap': 0.0,
                    'error_type': 'generation_failed', 'error_step': -1,
                    'confidence': 'none', 'fallback': True
                })
                pairs.append({
                    'chosen': None, 'rejected': None, 'pixel_values': None,
                    'instruction': '', 'prompt_len': 0, 'aux_labels': None,
                    'sample_id': str(i), 'generation_failed': True,
                    'chosen_score': 0.0, 'error_type': 'unknown'
                })

        self._save_checkpoint_with_progress(
            pairs=pairs, diagnosis_records=diagnosis_records,
            completed=len(dataset),
            candidates_path=candidates_save_path,
            diagnosis_path=diagnosis_save_path,
            progress_path=progress_save_path
        )

        print(f"\n✅ 生成完成: {len(pairs)} 个候选对")
        # 推理完毕，冻结层转回 bfloat16 为训练节省显存
        vla.train()
        for name, param in vla.named_parameters():
            if not param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
        print("  ✓ 冻结层已转回 bfloat16")
        return pairs

    def _save_checkpoint_with_progress(
        self, pairs, diagnosis_records, completed,
        candidates_path, diagnosis_path, progress_path
    ):
        if not pairs:
            return
        pairs_to_save = [{k: v for k, v in p.items() if k != 'pixel_values'}
                         for p in pairs]
        self._atomic_save(pairs_to_save, candidates_path)
        self._atomic_save(diagnosis_records, diagnosis_path)
        self._atomic_save({
            'completed_samples': completed,
            'total_pairs': len(pairs),
            'timestamp': datetime.now().isoformat()
        }, progress_path)
        print(f"\n  💾 检查点已保存: {completed} 个样本完成")

    def _prepare_pairs_for_training(
        self, candidates_path, diagnosis_path, dataset, components, round_num
    ) -> List[Dict]:
        """从已保存的候选数据重新加载图片，准备训练"""
        print(f"\n从已保存候选数据准备训练...")

        with open(candidates_path, 'r') as f:
            saved_pairs = json.load(f)
        with open(diagnosis_path, 'r') as f:
            diagnosis_records = json.load(f)

        id_to_idx = {str(r['sample_id']): r['sample_idx'] for r in diagnosis_records
                     if 'sample_id' in r and 'sample_idx' in r}

        vla = components['risk_vla']
        image_transform = vla.base_vla.vision_backbone.image_transform
        pairs = []

        from PIL import Image
        for pair in tqdm(saved_pairs, desc="加载图片"):
            if pair.get('generation_failed', False):
                continue
            sample_idx = id_to_idx.get(str(pair.get('sample_id', '')))
            if sample_idx is None or sample_idx >= len(dataset):
                continue
            sample = dataset[sample_idx]
            try:
                episode_folder = sample.get('image_path') or sample.get('episode_id')
                frame_name = sample.get('frame_idx')
                if not frame_name and 'index_list' in sample and sample['index_list']:
                    frame_name = sample['index_list'][0]
                if not episode_folder or not frame_name:
                    continue
                base_root = Path(self.config.image_base_path)
                folder_path = base_root / episode_folder
                img_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = folder_path / f"{frame_name}{ext}"
                    if test_path.exists():
                        img_path = test_path
                        break
                if img_path is None:
                    continue
                img = Image.open(img_path).convert('RGB')
                tr_img = image_transform(img)
                pixel_values = {}
                for k in tr_img.keys():
                    combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0)
                    pixel_values[k] = combined.unsqueeze(0).to(self.device)
                pairs.append({**pair, 'pixel_values': pixel_values})
            except Exception:
                continue

        print(f"✅ 有效训练 pairs: {len(pairs)}")
        return pairs

    # ==================== 训练 ====================

    def train_one_round(
        self,
        trainer: RiskTrainer,
        pairs: List[Dict],
        tokenizer,
        round_num: int
    ) -> List[Dict]:
        """训练一轮"""
        print(f"\n{'='*60}")
        print(f"Round {round_num}: GSPO训练")


        # stage3_aux_autofill_hook_v1
        if round_num >= 1:
            try:
                import subprocess, sys, os
                hook = os.path.join(os.path.dirname(__file__), "auto_fill_aux_labels_stage3.py")
                subprocess.run([sys.executable, hook, "--round", str(round_num)], check=True)
            except Exception as e:
                print(f"  ⚠️ stage3 aux auto-fill 失败: {e}")
            if round_num >= 1:
                try:
                    import subprocess, sys, os
                    hook = os.path.join(os.path.dirname(__file__), "auto_fill_aux_labels_stage3.py")
                    subprocess.run([sys.executable, hook, "--round", str(round_num)], check=True)
                except Exception as e:
                    print(f"  ⚠️ stage3 aux auto-fill 失败: {e}")
        print(f"{'='*60}")

        # ⭐ 安全注入：确保 aux_labels 不因断点恢复而丢失
        aux_labels_path = self.config.get_aux_labels_path(round_num)
        if Path(aux_labels_path).exists():
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

        valid_pairs = [p for p in pairs
                       if not p.get('generation_failed', False)
                       and p.get('pixel_values') is not None]

        print(f"有效训练样本数: {len(valid_pairs)}/{len(pairs)}")
        if not valid_pairs:
            raise RuntimeError("❌ 没有有效的训练样本！")

        stats_history = []
        empty_batch_count = 0

        for step in tqdm(range(self.config.steps_per_round), desc=f"Round {round_num}"):
            batch_size = min(self.config.batch_size, len(valid_pairs))
            batch_pairs = random.sample(valid_pairs, batch_size)
            batch = trainer.prepare_batch(batch_pairs, tokenizer)

            if batch is None:
                empty_batch_count += 1
                if empty_batch_count >= 10:
                    raise RuntimeError("❌ 连续10个空batch，训练终止！")
                continue
            empty_batch_count = 0

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
        """打印训练统计（含 risk 相关字段）"""
        print(f"\n[Round {round_num} | Step {step}]")
        print(f"  Total Loss: {stats.get('total_loss', 0):.4f}")
        print(f"    ├─ GSPO Loss:    {stats.get('gspo_loss', 0):.4f}  "
              f"(acc={stats.get('gspo_accuracy', 0):.2%}, "
              f"margin={stats.get('gspo_margin', 0):.4f})")
        print(f"    ├─ Aux Loss:     {stats.get('aux_total_loss', 0):.4f}")
        if stats.get('keyword_loss', 0) > 0:
            print(f"    │   ├─ Keyword:   {stats['keyword_loss']:.4f}")
        if stats.get('direction_loss', 0) > 0:
            print(f"    │   ├─ Direction: {stats['direction_loss']:.4f}")
        if stats.get('quality_loss', 0) > 0:
            print(f"    │   ├─ Quality:   {stats['quality_loss']:.4f}")
        if stats.get('validity_loss', 0) > 0:
            print(f"    │   └─ Validity:  {stats['validity_loss']:.4f}")
        # ⭐ stage3 新增 risk 打印
        print(f"    ├─ Risk Loss:    {stats.get('risk_loss_total', 0):.4f}  "
              f"(weight={stats.get('risk_weight', 1.0):.3f})")
        if stats.get('risk_loss_1', 0) > 0:
            print(f"    │   ├─ Risk_1 (MSE): {stats['risk_loss_1']:.4f}")
        if stats.get('risk_loss_2', 0) > 0:
            print(f"    │   └─ Risk_2 (CE):  {stats['risk_loss_2']:.4f}")
        print(f"    └─ Grad Norm:   {stats.get('grad_norm', 0):.4f}")

    # ==================== ⭐ Stage3 新增：重新生成 aux_labels ====================

    def _regenerate_aux_labels(
        self,
        components: Dict,
        dataset: List[Dict],
        round_num: int
    ):
        """
        训练结束后，用当前模型重新生成下一轮的 aux_labels

        流程：
        1. 用当前模型重新生成候选CoT
        2. DiagnosisScorer重新打分，得到新的 chosen_score 和 error_type
        3. 找出新的弱样本（分数最低35%）
        4. 对每个弱样本生成完整4字段 aux_labels（调Qwen）
        5. 保存为 auxiliary_labels_round{round_num+1}.json
        """
        next_round = round_num + 1
        output_path = Path(self.config.get_aux_labels_path(next_round))

        print(f"\n{'='*60}")
        print(f"重新生成 auxiliary_labels_round{next_round}.json")
        print(f"{'='*60}")

        if output_path.exists():
            print(f"  ⚠️  已存在，跳过: {output_path}")
            return

        vla = components['risk_vla']
        scorer = components['diagnosis_scorer']
        tokenizer = components['tokenizer']

        # === Step 1: 重新生成候选并打分 ===
        print(f"\n[Step 1/4] 重新生成候选CoT并打分...")
        vla.eval()
        vla.float()  # 推理统一fp32
        new_diagnosis_records = []

        for i in tqdm(range(len(dataset)), desc="重新打分"):
            sample = dataset[i]
            instruction = sample.get('gpt_instruction') or sample.get('instruction') or ''
            frame_name = sample.get('frame_idx')
            if not frame_name and 'index_list' in sample and sample['index_list']:
                frame_name = sample['index_list'][0]

            try:
                candidates = self.generate_candidates_simple(
                    vla, sample, tokenizer,
                    num_candidates=self.config.num_candidates
                )
                if not candidates:
                    new_diagnosis_records.append({
                        'sample_idx': i, 'sample_id': str(i),
                        'instruction': instruction,
                        'chosen_score': 0.5, 'error_type': 'unknown'
                    })
                    continue

                _, _, diagnosis_info = self._rank_candidates_with_scorer(
                    candidates=candidates,
                    sample=sample,
                    scorer=scorer,
                    frame_idx=frame_name
                )

                new_diagnosis_records.append({
                    'sample_idx': i,
                    'sample_id': str(i),
                    'instruction': instruction,
                    'chosen_score': diagnosis_info['chosen_score'],
                    'error_type': diagnosis_info.get('error_type', 'unknown'),
                    'confidence': diagnosis_info.get('confidence', 'low'),
                    'model_cot': self._parse_generated_cot(
                        candidates[0].get('generated_only', ''))[0],
                    'model_actions': [self._parse_generated_cot(
                        candidates[0].get('generated_only', ''))[1]]
                        if self._parse_generated_cot(
                            candidates[0].get('generated_only', ''))[1] is not None else [],
                    'gt_cot': self._extract_ground_truth(sample, frame_name).get('cot', ''),
                    'gt_actions': self._extract_ground_truth(sample, frame_name).get('actions', [])
                })
            except Exception as e:
                new_diagnosis_records.append({
                    'sample_idx': i, 'sample_id': str(i),
                    'instruction': instruction,
                    'chosen_score': 0.5, 'error_type': 'unknown'
                })

        # === Step 2: 找出新的弱样本（分数最低35%）===
        print(f"\n[Step 2/4] 找出新的弱样本...")
        sorted_records = sorted(new_diagnosis_records, key=lambda x: x['chosen_score'])
        weak_count = int(len(sorted_records) * 0.35)
        weak_records = sorted_records[:weak_count]
        print(f"  弱样本数量: {weak_count}/{len(sorted_records)}")

        # === Step 3: 初始化 AuxiliaryLabeler ===
        print(f"\n[Step 3/4] 初始化 AuxiliaryLabeler...")
        labeler = AuxiliaryLabeler(
            train_dataset_path=self.config.data_path,

            qwen_api_base="http://localhost:9998",

            qwen_model_name="Qwen3-VL-32B-Instruct",
            extract_keywords=True
        )

        # === Step 4: 对弱样本生成完整4字段 aux_labels ===
        print(f"\n[Step 4/4] 为弱样本生成完整4字段 aux_labels...")
        labeled_samples = []

        for record in tqdm(weak_records, desc="生成aux_labels"):
            instruction = record.get('instruction', '')
            model_cot = record.get('model_cot', '')
            model_actions = record.get('model_actions', [])
            gt_cot = record.get('gt_cot', '')

            try:
                # 4个字段全部生成，不按error_type只生成一个
                aux_labels = {
                    'keywords': labeler.label_keywords(instruction),
                    'direction': labeler.label_direction(instruction),
                    'cot_quality': labeler.label_cot_quality(
                        model_cot, gt_cot, instruction),  # 可能调Qwen
                    'action_validity': labeler.label_action_validity(
                        model_actions, instruction)
                }
            except Exception as e:
                print(f"  ⚠️  样本 {record['sample_idx']} aux_labels生成失败: {e}")
                aux_labels = {
                    'keywords': [0] * len(labeler.keyword_index),
                    'direction': -1,
                    'cot_quality': 0.5,
                    'action_validity': 0.5
                }

            labeled_samples.append({
                'sample_idx': record['sample_idx'],
                'sample_id': record['sample_id'],
                'sample': {
                    'instruction': instruction,
                    'image_path': dataset[record['sample_idx']].get('image_path', ''),
                },
                'chosen_score': record['chosen_score'],
                'error_type': record['error_type'],
                'confidence': record.get('confidence', 'low'),
                'aux_labels': aux_labels
            })

        # === 保存 ===
        output_data = {
            'total_samples': len(labeled_samples),
            'round': next_round,
            'generated_at': datetime.now().isoformat(),
            'weak_threshold': f"lowest {int(0.35*100)}%",
            'qwen_calls': labeler.stats.get('qwen_calls', 0),
            'samples': labeled_samples
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ auxiliary_labels_round{next_round}.json 生成完成")
        print(f"   弱样本数量: {len(labeled_samples)}")
        print(f"   Qwen调用次数: {labeler.stats.get('qwen_calls', 0)}")
        print(f"   保存路径: {output_path}")
        print(f"{'='*60}\n")

    # ==================== 主流程 ====================

    def _cleanup_old_checkpoints(self, current_round: int):
        import re
        checkpoints = sorted(
            self.save_dir.glob(f"round_{current_round}_step_*"),
            key=lambda x: int(x.name.split('_')[-1])
        )
        if len(checkpoints) > self.args.keep_checkpoints:
            for old_ckpt in checkpoints[:-self.args.keep_checkpoints]:
                shutil.rmtree(old_ckpt)

    def save_training_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        print(f"✓ 训练日志已保存: {self.log_file}")

    def run(self):
        """运行完整训练流程"""
        start_time = datetime.now()
        components = self.initialize_components()

        for round_num in range(self.config.num_rounds):

            # 跳过已完成的轮次
            final_ckpt = self.save_dir / f"round_{round_num}_final" / "adapter_model.safetensors"
            if final_ckpt.exists():
                print(f"\n⏭️ Round {round_num} 已完成，跳过")
                continue

            print(f"\n{'#'*60}")
            print(f"# Round {round_num + 1}/{self.config.num_rounds}")
            print(f"{'#'*60}")

            # ⭐ 每轮重新加载对应的 aux_labels
            dataset = self.load_data(round_num)

            # 生成候选
            pairs = self.generate_candidates_for_round(components, dataset, round_num)

            # 训练
            if getattr(self.args, 'candidate_only', False):
                print(f"  [candidate_only] Round {round_num} 候选生成完毕，跳过训练")
                return

            self.train_one_round(components['trainer'], pairs,
                                 components['tokenizer'], round_num)

            # 保存 checkpoint
            final_ckpt_path = self.save_dir / f"round_{round_num}_final"
            components['trainer'].save_checkpoint(final_ckpt_path, round_num)
            print(f"✅ Round {round_num} checkpoint: {final_ckpt_path}")

            self.save_training_log()

            # ⭐ 重新生成下一轮的 aux_labels（最后一轮不需要）
            if round_num < self.config.num_rounds - 1:
                self._regenerate_aux_labels(components, dataset, round_num)

        end_time = datetime.now()
        print(f"\n{'='*60}")
        print("✅ Stage 3 训练完成！")
        print(f"{'='*60}")
        print(f"总耗时: {end_time - start_time}")
        print(f"保存目录: {self.save_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Risk-Aware GSPO Training")
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--save_interval', type=int, default=400)
    parser.add_argument('--keep_checkpoints', type=int, default=3)
    parser.add_argument('--candidate_num_shards', type=int, default=1, help='候选生成分片总数')
    parser.add_argument('--candidate_shard_id', type=int, default=0, help='当前分片ID')
    parser.add_argument('--candidate_only', action='store_true', help='只生成候选，不训练')
    args = parser.parse_args()

    config = Stage3Config()

    if args.test:
        print("⚠️  测试模式")
        config.num_rounds = 1
        config.steps_per_round = 10
        config.num_candidates = 2
        config.temperatures = [0.7, 1.5]
        config.max_new_tokens = 256

    config.print_config()

    trainer = Stage3Trainer(config, args)
    trainer.run()


if __name__ == '__main__':
    main()
