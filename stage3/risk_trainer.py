"""
Risk-Aware GSPO Trainer (RiskTrainer)

基于 stage2/multitask_gspo_trainer.py 扩展，新增：
- _prepare_aux_labels 里提取 chosen_score 和 root_cause_label
- train_step 里把这两个字段传给 RiskLoss.compute_total_loss
- save_checkpoint 里保存 risk heads 权重
- 打印 stats 时显示 risk 相关字段
"""

import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional
from pathlib import Path
import json
import os
from torch.cuda.amp import autocast, GradScaler

# ===== 路径设置 =====
algorithm_path = '/home/ubuntu/data1/lyy/full_rlds_project-3/algorithms'
training_path = '/home/ubuntu/data1/lyy/full_rlds_project-3/algorithms/gspo'

for p in [algorithm_path, training_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"正在加载 GSPO 模块...")
from gspo.sequence_logprob import compute_sequence_logprob, IGNORE_INDEX
print("✅ GSPO 模块加载成功")

SEQUENCE_LOGPROB_AVAILABLE = True


class RiskTrainer:
    """
    Risk-Aware GSPO Trainer

    相比 stage2 MultiTaskGSPOTrainer 的变化：
    1. _prepare_aux_labels 额外提取 chosen_score 和 root_cause_label
    2. train_step 把这两个字段传给 RiskLoss
    3. save_checkpoint 保存6个heads（含2个risk heads）
    """

    def __init__(
        self,
        risk_vla,        # RiskAwareVLA
        risk_loss,       # RiskLoss
        config,          # Stage3Config
        device: str = "cuda"
    ):
        self.vla = risk_vla
        self.risk_loss = risk_loss
        self.config = config
        self.device = device

        # 优化器：优先用8-bit Adam节省显存
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                filter(lambda p: p.requires_grad, self.vla.parameters()),
                lr=config.learning_rate
            )
            print(f"✅ 使用 8-bit AdamW 优化器")
        except ImportError:
            print(f"⚠️  bitsandbytes 未安装，使用标准 AdamW")
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.vla.parameters()),
                lr=config.learning_rate
            )

        self.scaler = GradScaler()
        self.stats_history = []
        self._global_step = 0

        print(f"\n{'='*60}")
        print("RiskTrainer 初始化完成")
        print(f"{'='*60}")
        print(f"学习率:     {config.learning_rate}")
        print(f"Beta:       {config.beta}")
        print(f"梯度裁剪:   {config.max_grad_norm}")
        print(f"alpha:      {config.alpha}")
        print(f"gamma:      {config.gamma}")
        print(f"mu:         {config.mu}")
        print(f"{'='*60}\n")

    def prepare_batch(
        self,
        pairs: List[Dict],
        tokenizer
    ) -> Optional[Dict]:
        """
        准备训练 batch（和 stage2 完全一致，图像处理逻辑不变）
        """
        image_transform = self.vla.base_vla.vision_backbone.image_transform

        # === 1. 处理图像 ===
        pixel_values_list = []
        valid_pairs_indices = []

        for i, pair in enumerate(pairs):
            try:
                image_input = pair.get('pixel_values', pair.get('image'))
                if image_input is None:
                    continue

                if isinstance(image_input, dict) and len(image_input) > 0:
                    first_value = next(iter(image_input.values()))
                    if isinstance(first_value, torch.Tensor):
                        single_pixel_values = {}
                        for k, v in image_input.items():
                            if v.dim() == 4:
                                single_pixel_values[k] = v.squeeze(0).to(self.device)
                            else:
                                single_pixel_values[k] = v.to(self.device)
                        pixel_values_list.append(single_pixel_values)
                        valid_pairs_indices.append(i)
                        continue

                if isinstance(image_input, str):
                    from PIL import Image
                    if os.path.exists(image_input):
                        image = Image.open(image_input).convert('RGB')
                    else:
                        full_path = f"/home/ubuntu/data1/lyy/full_rlds_project/images/{image_input}"
                        image = Image.open(full_path).convert('RGB')
                elif hasattr(image_input, 'convert'):
                    image = image_input
                else:
                    raise ValueError(f"不支持的图片类型: {type(image_input)}")

                tr_img = image_transform(image)
                single_pixel_values = {}
                for k in tr_img.keys():
                    combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0)
                    single_pixel_values[k] = combined.to(self.device)

                pixel_values_list.append(single_pixel_values)
                valid_pairs_indices.append(i)

            except Exception as e:
                print(f"  ⚠️ Pair {i} 图像处理失败: {e}")
                continue

        if not pixel_values_list:
            return None

        pixel_values = {}
        for k in pixel_values_list[0].keys():
            pixel_values[k] = torch.stack([pv[k] for pv in pixel_values_list])

        valid_pairs = [pairs[i] for i in valid_pairs_indices]

        # === 2. Tokenize ===
        chosen_input_ids_list = []
        chosen_labels_list = []
        rejected_input_ids_list = []
        rejected_labels_list = []

        for pair in valid_pairs:
            # Chosen
            if 'generated_text' in pair['chosen']:
                chosen_text = pair['chosen']['generated_text']
            elif 'text' in pair['chosen']:
                chosen_text = pair['chosen']['text']
            else:
                c_thinking = pair['chosen'].get('thinking', '')
                c_action = pair['chosen'].get('predicted_action', '')
                chosen_text = f"<thinking>{c_thinking}</thinking><action>{c_action}</action>"

            chosen_ids = tokenizer(
                chosen_text, add_special_tokens=True,
                truncation=True, max_length=512,
                return_tensors="pt"
            ).input_ids[0]

            prompt_len = pair.get('prompt_len', 0)
            chosen_labels = chosen_ids.clone()
            chosen_labels[:prompt_len] = -100

            chosen_input_ids_list.append(chosen_ids)
            chosen_labels_list.append(chosen_labels)

            # Rejected
            if 'generated_text' in pair['rejected']:
                rejected_text = pair['rejected']['generated_text']
            elif 'text' in pair['rejected']:
                rejected_text = pair['rejected']['text']
            else:
                r_thinking = pair['rejected'].get('thinking', '')
                r_action = pair['rejected'].get('predicted_action', '')
                rejected_text = f"<thinking>{r_thinking}</thinking><action>{r_action}</action>"

            rejected_ids = tokenizer(
                rejected_text, add_special_tokens=True,
                truncation=True, max_length=512,
                return_tensors="pt"
            ).input_ids[0]

            rejected_labels = rejected_ids.clone()
            rejected_labels[:prompt_len] = -100

            rejected_input_ids_list.append(rejected_ids)
            rejected_labels_list.append(rejected_labels)

        # === 3. Padding ===
        chosen_input_ids = pad_sequence(
            chosen_input_ids_list, batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(self.device)

        chosen_labels = pad_sequence(
            chosen_labels_list, batch_first=True,
            padding_value=-100
        ).to(self.device)

        rejected_input_ids = pad_sequence(
            rejected_input_ids_list, batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(self.device)

        rejected_labels = pad_sequence(
            rejected_labels_list, batch_first=True,
            padding_value=-100
        ).to(self.device)

        # === 4. 辅助标签（含 stage3 新增字段）===
        aux_labels, chosen_score_tensor, root_cause_tensor = \
            self._prepare_aux_labels(valid_pairs)

        return {
            'pixel_values': pixel_values,
            'chosen_input_ids': chosen_input_ids,
            'chosen_labels': chosen_labels,
            'rejected_input_ids': rejected_input_ids,
            'rejected_labels': rejected_labels,
            'aux_labels': aux_labels,
            # stage3 新增
            'chosen_score': chosen_score_tensor,      # [batch, 1] 或 None
            'root_cause_label': root_cause_tensor     # [batch] 或 None
        }

    def _prepare_aux_labels(
        self,
        pairs: List[Dict]
    ):
        """
        从 pairs 中提取辅助标签

        返回三个值：
        - aux_labels: 4个辅助任务标签（全部有才返回，否则None）
        - chosen_score_tensor: [batch, 1] 全部样本都返回
        - root_cause_tensor: [batch] 全部样本都有error_type才返回，否则None

        error_type 映射：
            perception    → 0
            comprehension → 1
            reasoning     → 2
            decision      → 3
        """
        error_type_map = {
            "perception": 0,
            "comprehension": 1,
            "reasoning": 2,
            "decision": 3
        }

        # ===== Stage2 原有4个辅助标签 =====
        if not all(p.get('aux_labels') is not None for p in pairs):
            aux_labels = None
        else:
            keywords_list, direction_list, quality_list, validity_list = [], [], [], []
            for pair in pairs:
                aux = pair['aux_labels']
                keywords_list.append(torch.tensor(aux['keywords'], dtype=torch.float32))
                direction_list.append(torch.tensor(aux['direction'], dtype=torch.long))
                quality_list.append(torch.tensor([aux['cot_quality']], dtype=torch.float32))
                validity_list.append(torch.tensor([aux['action_validity']], dtype=torch.float32))

            aux_labels = {
                'keywords': torch.stack(keywords_list).to(self.device),
                'direction': torch.stack(direction_list).to(self.device),
                'cot_quality': torch.stack(quality_list).to(self.device),
                'action_validity': torch.stack(validity_list).to(self.device)
            }

        # ===== Stage3 新增：chosen_score（全部样本都提取）=====
        chosen_score_list = []
        for pair in pairs:
            score = pair.get('chosen_score', None)
            if score is not None:
                chosen_score_list.append(float(score))
            else:
                chosen_score_list.append(0.5)  # 无分数时用中间值

        chosen_score_tensor = torch.tensor(
            chosen_score_list, dtype=torch.float32
        ).unsqueeze(1).to(self.device)  # [batch, 1]

        # ===== Stage3 新增：root_cause_label（仅弱样本有，全部有才返回）=====
        root_cause_list = []
        all_have_error_type = True

        for pair in pairs:
            error_type = pair.get('error_type', None)
            if error_type is None or error_type == 'unknown':
                all_have_error_type = False
                break
            label = error_type_map.get(error_type, -1)
            if label == -1:
                all_have_error_type = False
                break
            root_cause_list.append(label)

        if all_have_error_type and len(root_cause_list) == len(pairs):
            root_cause_tensor = torch.tensor(
                root_cause_list, dtype=torch.long
            ).to(self.device)  # [batch]
        else:
            root_cause_tensor = None  # 非弱样本batch，risk_loss_2跳过

        return aux_labels, chosen_score_tensor, root_cause_tensor

    def _compute_sequence_logprob_fallback(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Fallback版本的log概率计算（和stage2完全一致）"""
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        labels_expanded = labels.unsqueeze(-1)
        selected_log_probs = log_probs_all.gather(
            dim=-1, index=labels_expanded).squeeze(-1)
        mask = (labels != IGNORE_INDEX).float()
        selected_log_probs = selected_log_probs * mask
        return selected_log_probs.sum(dim=1)

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        单步训练

        相比 stage2 的变化：
        - 从 batch 里取出 chosen_score 和 root_cause_label
        - 传给 risk_loss.compute_total_loss
        """
        self.vla.train()

        # 截断超长序列（和stage2一致）
        max_len = 400
        for key in ['chosen', 'rejected']:
            input_key = f'{key}_input_ids'
            label_key = f'{key}_labels'
            if batch[input_key].shape[1] > max_len:
                keep_front = int(max_len * 0.4)
                keep_back = max_len - keep_front
                batch[input_key] = torch.cat([
                    batch[input_key][:, :keep_front],
                    batch[input_key][:, -keep_back:]
                ], dim=1)
                batch[label_key] = torch.cat([
                    batch[label_key][:, :keep_front],
                    batch[label_key][:, -keep_back:]
                ], dim=1)

        with autocast(dtype=torch.float16):
            # === 1. Forward Chosen（获取辅助输出）===
            chosen_outputs = self.vla(
                pixel_values=batch['pixel_values'],
                input_ids=batch['chosen_input_ids'],
                return_aux_outputs=True
            )
            chosen_logits = chosen_outputs['logits']
            chosen_aux_outputs = chosen_outputs.get('aux_outputs', None)

            # 如果是fallback零值输出，跳过辅助loss
            if chosen_aux_outputs is not None:
                if chosen_aux_outputs.get('_is_fallback', False):
                    chosen_aux_outputs = None

            # 计算 chosen log prob
            if SEQUENCE_LOGPROB_AVAILABLE:
                chosen_log_probs = compute_sequence_logprob(
                    chosen_logits, batch['chosen_labels'])
            else:
                chosen_log_probs = self._compute_sequence_logprob_fallback(
                    chosen_logits, batch['chosen_labels'])

            # === 2. Forward Rejected ===
            rejected_outputs = self.vla(
                pixel_values=batch['pixel_values'],
                input_ids=batch['rejected_input_ids'],
                return_aux_outputs=False
            )
            rejected_logits = rejected_outputs['logits']

            if SEQUENCE_LOGPROB_AVAILABLE:
                rejected_log_probs = compute_sequence_logprob(
                    rejected_logits, batch['rejected_labels'])
            else:
                rejected_log_probs = self._compute_sequence_logprob_fallback(
                    rejected_logits, batch['rejected_labels'])

            chosen_lengths = (batch['chosen_labels'] != -100).sum(dim=1).float()
            rejected_lengths = (batch['rejected_labels'] != -100).sum(dim=1).float()

            # === 3. 计算 Total Loss ===
            total_loss, loss_stats = self.risk_loss.compute_total_loss(
                chosen_log_probs=chosen_log_probs,
                rejected_log_probs=rejected_log_probs,
                chosen_aux_outputs=chosen_aux_outputs,
                aux_labels=batch.get('aux_labels'),
                beta=self.config.beta,
                chosen_lengths=chosen_lengths,
                rejected_lengths=rejected_lengths,
                # stage3 新增
                normalized_chosen_score=batch.get('chosen_score'),
                root_cause_label=batch.get('root_cause_label'),
                current_step=self._global_step,
                warmup_steps=self.config.warmup_steps
            )

        # === 4. 反向传播 ===
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)

        # 修3：分组梯度裁剪
        # risk heads 单独裁剪（更严格），防止随机初始化早期梯度爆炸
        risk_params = (
            list(self.vla.risk_head_1.parameters()) +
            list(self.vla.risk_head_2.parameters())
        )
        base_params = [p for p in self.vla.parameters()
                       if p.requires_grad and
                       not any(p is rp for rp in risk_params)]

        risk_grad_norm = torch.nn.utils.clip_grad_norm_(
            risk_params, max_norm=0.5)   # risk heads 更严格
        base_grad_norm = torch.nn.utils.clip_grad_norm_(
            base_params, max_norm=self.config.max_grad_norm)

        # 取两组中较大的作为监控值
        grad_norm = max(risk_grad_norm, base_grad_norm)

        if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
            self.scaler.step(self.optimizer)
        else:
            print(f"  ⚠️  梯度异常 (risk={risk_grad_norm:.3f}, "
                  f"base={base_grad_norm:.3f})，跳过本步")
        self.scaler.update()

        self._global_step = getattr(self, '_global_step', 0) + 1
        stats = {**loss_stats, 'grad_norm': grad_norm.item()}
        self.stats_history.append(stats)

        del chosen_outputs, rejected_outputs
        del chosen_logits, rejected_logits
        torch.cuda.empty_cache()

        return stats

    def save_checkpoint(self, save_path: Path, round_num: int):
        """
        保存 checkpoint

        保存内容：
        - LoRA 权重
        - projector.pt
        - auxiliary_heads.pt（6个heads，含2个risk heads）
        - round_X_stats.json
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. LoRA 权重
        try:
            self.vla.base_vla.llm_backbone.llm.save_pretrained(save_path)
        except Exception as e:
            print(f"⚠️  save_pretrained失败: {e}")
            lora_state = {
                k: v for k, v in
                self.vla.base_vla.llm_backbone.llm.state_dict().items()
                if 'lora' in k.lower()
            }
            torch.save(lora_state, save_path / "adapter_model.bin")

        # 2. Projector
        if hasattr(self.vla.base_vla, 'projector'):
            torch.save(
                self.vla.base_vla.projector.state_dict(),
                save_path / "projector.pt"
            )

        # 3. 所有辅助heads（6个，含risk heads）
        torch.save({
            'keyword_head': self.vla.keyword_head.state_dict(),
            'direction_head': self.vla.direction_head.state_dict(),
            'cot_quality_head': self.vla.cot_quality_head.state_dict(),
            'action_validity_head': self.vla.action_validity_head.state_dict(),
            'risk_head_1': self.vla.risk_head_1.state_dict(),
            'risk_head_2': self.vla.risk_head_2.state_dict(),
        }, save_path / "auxiliary_heads.pt")

        # 4. 训练统计
        stats_file = save_path / f"round_{round_num}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats_history, f, indent=2)

        print(f"✅ Checkpoint 保存到: {save_path}")
        print(f"   - LoRA 权重")
        print(f"   - projector.pt")
        print(f"   - auxiliary_heads.pt（6个heads）")
        print(f"   - round_{round_num}_stats.json")
