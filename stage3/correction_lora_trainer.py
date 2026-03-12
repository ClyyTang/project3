"""
Correction LoRA 离线训练脚本 (B2)

策略：
- 基于 stage3 最终权重，单独训一个小 LoRA（rank=4）
- 用 SFT loss（不用 GSPO），只学高危场景的脱险 CoT
- 训完后权重独立保存，不修改 stage3 权重
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime
import argparse

PROJECT_ROOT = Path('/home/ubuntu/data1/lyy/full_rlds_project-3')
for p in [str(PROJECT_ROOT / 'stage3'), str(PROJECT_ROOT / 'stage2'),
          str(PROJECT_ROOT), '/home/ubuntu/data1/lyy/OpenFly-Platform/train']:
    if p not in sys.path:
        sys.path.insert(0, p)

from stage3_config import Stage3Config
from risk_model import RiskAwareVLA


# ==================== Dataset ====================

class CorrectionLoRADataset(Dataset):
    """
    Correction LoRA 训练数据集

    每个样本：图片 + 指令 + 反事实脱险 CoT（Qwen生成）
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_transform,
        image_base_path: str,
        max_length: int = 512,
        verbose: bool = True
    ):
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.image_base_path = Path(image_base_path)
        self.max_length = max_length

        with open(data_path, 'r') as f:
            raw = json.load(f)

        samples = raw.get('samples', raw) if isinstance(raw, dict) else raw
        self.samples = [s for s in samples
                        if s.get('correction_cot') and
                        s.get('correction_action') is not None]

        if verbose:
            print(f"✅ CorrectionLoRADataset: {len(self.samples)} 个有效样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        instruction = item.get('instruction', '')
        correction_cot = item.get('correction_cot', '')
        correction_action = item.get('correction_action', 0)

        # 加载图片
        pixel_values = None
        img_path = item.get('image_path', '')
        if img_path and Path(img_path).exists():
            try:
                from PIL import Image
                img = Image.open(img_path).convert('RGB')
                tr_img = self.image_transform(img)
                pixel_values = {}
                for k in tr_img.keys():
                    combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0)
                    pixel_values[k] = combined
            except Exception:
                pixel_values = None

        # 构造目标序列: correction_cot + <action>N</action>
        target_text = (
            f"{correction_cot}\n<action>{correction_action}</action>"
        )

        # Tokenize
        input_ids = self.tokenizer(
            f"What action should the robot take to {instruction.lower()}?",
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors="pt"
        ).input_ids[0]

        label_ids = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors="pt"
        ).input_ids[0]

        # 拼接：prompt + target，label 里 prompt 部分设为 -100
        full_ids = torch.cat([input_ids, label_ids], dim=0)
        labels = torch.cat([
            torch.full_like(input_ids, -100),
            label_ids
        ], dim=0)

        return {
            'input_ids': full_ids,
            'labels': labels,
            'pixel_values': pixel_values,
            'risk_score': item.get('risk_score', 0.7),
            'error_type': item.get('error_type', 'unknown'),
        }


def collate_fn(batch):
    """自定义 collate，处理不定长序列和可能缺失的图片"""
    # 找最长序列
    max_len = max(b['input_ids'].shape[0] for b in batch)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for b in batch:
        seq_len = b['input_ids'].shape[0]
        pad_len = max_len - seq_len

        input_ids = torch.cat([
            b['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        labels = torch.cat([
            b['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        attention_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ])

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    result = {
        'input_ids': torch.stack(input_ids_list),
        'labels': torch.stack(labels_list),
        'attention_mask': torch.stack(attention_mask_list),
        'risk_scores': torch.tensor([b['risk_score'] for b in batch]),
    }

    # pixel_values：只有全部都有图片才合并
    pv_list = [b['pixel_values'] for b in batch if b['pixel_values'] is not None]
    if len(pv_list) == len(batch):
        combined_pv = {}
        for k in pv_list[0].keys():
            combined_pv[k] = torch.stack([pv[k] for pv in pv_list])
        result['pixel_values'] = combined_pv
    else:
        result['pixel_values'] = None

    return result


# ==================== 训练器 ====================

class CorrectionLoRATrainer:
    """
    Correction LoRA 训练器

    核心设计：
    - 在 stage3 权重上挂一个新的小 LoRA（rank=4）
    - 只训这个新 LoRA，stage3 的所有权重全部冻结
    - SFT loss（CrossEntropy on target tokens）
    """

    def __init__(
        self,
        stage3_checkpoint_dir: str,
        data_path: str,
        save_dir: str,
        config: Stage3Config,
        lora_rank: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 2,
        grad_accum_steps: int = 4,
        max_grad_norm: float = 1.0,
        device: str = "cuda:0"
    ):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.lora_rank = lora_rank

        print(f"\n{'='*60}")
        print("Correction LoRA 训练器初始化")
        print(f"{'='*60}")
        print(f"Stage3 权重: {stage3_checkpoint_dir}")
        print(f"LoRA rank:   {lora_rank} (stage3用的是8，这里用4)")
        print(f"Epochs:      {num_epochs}")
        print(f"Batch size:  {batch_size} x 梯度累积{grad_accum_steps}")
        print(f"LR:          {learning_rate}")
        print(f"{'='*60}")

        # [1] 加载 stage3 模型（全部冻结）
        print("\n[1/3] 加载 RiskAwareVLA（stage3 权重，全部冻结）...")
        self.model = RiskAwareVLA(
            stage2_checkpoint_dir=stage3_checkpoint_dir,
            num_unfrozen_layers=0,
            verbose=False
        ).to(device)

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # [2] 挂载 Correction LoRA
        print("\n[2/3] 挂载 Correction LoRA (rank=4)...")
        self._attach_correction_lora(lora_rank, lora_alpha, lora_dropout)

        # [3] 数据集
        print("\n[3/3] 加载训练数据...")
        tokenizer = self.model.base_vla.llm_backbone.tokenizer
        image_transform = self.model.base_vla.vision_backbone.image_transform

        dataset = CorrectionLoRADataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_transform=image_transform,
            image_base_path=config.image_base_path,
            verbose=True
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        self.tokenizer = tokenizer

        # 优化器（只优化 Correction LoRA 参数）
        correction_params = [p for n, p in self.model.named_parameters()
                             if 'correction_lora' in n and p.requires_grad]
        print(f"\n可训练参数: {sum(p.numel() for p in correction_params):,}")

        try:
            from bitsandbytes.optim import AdamW8bit
            self.optimizer = AdamW8bit(correction_params, lr=learning_rate)
            print("✅ 使用 8-bit AdamW")
        except Exception:
            self.optimizer = torch.optim.AdamW(correction_params, lr=learning_rate)
            print("✅ 使用标准 AdamW")

        print(f"\n✅ 初始化完成，训练样本数: {len(dataset)}")

    def _attach_correction_lora(
        self,
        rank: int,
        alpha: int,
        dropout: float
    ):
        """
        在 LLM backbone 的 attention 层挂载 Correction LoRA

        原理：给每个 q_proj/v_proj 增加一对低秩矩阵 A, B
        correction_lora_output = original_output + (alpha/rank) * x @ A @ B
        """
        from peft import get_peft_model, LoraConfig, TaskType

        llm = self.model.base_vla.llm_backbone.llm

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            # 用 modules_to_save 区分 correction lora 和原有 lora
        )

        # 用 peft 挂载新 LoRA
        self.model.base_vla.llm_backbone.llm = get_peft_model(llm, lora_config)

        # 只让新加的 LoRA 参数可训练
        for name, param in self.model.base_vla.llm_backbone.llm.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                # 重命名标记（便于保存时区分）
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in
                        self.model.base_vla.llm_backbone.llm.parameters()
                        if p.requires_grad)
        print(f"  ✅ Correction LoRA 挂载完成，可训练参数: {trainable:,}")

    def _compute_sft_loss(self, batch: Dict) -> torch.Tensor:
        """SFT loss：只在 target tokens 上计算 CrossEntropy"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        pixel_values = batch.get('pixel_values')

        if pixel_values is not None:
            pv = {k: v.to(self.device) for k, v in pixel_values.items()}
        else:
            pv = None

        outputs = self.model(
            pixel_values=pv,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.get('logits')
        if logits is None:
            raise RuntimeError("模型没有返回 logits")

        # Shift：logits[:-1] 预测 labels[1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return loss

    def train(self):
        """训练主循环"""
        print(f"\n{'='*60}")
        print("开始训练 Correction LoRA")
        print(f"{'='*60}\n")

        self.model.train()
        global_step = 0
        best_loss = float('inf')
        log = []

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            self.optimizer.zero_grad()

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for step, batch in enumerate(pbar):
                try:
                    loss = self._compute_sft_loss(batch)

                    # 梯度累积
                    (loss / self.grad_accum_steps).backward()

                    epoch_loss += loss.item()
                    num_batches += 1

                    if (step + 1) % self.grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters()
                             if p.requires_grad],
                            self.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        global_step += 1

                        avg_loss = epoch_loss / num_batches
                        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

                        log.append({
                            'epoch': epoch,
                            'step': global_step,
                            'loss': loss.item(),
                            'avg_loss': avg_loss
                        })

                except Exception as e:
                    print(f"\n  ⚠️  Step {step} 失败: {e}")
                    self.optimizer.zero_grad()
                    continue

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch+1} 完成，平均 Loss: {avg_epoch_loss:.4f}")

            # 保存每个 epoch 的权重
            ckpt_path = self.save_dir / f"epoch_{epoch+1}"
            self._save_checkpoint(ckpt_path)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = self.save_dir / "best"
                self._save_checkpoint(best_path)
                print(f"  ✅ 最佳模型已保存 (loss={best_loss:.4f})")

        # 保存训练日志
        with open(self.save_dir / "training_log.json", 'w') as f:
            json.dump(log, f, indent=2)

        print(f"\n{'='*60}")
        print("✅ Correction LoRA 训练完成")
        print(f"   最佳 Loss: {best_loss:.4f}")
        print(f"   保存路径: {self.save_dir}")
        print(f"{'='*60}\n")

    def _save_checkpoint(self, save_path: Path):
        """只保存 Correction LoRA 的权重（不保存 stage3 权重）"""
        save_path.mkdir(parents=True, exist_ok=True)

        # 只提取 lora_ 参数
        lora_state = {
            name: param.data.cpu()
            for name, param in self.model.base_vla.llm_backbone.llm.named_parameters()
            if 'lora_' in name
        }
        torch.save(lora_state, save_path / "correction_lora.pt")

        # 保存配置
        meta = {
            'lora_rank': self.lora_rank,
            'saved_at': datetime.now().isoformat(),
            'num_params': sum(p.numel() for p in lora_state.values())
        }
        with open(save_path / "correction_lora_config.json", 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  💾 Correction LoRA 已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage3_ckpt', type=str, required=True,
                        help='Stage3 最终权重目录')
    parser.add_argument('--data', type=str,
                        default='/home/ubuntu/data1/lyy/full_rlds_project-3/'
                                'stage3/correction_lora_train.json')
    parser.add_argument('--save_dir', type=str,
                        default='/home/ubuntu/data1/lyy/full_rlds_project-3/'
                                'checkpoints/correction_lora')
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    args = parser.parse_args()

    config = Stage3Config()
    trainer = CorrectionLoRATrainer(
        stage3_checkpoint_dir=args.stage3_ckpt,
        data_path=args.data,
        save_dir=args.save_dir,
        config=config,
        lora_rank=args.lora_rank,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    trainer.train()


if __name__ == '__main__':
    main()
