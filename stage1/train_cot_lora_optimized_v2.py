"""
CoT 训练 - LoRA 版本（省内存优化版）
优化项：
1. Gradient Checkpointing - 减少50%显存
2. FP16混合精度 - 减少50%显存
3. 梯度累积 - 模拟更大batch
4. 减小LoRA rank - r=8而非r=16
"""
import sys
import os
sys.path.insert(0, '/home/ubuntu/data1/lyy/OpenFly-Platform/train')
sys.path.insert(0, '/home/ubuntu/data1/zx/OpenFly-Platform/train')

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler  # 🔥 混合精度

# LoRA 相关
from peft import LoraConfig, get_peft_model, TaskType

# Monkey patch llm_backbone
import model.llm_backbone as llm_backbone_module
original_init = llm_backbone_module.LLaMa2LLMBackbone.__init__

def patched_init(self, llm_max_length, hf_token=None, use_flash_attention_2=False, inference_mode=False):
    from transformers import LlamaForCausalLM, AutoConfig
    super(llm_backbone_module.LLaMa2LLMBackbone, self).__init__()
    self.llm_max_length = llm_max_length
    self.inference_mode = inference_mode
    
    hf_hub_path = "/home/ubuntu/data1/zx/OpenFly-Platform/train/meta-llama/llama2-7b-hf"
    
    if not inference_mode:
        self.llm = LlamaForCausalLM.from_pretrained(
            hf_hub_path,
            local_files_only=True,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    else:
        llm_config = AutoConfig.from_pretrained(hf_hub_path, local_files_only=True)
        self.llm = LlamaForCausalLM._from_config(llm_config)
    
    self.tokenizer = llm_backbone_module.AutoTokenizer.from_pretrained(
        hf_hub_path, 
        local_files_only=True,
        model_max_length=llm_max_length, 
        padding_side="right", 
        use_fast=False
    )
    self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    self.llm.config.pad_token_id = self.tokenizer.pad_token_id
    self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

llm_backbone_module.LLaMa2LLMBackbone.__init__ = patched_init

from model.overwatch import initialize_overwatch
from model.load_model import load_vla
from model.prompt_llama2 import LLaMa2ChatPromptBuilder

sys.path.insert(0, '/home/ubuntu/data1/lyy/OpenFly-Platform/train')
from cot_dataset_final import CoTDataset

IGNORE_INDEX = -100

def setup_distributed():
    if 'RANK' in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        gpu_id = local_rank
        torch.cuda.set_device(gpu_id)
        return local_rank, dist.get_world_size(), gpu_id
    else:
        torch.cuda.set_device(0)
        return 0, 1, 0

def add_lora_to_model(vla):
    """给 VLA 的 LLM 添加 LoRA"""
    # LoRA 配置（优化版：减小rank）
    lora_config = LoraConfig(
        r=8,  # 🔥 从16改为8，减少参数量
        lora_alpha=16,  # 🔥 保持2倍关系
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    vla.llm_backbone.llm = get_peft_model(vla.llm_backbone.llm, lora_config)
    
    print("=" * 60)
    print("🎯 LoRA 配置 (优化版)")
    print("=" * 60)
    vla.llm_backbone.llm.print_trainable_parameters()
    print("=" * 60)
    
    return vla

def load_images_for_sample(sample, image_base_path):
    """加载真实图片（3帧）"""
    from pathlib import Path
    
    episode_id = sample['episode_id']
    frame_idx = sample['frame_idx']
    
    episode_path = Path(image_base_path) / episode_id
    img_file = f"{frame_idx}.png"
    img_path = episode_path / img_file
    
    if img_path.exists():
        img = Image.open(img_path).convert('RGB')
    else:
        print(f"⚠️  图片不存在: {img_path}")
        img = Image.new('RGB', (224, 224), color='gray')
    
    return img, img, img

def process_batch_cot(batch, vla, image_base_path, device):
    i = 0
    img_cur, img_past1, img_past2 = load_images_for_sample(
        {'episode_id': batch['episode_id'][i], 'frame_idx': batch['frame_idx'][i]},
        image_base_path
    )
    
    model = vla.module if hasattr(vla, "module") else vla
    tokenizer = model.llm_backbone.get_tokenizer()
    image_transform = model.vision_backbone.get_image_transform()
    
    lang = batch['instruction'][i]
    cot_output = batch['cot_output'][i]
    
    prompt_builder = LLaMa2ChatPromptBuilder("prismatic")
    conversation = [
        {"from": "human", "value": f"What action should the robot take to {lang}?"},
        {"from": "gpt", "value": cot_output},
    ]
    for turn in conversation:
        prompt_builder.add_turn(turn["from"], turn["value"])
    
    input_ids = tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
    labels = list(input_ids)
    
    tr_img_cur = image_transform(img_cur)
    tr_pst1 = image_transform(img_past1)
    tr_pst2 = image_transform(img_past2)
    
    pixel_values = {}
    for k in tr_img_cur.keys():
        combined = torch.cat((tr_img_cur[k], tr_pst1[k], tr_pst2[k]), dim=0)
        pixel_values[k] = combined.unsqueeze(0).to(device)
    
    cot_tokens = tokenizer(cot_output, add_special_tokens=False).input_ids
    labels = [IGNORE_INDEX if i < len(labels) - len(cot_tokens) - 1 else labels[i] for i in range(len(labels))]
    
    return {
        'pixel_values': pixel_values,
        'input_ids': torch.tensor([input_ids]).to(device),
        'labels': torch.tensor([labels]).to(device)
    }


import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def main():
    print("="*60)
    print("🎯 CoT LoRA 训练（优化版）")
    print("="*60)
    
    local_rank, world_size, gpu_id = setup_distributed()
    is_rank0 = (local_rank == 0)
    
    if is_rank0:
        print(f"✅ World size: {world_size}, GPU: {gpu_id}")
        print("🔥 优化项:")
        print("   - Gradient Checkpointing")
        print("   - FP16 混合精度")
        print("   - 梯度累积 (accumulation_steps=4)")
        print("   - LoRA rank=8")
    
    # 数据
    if is_rank0:
        print("\n📁 加载数据...")
    dataset = CoTDataset("/home/ubuntu/data1/lyy/full_rlds_project-3/data/train_with_cot_4500.json")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 模型
    if is_rank0:
        print(f"\n📦 加载模型...")
    
    checkpoint_path = "/home/ubuntu/data1/zx/OpenFly-Platform/train/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
    overwatch = initialize_overwatch("cot-lora-optimized")
    
    vla = load_vla(checkpoint_path, hf_token="", load_for_training=True, grid_size=16)
    
    # 🔥 启用 Gradient Checkpointing
    if is_rank0:
        print("\n🔥 启用 Gradient Checkpointing...")
    if hasattr(vla.llm_backbone.llm, 'gradient_checkpointing_enable'):
        vla.llm_backbone.llm.gradient_checkpointing_enable()
        if is_rank0:
            print("   ✅ LLM Gradient Checkpointing 已启用")
    
    # 添加 LoRA
    if is_rank0:
        print("\n🔧 添加 LoRA 适配器...")
    lora_path = "/home/ubuntu/data1/lyy/full_rlds_project-3/checkpoints/stage1_final"
    if os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json")) and is_rank0:
        print("🔄 从已有 LoRA 继续训练...")
        from peft import PeftModel
        vla.llm_backbone.llm = PeftModel.from_pretrained(vla.llm_backbone.llm, lora_path, is_trainable=True)
    else:
        if is_rank0:
            print("🆕 创建新的 LoRA...")
        vla = add_lora_to_model(vla)

    # 解冻 projector
    if is_rank0:
        print("\n🔓 解冻 projector...")
    if hasattr(vla, "projector"):
        for param in vla.projector.parameters():
            param.requires_grad = True
        if is_rank0:
            trainable = sum(p.numel() for p in vla.projector.parameters() if p.requires_grad)
            print(f"   Projector 可训练参数: {trainable:,}")

    vla = vla.to(f"cuda:{gpu_id}")
    
    # DDP
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        vla = DDP(vla, device_ids=[gpu_id], find_unused_parameters=True)
    
    vla.train()
    
    if is_rank0:
        print(f"✅ 模型就绪")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, vla.parameters()),
        lr=1e-4
    )
    
    # 🔥 混合精度 Scaler
    scaler = GradScaler()
    
    # 训练配置
    if is_rank0:
        print("\n" + "="*60)
        print("🚀 开始训练")
        print("="*60)
    
    image_base_path = "/home/ubuntu/data1/lyy/full_rlds_project-3/images"

    global_step = 0
    num_epochs = 2  # 训练3个epoch
    max_steps = num_epochs * len(dataset)  # 基于数据集大小
    accumulation_steps = 4

    if is_rank0:
        print(f"📊 训练配置:")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   训练epochs: {num_epochs}")
        print(f"   总训练步数: {max_steps}")
        print(f"   预计有效batch数: {max_steps // accumulation_steps}")
    

    for epoch in range(num_epochs):
        if is_rank0:
            print(f"\n{'='*60}")
            print(f"📚 Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")
        
        for batch in dataloader:
            if global_step >= max_steps:
                break

        
            try:
                # 🔥 添加60秒超时保护
                with time_limit(60):
                    processed = process_batch_cot(batch, vla, image_base_path, f"cuda:{gpu_id}")
                    
                    # 🔥 混合精度前向传播
                    with autocast():
                        outputs = vla(
                            pixel_values=processed['pixel_values'],
                            input_ids=processed['input_ids'],
                            labels=processed['labels']
                        )
                        loss = outputs.loss / accumulation_steps  # 平均损失
                    
                    # 🔥 混合精度反向传播
                    scaler.scale(loss).backward()
                    
                    # 🔥 梯度累积：每accumulation_steps步更新一次
                    if (global_step + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    if is_rank0 and (global_step + 1) % accumulation_steps == 0:
                        actual_loss = loss.item() * accumulation_steps
                        print(f"[Step {global_step:03d}] Loss: {actual_loss:.4f}")
                    
                    global_step += 1
                    
            except TimeoutException:
                if is_rank0:
                    print(f"⏰ Step {global_step} 超时（>60s），跳过...")
                global_step += 1
                optimizer.zero_grad()
                continue
                
            except Exception as e:
                if is_rank0:
                    print(f"❌ Step {global_step}: {e}")
                global_step += 1
                optimizer.zero_grad()
                continue
    
    # 保存权重
    if is_rank0:
        save_dir = "/home/ubuntu/data1/lyy/full_rlds_project-3/checkpoints/stage1_final"
        os.makedirs(save_dir, exist_ok=True)
        
        model = vla.module if world_size > 1 else vla
        model.llm_backbone.llm.save_pretrained(save_dir)
        
        projector_path = os.path.join(save_dir, "projector.pt")
        torch.save(model.projector.state_dict(), projector_path)
        print(f"\n✅ LoRA 权重已保存到: {save_dir}")
        print(f"✅ Projector 权重已保存到: {projector_path}")
    
    if is_rank0:
        print("\n✅ 训练完成！")

if __name__ == "__main__":
    main()