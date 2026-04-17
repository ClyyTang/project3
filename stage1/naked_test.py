import torch
import sys
from pathlib import Path
from PIL import Image

# ================= 0. 环境路径与导包 =================
sys.path.insert(0, '/home/ubuntu/data1/lyy/OpenFly-Platform/train')
from peft import PeftModel
from model.load_model import load_vla

# ================= 1. 加载模型 =================
base_checkpoint = "/home/ubuntu/data1/zx/OpenFly-Platform/train/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
lora_dir = "/home/ubuntu/data1/lyy/full_rlds_project-3/checkpoints/stage1_final"

print("⏳ 正在加载基础模型...")
policy = load_vla(base_checkpoint, load_for_training=False)

print("⏳ 正在加载 LoRA 和 Projector...")
policy.llm_backbone.llm = PeftModel.from_pretrained(
    policy.llm_backbone.llm,
    lora_dir,
    is_trainable=False
)
projector_path = Path(lora_dir) / "projector.pt"
policy.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

# 🔥 补丁 2: 注入 use_mor / use_mora 兼容属性
for name, mod in policy.named_modules():
    if "LLaMa2LLMBackbone" in type(mod).__name__:
        if not hasattr(mod, "use_mor"): mod.use_mor = False
        if not hasattr(mod, "use_mora"): mod.use_mora = False

device = "cuda:0"
policy = policy.to(device)

# 🔥 补丁 3 (Part 1): 获取模型真实 dtype
model_dtype = next(policy.parameters()).dtype
print(f"✅ 模型加载完毕！使用 Dtype: {model_dtype}\n")

# ================= 2. 构造输入 =================
tokenizer = policy.llm_backbone.get_tokenizer()
image_transform = policy.vision_backbone.image_transform

instruction = "What action should the robot take to move forward to the gray building?"
from model.prompt_llama2 import LLaMa2ChatPromptBuilder
prompt_builder = LLaMa2ChatPromptBuilder("prismatic")
prompt_builder.add_turn("human", instruction)
prompt = prompt_builder.get_prompt()
prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(device)

dummy_img = Image.new('RGB', (224, 224), color='gray')
tr_img = image_transform(dummy_img)
pixel_values = {}
for k in tr_img.keys():
    combined = torch.cat((tr_img[k], tr_img[k], tr_img[k]), dim=0) 
    # 🔥 补丁 3 (Part 2): 严格对齐 dtype
    pixel_values[k] = combined.unsqueeze(0).to(device=device, dtype=model_dtype)

# ================= 3. 裸测生成循环 =================
print("🚀 开始逐字生成，透视 Logits 概率...")
current_ids = prompt_ids.clone()

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for step in range(20): 
            outputs = policy(pixel_values=pixel_values, input_ids=current_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            probs = torch.softmax(next_token_logits, dim=-1)
            top3_probs, top3_indices = torch.topk(probs, 3, dim=-1)
            
            # 🔥 补丁 1: tensor 转 int，防止 decode 报错
            top3_tokens = [tokenizer.decode([int(idx.item())]) for idx in top3_indices[0]]
            top3_probs_percent = [f"{p.item()*100:.1f}%" for p in top3_probs[0]]
            
            next_token = top3_indices[:, 0:1]
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # 🔥 补丁 1: tensor 转 int
            best_token_str = tokenizer.decode([int(next_token[0,0].item())])
            print(f"Step {step+1:02d} | 决定输出: {best_token_str!r:<15} | Top3 候选项: {list(zip(top3_tokens, top3_probs_percent))}")
            
            if next_token.item() == tokenizer.eos_token_id:
                print("🛑 模型主动输出了 EOS (结束符)，生成停止。")
                break

print("\n✨ 最终解码结果:")
gen_ids = current_ids[0, prompt_ids.shape[1]:]
print(tokenizer.decode(gen_ids, skip_special_tokens=False))
