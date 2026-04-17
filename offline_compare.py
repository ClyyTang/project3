"""
离线对照: base vs Stage1, 在训练数据上看动作输出
完全复用 eval_cogfly.py 的 prompt/图像/prefix 构造
"""
import sys, json, torch, re
from pathlib import Path
from PIL import Image

sys.path.insert(0, "/home/ubuntu/data1/lyy/OpenFly-Platform/train")
from model.load_model import load_vla
from model.prompt_llama2 import LLaMa2ChatPromptBuilder
from peft import PeftModel

BASE_CKPT  = "/home/ubuntu/data1/zx/OpenFly-Platform/train/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
STAGE1_DIR = "/home/ubuntu/data1/lyy/full_rlds_project-3/checkpoints/stage1_final"
DATA_JSON  = "/home/ubuntu/data1/lyy/full_rlds_project-3/data/train_with_cot_4500_stage1_clean_v3.json"
IMG_BASE   = "/home/ubuntu/data1/lyy/full_rlds_project-3/images"
DEVICE = "cuda:0"

def load_base():
    print("[加载 base]")
    p = load_vla(BASE_CKPT, load_for_training=False)
    return p.to(DEVICE).eval()

def load_stage1():
    print("[加载 Stage1 = base + LoRA + projector]")
    p = load_vla(BASE_CKPT, load_for_training=False)
    p.llm_backbone.llm = PeftModel.from_pretrained(p.llm_backbone.llm, STAGE1_DIR, is_trainable=False)
    proj_path = Path(STAGE1_DIR) / "projector.pt"
    p.projector.load_state_dict(torch.load(proj_path, map_location="cpu"))
    return p.to(DEVICE).eval()

def build_input(policy, instruction, image):
    """完全复用 eval_cogfly.py 520~545 行"""
    tokenizer = policy.llm_backbone.get_tokenizer()
    image_transform = policy.vision_backbone.image_transform   # 属性,不是方法

    pb = LLaMa2ChatPromptBuilder("prismatic")
    pb.add_turn("human", f"What action should the robot take to {instruction}?")
    prompt = pb.get_prompt()
    if prompt.endswith("</s>"):
        prompt = prompt[:-4].rstrip()

    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(DEVICE)
    prefix_ids = tokenizer("<thinking>", add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)

    tr = image_transform(image)
    pixel_values = {}
    for k in tr.keys():
        combined = torch.cat((tr[k], tr[k], tr[k]), dim=0)
        pixel_values[k] = combined.unsqueeze(0).to(DEVICE)

    return torch.cat([prompt_ids, prefix_ids], dim=1), pixel_values, tokenizer

def generate(policy, current_ids, pixel_values, tokenizer, max_new=400):
    """贪心生成,遇到 </action>/</next_action>/eos 停止"""
    out_ids = current_ids.clone()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(max_new):
            outputs = policy(pixel_values=pixel_values, input_ids=out_ids)
            next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            out_ids = torch.cat([out_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            tail = tokenizer.decode(out_ids[0, -8:], skip_special_tokens=False)
            if "</action>" in tail or "</next_action>" in tail:
                break
    full = tokenizer.decode(out_ids[0, current_ids.shape[1]:], skip_special_tokens=False)
    return "<thinking>" + full

def extract_action(text):
    """宽松提取: 命中XML返回数字, 不命中返回None (保留原始文本供观察)"""
    m = re.search(r"<(?:action|next_action)>\s*(\d+)\s*</(?:action|next_action)>", text)
    return int(m.group(1)) if m else None

def main():
    data = json.load(open(DATA_JSON))
    sample = data[0]
    instruction = sample["gpt_instruction"]
    image_path  = sample["image_path"]
    frames = sample["index_list"]
    actions_gt = sample["action"]
    print(f"\n样本: {image_path}")
    print(f"指令: {instruction[:100]}...")
    print(f"真值动作: {actions_gt}")
    print(f"Frame 数: {len(frames)}\n")

    # === Base 段已跳过(只诊断 Stage1) ===
    print("\n[跳过 Base,只跑 Stage1 验证]\n")
    base_outs = [(None, '(skipped)')] * len(frames)


    # === Stage1 全跑 ===
    s1 = load_stage1()
    print("\n" + "="*80)
    print("STAGE1 模型输出")
    print("="*80)
    s1_outs = []
    for i, fid in enumerate(frames):
        if i >= len(actions_gt):
            break
        img = Image.open(f"{IMG_BASE}/{image_path}/{fid}.png").convert("RGB")
        cur, pv, tok = build_input(s1, instruction, img)
        text = generate(s1, cur, pv, tok)
        act = extract_action(text)
        s1_outs.append((act, text))
        print(f"\nFrame {i}  真值={actions_gt[i]}  Stage1动作={act}")
        print(f"  原始文本: {text[:300]!r}")

    # === 对照表 ===
    print("\n" + "="*80)
    print("对照表")
    print("="*80)
    print(f"{'Frame':<6}{'真值':<6}{'Base':<8}{'Stage1':<8}标记")
    for i in range(len(s1_outs)):
        b = base_outs[i][0]; s = s1_outs[i][0]
        gt = actions_gt[i]
        flags = []
        if b == s:                       flags.append("Base==Stage1")
        if s == 9 and gt != 9:           flags.append("Stage1塌缩到9")
        if s == gt:                      flags.append("✓正确")
        print(f"{i:<6}{gt:<6}{str(b):<8}{str(s):<8}{' '.join(flags)}")

if __name__ == "__main__":
    main()
