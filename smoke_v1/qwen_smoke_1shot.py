"""
qwen_smoke_1shot.py
===================
Qwen3-VL vLLM 接口冒烟测试。两阶段:
  tiny : 极短文本 + 1 图, 验证多模态链路通
  real : 完整 Prompt v1 + 2 图, 验证真实调用格式通

用法:
  python3 qwen_smoke_1shot.py                    # 默认 both
  python3 qwen_smoke_1shot.py --stage tiny       # 只跑 tiny
  python3 qwen_smoke_1shot.py --stage real
  python3 qwen_smoke_1shot.py --sample-id 4      # 指定用哪条样本(默认 4)

诊断规则:
  - tiny 失败 => 服务/接口/多模态链路问题
  - tiny 成功但 real 失败 => prompt/长度/格式问题


  脚本里每个函数的作用(一句话)
函数作用png_to_data_urlPNG → base64 data URLcheck_service_alive调 /v1/models 确认 Qwen 活着,挂了报明确错误call_qwen通用 chat/completions 调用,统一返回 {ok, latency, response_text, error}save_result把请求(含 prompt 全文) + 响应 + 耗时落盘到 outputs/smoke_1shot_{stage}_{ts}.json,图像只存长度不存 base64run_tiny用极短文本 "What do you see..." + 1 图调用,排除接口/多模态问题run_real用完整 Prompt v1 + 2 图 + 师兄和我的 2 条保险条,排除真实 prompt 问题main参数解析 + 健康检查 + 调度 + 总结(三种结论对应师兄的诊断表)
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ============================================================
# 配置
# ============================================================
BASE_URL    = "http://localhost:9998/v1"
MODEL_NAME  = "Qwen3-VL-32B-Instruct"
TEMPERATURE = 0.2
MAX_TOKENS  = 1024
TIMEOUT     = 300    # 秒,real 阶段 32B VLM 推理可能较慢

SAMPLE_FILE = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/smoke_input_10.json"
OUT_DIR     = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/outputs"

# Prompt v1 模块(real 阶段用)
sys.path.insert(0, "/home/ubuntu/data1/lyy/full_rlds_project-3/prompts")


# ============================================================
# 工具函数
# ============================================================
def png_to_data_url(path: str) -> str:
    """把 PNG 文件读成 data:image/png;base64,... 格式"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def check_service_alive() -> bool:
    """调 /v1/models 看 Qwen 是不是活的"""
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        if r.status_code != 200:
            print(f"[ERROR] GET /v1/models 返回 {r.status_code}")
            return False
        data = r.json()
        print(f"[OK] 服务可达,已注册模型: {[m['id'] for m in data.get('data', [])]}")
        return True
    except requests.exceptions.ConnectionError:
        print("[FAIL] 无法连接 Qwen 服务")
        print(f"       请确认服务已启动: bash vllm_qwen.bash")
        print(f"       并检查端口: curl -m 5 {BASE_URL}/models")
        return False
    except Exception as e:
        print(f"[FAIL] 健康检查异常: {e}")
        return False


def call_qwen(messages, tag: str) -> dict:
    """调 Qwen,返回 {ok, response_text, latency, error, raw}"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )
        latency = time.time() - t0
        if r.status_code != 200:
            return {
                "ok": False,
                "tag": tag,
                "latency": latency,
                "error": f"HTTP {r.status_code}: {r.text[:500]}",
                "raw_response": None,
            }
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return {
            "ok": True,
            "tag": tag,
            "latency": latency,
            "response_text": text,
            "error": None,
            "raw_response": data,
        }
    except requests.exceptions.Timeout:
        return {"ok": False, "tag": tag, "latency": time.time()-t0,
                "error": f"Timeout ({TIMEOUT}s)", "raw_response": None}
    except Exception as e:
        return {"ok": False, "tag": tag, "latency": time.time()-t0,
                "error": f"{type(e).__name__}: {e}", "raw_response": None}


def save_result(stage: str, payload: dict, result: dict):
    """落盘:请求体 + 响应 + 耗时"""
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"smoke_1shot_{stage}_{ts}.json")
    # 请求体里的 image_url 太长,打印时缩略,但落盘保留完整
    with open(out_path, "w") as f:
        json.dump({
            "stage": stage,
            "timestamp": ts,
            "request": payload,
            "result": result,
        }, f, ensure_ascii=False, indent=2)
    print(f"[落盘] {out_path}")


# ============================================================
# Stage 1: tiny (极短文本 + 1 图)
# ============================================================
def run_tiny(sample: dict) -> bool:
    print("\n" + "="*70)
    print("Stage TINY: 极短文本 + 1 图 (验证多模态链路)")
    print("="*70)

    img_path = sample["curr_image_path"]
    print(f"[图片] {img_path}")
    if not os.path.isfile(img_path):
        print(f"[FAIL] 图片不存在: {img_path}")
        return False

    data_url = png_to_data_url(img_path)
    print(f"[base64 长度] {len(data_url)} 字符")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "What do you see in this image? Answer in one short sentence."},
            ],
        }
    ]

    print(f"[调用] POST {BASE_URL}/chat/completions")
    result = call_qwen(messages, tag="tiny")

    # 落盘
    payload_for_save = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url_length": len(data_url), "source_file": img_path}},
                    {"type": "text", "text": "What do you see in this image? Answer in one short sentence."},
                ],
            }
        ],
    }
    save_result("tiny", payload_for_save, result)

    # 打印结果
    if result["ok"]:
        print(f"[OK] 耗时 {result['latency']:.2f}s")
        print(f"[Qwen 回复]\n  {result['response_text']}\n")
        return True
    else:
        print(f"[FAIL] {result['error']}")
        return False


# ============================================================
# Stage 2: real (完整 Prompt v1 + 2 图)
# ============================================================
def run_real(sample: dict) -> bool:
    print("\n" + "="*70)
    print("Stage REAL: 完整 Prompt v1 + 2 图 (验证真实调用格式)")
    print("="*70)

    from prompt_v1 import SYSTEM_PROMPT, build_user_prompt

    prev_img = sample["prev_image_path"]
    curr_img = sample["curr_image_path"]
    for pth in (prev_img, curr_img):
        if not os.path.isfile(pth):
            print(f"[FAIL] 图片不存在: {pth}")
            return False

    prev_url = png_to_data_url(prev_img)
    curr_url = png_to_data_url(curr_img)
    print(f"[prev base64 长度] {len(prev_url)}")
    print(f"[curr base64 长度] {len(curr_url)}")

    # history_actions 当前存的是 list(我们之前改过),转成字符串供 prompt 用
    history_str = str(sample["history_actions"])

    user_prompt = build_user_prompt(
        gpt_instruction      = sample["gpt_instruction"],
        subtask_list         = sample["subtask_list"],
        history_actions      = history_str,
        current_subtask_hint = sample.get("current_subtask_hint", ""),
        prev_index           = str(sample["prev_index"]),
        prev_obs             = sample["prev_obs"],
        curr_index           = str(sample["curr_index"]),
        curr_obs             = sample["curr_obs"],
    )

    # 师兄 + 我补的 2 条保险
    insurance = (
        "\n\nIMPORTANT ADDITIONAL RULES:\n"
        "1. If text observations conflict with visual evidence, trust the images.\n"
        "2. Reasoning should be based on observable spatial properties "
        "(bearing, distance, occlusion). Avoid referencing fine visual "
        "details that may not be available to downstream models."
    )
    user_prompt_with_insurance = user_prompt + insurance
    print(f"[user prompt 字符数] {len(user_prompt_with_insurance)}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": prev_url}},
                {"type": "image_url", "image_url": {"url": curr_url}},
                {"type": "text", "text": user_prompt_with_insurance},
            ],
        },
    ]

    print(f"[样本信息] id={sample['id']} target_action={sample['target_action']} group={sample['target_group']}")
    print(f"[调用] POST {BASE_URL}/chat/completions")
    result = call_qwen(messages, tag="real")

    # 落盘(图像只存长度,不存完整 base64)
    payload_for_save = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "sample_id": sample["id"],
        "target_action": sample["target_action"],
        "messages": [
            {"role": "system", "content_length": len(SYSTEM_PROMPT)},
            {"role": "user", "content": [
                {"type": "image_url", "source_file": prev_img, "url_length": len(prev_url)},
                {"type": "image_url", "source_file": curr_img, "url_length": len(curr_url)},
                {"type": "text", "text_length": len(user_prompt_with_insurance)},
            ]},
        ],
        "full_user_prompt": user_prompt_with_insurance,    # 完整 user prompt 落盘,便于复现
    }
    save_result("real", payload_for_save, result)

    # 打印结果
    if result["ok"]:
        print(f"[OK] 耗时 {result['latency']:.2f}s")
        print(f"\n[Qwen 完整输出] (不做过滤)")
        print("-"*70)
        print(result["response_text"])
        print("-"*70)
        print(f"\n[真值动作] {sample['target_action']} ({sample['target_group']})")
        return True
    else:
        print(f"[FAIL] {result['error']}")
        return False


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tiny", "real", "both"], default="both")
    parser.add_argument("--sample-id", type=int, default=4,
                        help="用 smoke_input_10.json 里第几条(默认 4 = vertical_family/history 空)")
    args = parser.parse_args()

    # 健康检查
    print("[健康检查]")
    if not check_service_alive():
        sys.exit(1)

    # 加载样本
    with open(SAMPLE_FILE) as f:
        data = json.load(f)
    samples = data["samples"]
    target = next((s for s in samples if s["id"] == args.sample_id), None)
    if target is None:
        print(f"[FAIL] 找不到 sample_id={args.sample_id}")
        sys.exit(1)
    print(f"[已选样本] id={target['id']} action={target['target_action']} group={target['target_group']}")

    # 运行
    ok_tiny = ok_real = True
    if args.stage in ("tiny", "both"):
        ok_tiny = run_tiny(target)

    if args.stage == "both" and not ok_tiny:
        print("\n[诊断] tiny 失败 => 多模态接口/服务问题,real 跳过")
        sys.exit(2)

    if args.stage in ("real", "both"):
        ok_real = run_real(target)

    # 总结
    print("\n" + "="*70)
    print("冒烟结论")
    print("="*70)
    if args.stage == "both":
        if ok_tiny and ok_real:
            print("✅ 两阶段全通 => 接口 + Prompt v1 + 多模态调用格式都 OK")
            print("   下一步: 跑 generate_v1_smoke.py(10 条样本 × A2+C = 20 次)")
        elif ok_tiny and not ok_real:
            print("⚠️ tiny 通, real 失败 => prompt/长度/格式问题,不是接口问题")
            print("   排查: 看 outputs/smoke_1shot_real_*.json 里 error 字段")
        else:
            print("❌ tiny 失败 => 接口或多模态链路问题")
    elif args.stage == "tiny":
        print("✅ tiny 通" if ok_tiny else "❌ tiny 失败")
    elif args.stage == "real":
        print("✅ real 通" if ok_real else "❌ real 失败")


if __name__ == "__main__":
    main()
