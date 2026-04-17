"""
generate_v1_smoke.py
====================
对 smoke_input_10.json 的 10 条样本,每条按 A2 -> C 顺序各调一次 Qwen。
落盘到 outputs/generate_v1_smoke_{ts}.json 和 _latest.json。

A2 = Prompt v1 纯文本
C  = Prompt v1 图+文
除"有无图像"外其他条件一致,单变量对照。

失败策略:单次失败重试 1 次(2s),连续 3 次失败熔断。
"""

import argparse
import base64
import hashlib
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
TIMEOUT     = 600
RETRY_BACKOFF = 2
CIRCUIT_BREAKER_THRESHOLD = 3

SAMPLE_FILE = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/smoke_input_10.json"
OUT_DIR     = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/outputs"

INSURANCE = (
    "\n\nIMPORTANT ADDITIONAL RULES:\n"
    "1. Prioritize the most reliable spatial evidence from the provided inputs.\n"
    "2. Reasoning should be based on observable spatial properties "
    "(bearing, distance, occlusion). Avoid fine visual details."
)

sys.path.insert(0, "/home/ubuntu/data1/lyy/full_rlds_project-3/prompts")


# ============================================================
# 工具函数
# ============================================================
def png_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def hash12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def check_service_alive() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        if r.status_code != 200:
            print(f"[ERROR] GET /v1/models 返回 {r.status_code}")
            return False
        data = r.json()
        print(f"[OK] 服务可达,已注册模型: {[m['id'] for m in data.get('data', [])]}")
        return True
    except Exception as e:
        print(f"[FAIL] 服务不可达: {e}")
        return False


def build_full_user_prompt(sample: dict) -> str:
    from prompt_v1 import build_user_prompt
    base = build_user_prompt(
        gpt_instruction      = sample.get("gpt_instruction", ""),
        subtask_list         = sample.get("subtask_list", ""),
        history_actions      = str(sample.get("history_actions", [])),
        current_subtask_hint = sample.get("current_subtask_hint", ""),
        prev_index           = str(sample.get("prev_index", "")),
        prev_obs             = sample.get("prev_obs", ""),
        curr_index           = str(sample.get("curr_index", "")),
        curr_obs             = sample.get("curr_obs", ""),
    )
    return base + INSURANCE


def build_messages(sample: dict, mode: str, system_prompt: str, user_text: str):
    """返回 (messages, message_hash)"""
    if mode == "A2":
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ]
        mh = hash12(f"A2|{system_prompt}|{user_text}")
        return msgs, mh
    elif mode == "C":
        prev_url = png_to_data_url(sample.get("prev_image_path", ""))
        curr_url = png_to_data_url(sample.get("curr_image_path", ""))
        msgs = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": prev_url}},
                    {"type": "image_url", "image_url": {"url": curr_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        mh = hash12(f"C|{system_prompt}|{len(prev_url)}|{len(curr_url)}|{user_text}")
        return msgs, mh
    else:
        raise ValueError(f"unknown mode: {mode}")


def call_qwen_once(messages) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT)
        latency = time.time() - t0
        if r.status_code != 200:
            return {"ok": False, "latency": latency,
                    "error": f"HTTP {r.status_code}: {r.text[:300]}",
                    "response_text": None}
        data = r.json()
        return {"ok": True, "latency": latency,
                "response_text": data["choices"][0]["message"]["content"],
                "error": None}
    except requests.exceptions.Timeout:
        return {"ok": False, "latency": time.time()-t0,
                "error": f"Timeout ({TIMEOUT}s)", "response_text": None}
    except Exception as e:
        return {"ok": False, "latency": time.time()-t0,
                "error": f"{type(e).__name__}: {e}", "response_text": None}


def call_with_retry(messages, tag: str) -> dict:
    result = call_qwen_once(messages)
    result["retry_count"] = 0
    if not result["ok"]:
        print(f"    [重试] {tag}: {result['error']}")
        time.sleep(RETRY_BACKOFF)
        result = call_qwen_once(messages)
        result["retry_count"] = 1
    return result


# ============================================================
# 主循环
# ============================================================
def run(samples, mode_order):
    from prompt_v1 import SYSTEM_PROMPT

    results = []
    consecutive_fails = 0
    aborted = False

    for sample in samples:
        if aborted:
            break
        sid = sample["id"]
        print(f"\n--- Sample {sid}  target={sample.get('target_action')}  group={sample.get('target_group','unknown')} ---")

        user_text = build_full_user_prompt(sample)
        prompt_hash = hash12(user_text)

        for mode in mode_order:
            if aborted: break
            tag = f"s{sid}_{mode}"
            print(f"  [{tag}]  prompt_hash={prompt_hash}  ", end="", flush=True)

            # 构造 messages,失败(如图像读不到)记一条 ok=false 继续
            try:
                messages, message_hash = build_messages(sample, mode, SYSTEM_PROMPT, user_text)
                r = call_with_retry(messages, tag)
            except Exception as e:
                print(f"BUILD_FAIL  {type(e).__name__}: {e}")
                r = {"ok": False, "latency": 0.0, "retry_count": 0,
                     "response_text": None,
                     "error": f"build_messages failed: {type(e).__name__}: {e}"}
                message_hash = None

            entry = {
                "input_digest": f"sample{sid}_prev{sample.get('prev_index','?')}_curr{sample.get('curr_index','?')}_mode{mode}",
                "sample_id": sid,
                "target_action": sample.get("target_action"),
                "target_group": sample.get("target_group", "unknown"),
                "mode": mode,
                "prompt_hash": prompt_hash,
                "message_hash": message_hash,
                "ok": r["ok"],
                "latency": round(r["latency"], 2),
                "retry_count": r["retry_count"],
                "response_text": r.get("response_text"),
                "error": r.get("error"),
                "timestamp": datetime.now().isoformat(),
            }
            results.append(entry)

            if r["ok"]:
                print(f"OK  {r['latency']:.1f}s  retry={r['retry_count']}")
                consecutive_fails = 0
            else:
                print(f"FAIL  {r['error']}")
                consecutive_fails += 1
                if consecutive_fails >= CIRCUIT_BREAKER_THRESHOLD:
                    print(f"\n[熔断] 连续 {consecutive_fails} 次失败,停止后续调用")
                    aborted = True

    return results, aborted


def save_results(results, mode_order, aborted):
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta = {
        "timestamp": ts,
        "total_results": len(results),
        "ok_count": sum(1 for r in results if r["ok"]),
        "fail_count": sum(1 for r in results if not r["ok"]),
        "aborted_by_circuit_breaker": aborted,
        "mode_order": mode_order,
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "insurance_text": INSURANCE.strip(),
    }
    payload = {"_meta": meta, "results": results}

    ts_path = os.path.join(OUT_DIR, f"generate_v1_smoke_{ts}.json")
    with open(ts_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    latest_path = os.path.join(OUT_DIR, "generate_v1_smoke_latest.json")
    with open(latest_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[落盘]")
    print(f"  {ts_path}")
    print(f"  {latest_path}")
    return ts_path, latest_path, meta


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", type=str, default=None,
                        help="逗号分隔的 sample_id,默认跑全部 10 条")
    args = parser.parse_args()

    print("[健康检查]")
    if not check_service_alive():
        sys.exit(1)

    with open(SAMPLE_FILE) as f:
        data = json.load(f)
    samples = data["samples"]

    if args.sample_ids:
        want = set(int(x) for x in args.sample_ids.split(","))
        samples = [s for s in samples if s["id"] in want]
    print(f"[样本数] {len(samples)}")
    print(f"[预计调用] {len(samples) * 2} 次(每条 A2 + C)")
    print(f"[预计耗时] ~{len(samples) * 65 / 60:.1f} 分钟")

    mode_order = ["A2", "C"]
    t0 = time.time()
    results, aborted = run(samples, mode_order)
    elapsed = time.time() - t0

    ts_path, latest_path, meta = save_results(results, mode_order, aborted)

    print("\n" + "="*70)
    print("调用总结")
    print("="*70)
    print(f"总调用: {meta['total_results']}")
    print(f"  成功: {meta['ok_count']}")
    print(f"  失败: {meta['fail_count']}")
    if aborted:
        print(f"  ⚠️ 熔断触发")
    print(f"实际耗时: {elapsed/60:.1f} 分钟")
    print(f"\n下一步: 跑 filter_v1.py 读 {os.path.basename(latest_path)} 做 6 条硬规则过滤")


if __name__ == "__main__":
    main()
