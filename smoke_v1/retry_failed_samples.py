"""
retry_failed_samples.py
========================
对 v1 A2 失败样本做重试实验(温度 0.1 -> 0.0),
统计救回率,区分"随机失手" vs "系统性想错"。

输入: 100 条 v1 A2 原始结果
流程:
  1. 抽出所有 A2 失败样本(strict 不匹配)
  2. 每条: temp=0.1 重试 → 过了标记救回,未过 temp=0.0 再试
  3. 统计: 总救回率、按 gt 分救回率、转移矩阵

使用: python3 retry_failed_samples.py
"""

import base64
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import requests

# ============================================================
# 配置
# ============================================================
BASE_URL    = "http://localhost:9998/v1"
MODEL_NAME  = "Qwen3-VL-32B-Instruct"
MAX_TOKENS  = 1024
TIMEOUT     = 600
RETRY_TEMPERATURES = [0.1, 0.0]   # 依次降温重试

# 输入:100 条 v1 A2 原始结果
GEN_RESULT = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/outputs/generate_v1_smoke_curr_only_20260416_094133_exp_20260416_083507.json"
# 样本数据(有完整 input 字段)
SAMPLE_FILE = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/smoke_input_100_curr_only_20260416_083507.json"
# 输出
OUT_DIR = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/outputs"

INSURANCE = (
    "\n\nIMPORTANT ADDITIONAL RULES:\n"
    "1. Prioritize the most reliable spatial evidence from the provided inputs.\n"
    "2. Reasoning should be based on observable spatial properties "
    "(bearing, distance, occlusion). Avoid fine visual details."
)

sys.path.insert(0, "/home/ubuntu/data1/lyy/full_rlds_project-3/prompts")


def extract_pred(text):
    if not text:
        return None
    m = re.search(r'<next_action>\s*(\d+)\s*</next_action>', text)
    return int(m.group(1)) if m else None


def build_user_prompt(sample):
    from prompt_v1_curr_only import build_user_prompt as _bup
    base = _bup(
        gpt_instruction=sample.get("gpt_instruction", ""),
        subtask_list=sample.get("subtask_list", ""),
        history_actions=str(sample.get("history_actions", [])),
        current_subtask_hint=sample.get("current_subtask_hint", ""),
        curr_index=str(sample.get("curr_index", "")),
        curr_obs=sample.get("curr_obs", ""),
    )
    return base + INSURANCE


def call_qwen(system_prompt, user_text, temperature):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_text},
    ]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT)
        latency = time.time() - t0
        if r.status_code != 200:
            return None, latency, f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        return data["choices"][0]["message"]["content"], latency, None
    except Exception as e:
        return None, time.time()-t0, f"{type(e).__name__}: {e}"


# ============================================================
# Main
# ============================================================
def main():
    # 健康检查
    print("[健康检查]")
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        assert r.status_code == 200
        print(f"[OK] {[m['id'] for m in r.json().get('data', [])]}")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    # 加载
    print(f"\n[读取 原始结果] {GEN_RESULT}")
    gen = json.load(open(GEN_RESULT))
    a2_results = [r for r in gen["results"] if r["mode"] == "A2"]
    print(f"  A2 共 {len(a2_results)} 条")

    # 筛失败
    failed = []
    for r in a2_results:
        pred = extract_pred(r.get("response_text"))
        gt = r.get("target_action")
        if pred != gt:
            failed.append({
                "sample_id": r["sample_id"],
                "target_action": gt,
                "target_group": r.get("target_group"),
                "old_pred": pred,
                "old_response": r.get("response_text"),
            })
    print(f"  A2 失败 {len(failed)} 条")

    # 加载样本输入
    print(f"\n[读取 样本输入] {SAMPLE_FILE}")
    samples = {s["id"]: s for s in json.load(open(SAMPLE_FILE))["samples"]}

    # 加载 system prompt
    from prompt_v1_curr_only import SYSTEM_PROMPT

    # 预计耗时
    print(f"\n[预计耗时] {len(failed)} 条 × 最多 {len(RETRY_TEMPERATURES)} 次 × ~15s = {len(failed)*len(RETRY_TEMPERATURES)*15//60}-{len(failed)*len(RETRY_TEMPERATURES)*15//60*2} 分钟")
    print()

    # 跑重试
    retry_records = []
    for idx, f_rec in enumerate(failed):
        sid = f_rec["sample_id"]
        sample = samples.get(sid)
        if sample is None:
            print(f"  [s{sid}] 找不到样本输入,跳过")
            continue

        user_text = build_user_prompt(sample)

        attempts = []
        rescued = False
        final_pred = f_rec["old_pred"]
        final_temp = None

        for temp in RETRY_TEMPERATURES:
            text, lat, err = call_qwen(SYSTEM_PROMPT, user_text, temp)
            new_pred = extract_pred(text)
            attempts.append({
                "temperature": temp,
                "pred": new_pred,
                "latency": round(lat, 1),
                "error": err,
                "response_text": text,
            })
            if err is None and new_pred == f_rec["target_action"]:
                rescued = True
                final_pred = new_pred
                final_temp = temp
                break
            final_pred = new_pred

        retry_records.append({
            "sample_id": sid,
            "target_action": f_rec["target_action"],
            "target_group": f_rec["target_group"],
            "old_pred": f_rec["old_pred"],
            "final_pred": final_pred,
            "rescued": rescued,
            "rescued_at_temp": final_temp,
            "attempts": attempts,
        })

        status = "✓救回" if rescued else "✗仍错"
        temps_tried = "/".join(f"{a['temperature']}→{a['pred']}" for a in attempts)
        print(f"  [{idx+1:>2}/{len(failed)}] s{sid:>3}  gt={f_rec['target_action']}  old={f_rec['old_pred']}  {temps_tried}  {status}")

    # ============================================================
    # 统计分析
    # ============================================================
    print("\n" + "="*70)
    print("重试统计")
    print("="*70)

    total = len(retry_records)
    rescued_count = sum(1 for r in retry_records if r["rescued"])
    rescue_rate = rescued_count / max(total, 1)

    # 统计 1: 总救回率
    print(f"\n[统计 1: 总救回率]")
    print(f"  {rescued_count} / {total} = {rescue_rate:.1%}")

    # 统计 2: 按 gt 分救回率
    print(f"\n[统计 2: 按 gt_action 分救回率]")
    by_gt = defaultdict(lambda: {"total": 0, "rescued": 0})
    for r in retry_records:
        g = r["target_action"]
        by_gt[g]["total"] += 1
        if r["rescued"]:
            by_gt[g]["rescued"] += 1
    print(f"  {'gt':>3}  {'rescued':>8}/{'total':>5}  {'rate':>6}")
    for gt in sorted(by_gt.keys()):
        s = by_gt[gt]
        print(f"  {gt:>3}  {s['rescued']:>8}/{s['total']:>5}  {s['rescued']/max(s['total'],1):>6.1%}")

    # 统计 3: 转移矩阵 (old_pred -> final_pred)
    print(f"\n[统计 3: 转移 old_pred -> final_pred (只看救回的)]")
    transfers = Counter()
    for r in retry_records:
        if r["rescued"]:
            transfers[(r["old_pred"], r["final_pred"])] += 1
    print(f"  {'old→new':>10}  {'count':>5}")
    for (o, n), c in transfers.most_common():
        print(f"  {o}→{n:<5}  {c:>5}")

    # 重试温度分布
    temp_dist = Counter()
    for r in retry_records:
        if r["rescued"]:
            temp_dist[r["rescued_at_temp"]] += 1
    print(f"\n[救回时的温度分布]")
    for temp, cnt in sorted(temp_dist.items()):
        print(f"  temp={temp}: {cnt} 条")

    # 统计 4: 三档决策
    print(f"\n[统计 4: 决策门槛]")
    if rescue_rate >= 0.25:
        verdict = ">= 25%,建议直接扩到 4500 全量"
    elif rescue_rate >= 0.10:
        verdict = "10-25%,建议先跑 300 条中试"
    else:
        verdict = "< 10%,重试无效,需要换思路"
    print(f"  救回率 {rescue_rate:.1%}: {verdict}")

    # ============================================================
    # 落盘
    # ============================================================
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"retry_failed_{ts}.json")

    meta = {
        "timestamp": ts,
        "source_gen_result": GEN_RESULT,
        "source_sample_file": SAMPLE_FILE,
        "retry_temperatures": RETRY_TEMPERATURES,
        "total_failed_input": len(failed),
        "total_retried": total,
        "rescued_count": rescued_count,
        "rescue_rate": round(rescue_rate, 3),
        "by_gt": {str(k): v for k, v in by_gt.items()},
        "transfers": {f"{o}_to_{n}": c for (o, n), c in transfers.items()},
        "rescued_at_temp": {str(k): v for k, v in temp_dist.items()},
        "verdict": verdict,
    }
    with open(out_path, "w") as fout:
        json.dump({"_meta": meta, "records": retry_records}, fout, ensure_ascii=False, indent=2)

    print(f"\n[落盘] {out_path}")


if __name__ == "__main__":
    main()
