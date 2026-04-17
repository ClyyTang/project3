"""
sample_100.py
=============
分层抽样 100 条样本作为 Prompt v1 (curr_only) 的小样本统计输入。

配额(师兄拍板):
  0:10  1:10  2:10  3:10  4:10  5:10  8:15  9:25  (共 100)
  
注:动作 6 (strafe_left) 和 7 (strafe_right) 数据集里候选数 = 0,不抽。

落盘 2 个文件:
  - smoke_input_100_{TS}.json            : 双帧版(prev + curr 都有,留作 ablation)
  - smoke_input_100_curr_only_{TS}.json  : curr_only 版(只有 curr,curr=idx[i])

设计原则:
  - seed=42 可复现
  - 失败补抽,直到每个动作配额凑齐
  - 配额未凑齐时 fail-fast (sys.exit(1))
  - 数据归一化:target_action/history_actions 全部走 normalize_action(-1→4, -2→5)
  - 保留 raw_target_action 和 raw_history_actions 供审计追溯
  - _meta 记录 alias_count, selection_reason, excluded_stats
"""

import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

# 引入项目通用动作语义
sys.path.insert(0, "/home/ubuntu/data1/lyy/full_rlds_project-3")
from action_semantics import normalize_action

# ============================================================
# 配置
# ============================================================
SEED = 42

DATA      = "/home/ubuntu/data1/lyy/full_rlds_project-3/data/train_with_cot_4500_stage1_clean_v3.json"
SUBTASKS  = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/outputs/subtasks.json"
OBS       = "/home/ubuntu/data1/lyy/full_rlds_project-3/1_cot_generation/outputs/observations.json"
IMG_BASE  = "/home/ubuntu/data1/lyy/full_rlds_project-3/images"
OUT_DIR   = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1"

# 师兄拍板配额
QUOTA = {0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 8: 15, 9: 25}

# 时间戳(本轮实验 ID,用于串起来 sample/generate/filter 三个阶段的产物)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DUAL    = os.path.join(OUT_DIR, f"smoke_input_100_{TS}.json")
OUT_CURRONLY = os.path.join(OUT_DIR, f"smoke_input_100_curr_only_{TS}.json")

# 图像路径解析:优先 .png,找不到试 .jpg
def resolve_image_path(image_path: str, frame_id: str):
    """返回真实图像文件路径,找不到返回 None"""
    for ext in ("png", "jpg"):
        candidate = f"{IMG_BASE}/{image_path}/{frame_id}.{ext}"
        if os.path.isfile(candidate):
            return candidate
    return None


# 动作分组(后续 filter 用于按族统计)
ACTION_GROUPS = {
    0: "stop",
    1: "forward_family", 8: "forward_family", 9: "forward_family",
    2: "turn_family", 3: "turn_family",
    4: "vertical_family", 5: "vertical_family",
    6: "strafe_family", 7: "strafe_family",
}

def get_group(a):
    return ACTION_GROUPS.get(a, "unknown")


# ============================================================
# 加载数据 + 建索引
# ============================================================
print(f"[加载] {DATA}")
data = json.load(open(DATA))
print(f"[加载] {SUBTASKS}")
sub_idx = {x["image_path"]: x.get("sub-tasks", {}) for x in json.load(open(SUBTASKS))}
print(f"[加载] {OBS}")
obs_idx = {x["image_path"]: x.get("current", {}) for x in json.load(open(OBS))}
print(f"[加载完成] 总轨迹 {len(data)} 条\n")


# ============================================================
# 候选枚举: (episode_idx, frame_i, target_action_normalized)
# ============================================================
print("[枚举候选]")
candidates_by_action = defaultdict(list)
alias_count = 0
for ep_idx, ep in enumerate(data):
    acts = ep.get("action", [])
    for i, a in enumerate(acts):
        a_norm = normalize_action(a)
        if a_norm not in QUOTA: continue
        candidates_by_action[a_norm].append((ep_idx, i))
        if a != a_norm:   # 被映射过(原值是 -1 或 -2)
            alias_count += 1

for a, lst in sorted(candidates_by_action.items()):
    print(f"  动作 {a} ({get_group(a)}): 候选 {len(lst)} 个")
print(f"\n[全局统计] 原值 -1/-2 被归一化的总数: {alias_count}")


# ============================================================
# 校验单条候选 + 构建样本(双帧版,稍后转 curr_only)
# ============================================================
def validate_and_build(ep_idx, i):
    ep = data[ep_idx]
    image_path = ep["image_path"]
    idx_list = ep["index_list"]
    acts = ep["action"]

    is_last = (i == len(acts) - 1)
    if is_last:
        # 故意丢弃单步轨迹(i==0 且 is_last):
        # 理论上 curr_only 语义可保留(看 idx[0] 推 action[0]),
        # 但和 sample_10.py 的抽样行为保持一致,避免引入新变量影响对照。
        # 排除统计里会单独记 single_step_episode 计数。
        if i == 0: return False, "single_step_episode", None
        if i >= len(idx_list): return False, "index_mismatch", None
        prev_id = idx_list[i-1]
        curr_id = idx_list[i]
    else:
        if i + 1 >= len(idx_list): return False, "index_mismatch", None
        prev_id = idx_list[i]
        curr_id = idx_list[i+1]

    # obs 非空
    obs_dict = obs_idx.get(image_path, {})
    prev_obs = obs_dict.get(prev_id, "").strip()
    curr_obs = obs_dict.get(curr_id, "").strip()
    if not prev_obs: return False, "missing_obs", None
    if not curr_obs: return False, "missing_obs", None

    # 图像文件存在(支持 .png 和 .jpg)
    prev_img = resolve_image_path(image_path, prev_id)
    curr_img = resolve_image_path(image_path, curr_id)
    if prev_img is None: return False, "missing_image", None
    if curr_img is None: return False, "missing_image", None

    # subtasks
    st_dict = sub_idx.get(image_path, {})
    if not isinstance(st_dict, dict) or not st_dict:
        return False, "missing_subtasks", None
    # 按数字 key 排序(防止不同来源 JSON 顺序不一致影响 prompt 哈希)
    import re as _re
    def _key_to_int(k):
        digits = _re.sub(r"\D", "", str(k))
        return int(digits) if digits else 0
    sorted_items = sorted(st_dict.items(), key=lambda kv: _key_to_int(kv[0]))
    subtask_list_str = "\n".join(f"{k}. {v}" for k, v in sorted_items)

    # history (raw + normalized)
    raw_history       = list(acts[:i])
    normalized_history = [normalize_action(a) for a in raw_history]

    raw_target = acts[i]
    normalized_target = normalize_action(raw_target)

    return True, "ok", {
        "image_path": image_path,
        "ep_idx": ep_idx,
        "frame_i": i,
        "is_last_step": is_last,
        # 归一化的标签(训练用)
        "target_action": normalized_target,
        "target_group":  get_group(normalized_target),
        "history_actions": normalized_history,
        # 审计追溯字段(原始值)
        "raw_target_action": raw_target,
        "raw_history_actions": raw_history,
        "target_was_aliased": (raw_target != normalized_target),
        # 通用字段
        "gpt_instruction": ep["gpt_instruction"],
        "subtask_list": subtask_list_str,
        "current_subtask_hint": "",
        # 双帧字段(双帧版需要)
        "prev_index": prev_id,
        "curr_index": curr_id,
        "prev_obs": prev_obs,
        "curr_obs": curr_obs,
        "prev_image_path": prev_img,
        "curr_image_path": curr_img,
    }


# ============================================================
# 分层抽样
# ============================================================
random.seed(SEED)
picked_by_action = defaultdict(list)
excluded_stats = defaultdict(lambda: defaultdict(int))

print(f"\n[抽样开始] seed={SEED}")
for action, quota in QUOTA.items():
    pool = candidates_by_action[action][:]
    random.shuffle(pool)
    for ep_idx, i in pool:
        if len(picked_by_action[action]) >= quota: break
        ok, reason, sample = validate_and_build(ep_idx, i)
        if not ok:
            excluded_stats[action][reason] += 1
            continue
        sample["selection_reason"] = (
            f"action_{action}_quota_fill_{len(picked_by_action[action])+1}_of_{quota}"
        )
        picked_by_action[action].append(sample)


# 检查是否凑齐
unfilled = [a for a, q in QUOTA.items() if len(picked_by_action[a]) < q]
if unfilled:
    print(f"\n❌ 错误: 以下动作未凑齐配额: {unfilled}")
    for a in unfilled:
        print(f"  动作 {a}: 想要 {QUOTA[a]} 条, 实际 {len(picked_by_action[a])} 条")
        print(f"    排除原因: {dict(excluded_stats[a])}")
    print(f"\n[fail-fast] 配额未凑齐,不落盘")
    sys.exit(1)
else:
    print(f"\n✅ 所有配额已凑齐")


# ============================================================
# 整理 + 落盘 (双帧版)
# ============================================================
samples_dual = []
sample_id = 0
local_alias_count = 0
for action in sorted(QUOTA.keys()):
    for s in picked_by_action[action]:
        s["id"] = sample_id
        if s["target_was_aliased"]:
            local_alias_count += 1
        samples_dual.append(s)
        sample_id += 1

dual_meta = {
    "experiment_id": TS,
    "seed": SEED,
    "generated_at": datetime.now().isoformat(),
    "variant": "dual_frame",
    "total_samples": len(samples_dual),
    "quota": QUOTA,
    "actually_picked": {a: len(picked_by_action[a]) for a in QUOTA},
    "excluded_stats": {a: dict(v) for a, v in excluded_stats.items()},
    "alias_count_picked": local_alias_count,
    "alias_count_dataset_total": alias_count,
    "data_source": DATA,
    "note": "Skipped actions 6 (strafe_left) and 7 (strafe_right) because dataset has 0 candidates for them.",
}

with open(OUT_DUAL, "w") as f:
    json.dump({"_meta": dual_meta, "samples": samples_dual}, f, ensure_ascii=False, indent=2)
print(f"\n[落盘 双帧版] {OUT_DUAL}")
print(f"  样本数: {len(samples_dual)}")


# ============================================================
# 转 curr_only 版本(curr 的语义改为 idx[i],详见 convert 转换规则)
# ============================================================
samples_currof = []
for s in samples_dual:
    is_last = s["is_last_step"]
    if is_last:
        new_curr_index      = s["curr_index"]
        new_curr_obs        = s["curr_obs"]
        new_curr_image_path = s["curr_image_path"]
        conversion_note     = "last_step: curr unchanged"
        assert new_curr_index == s["curr_index"]
    else:
        new_curr_index      = s["prev_index"]
        new_curr_obs        = s["prev_obs"]
        new_curr_image_path = s["prev_image_path"]
        conversion_note     = "non_last: new_curr <- old_prev"
        assert new_curr_index == s["prev_index"]

    new_s = {
        "id":                 s["id"],
        "image_path":         s["image_path"],
        "ep_idx":             s["ep_idx"],
        "frame_i":            s["frame_i"],
        "is_last_step":       is_last,
        # 归一化标签
        "target_action":      s["target_action"],
        "target_group":       s["target_group"],
        "history_actions":    s["history_actions"],
        # 审计字段
        "raw_target_action":  s["raw_target_action"],
        "raw_history_actions": s["raw_history_actions"],
        "target_was_aliased": s["target_was_aliased"],
        # 通用
        "gpt_instruction":    s["gpt_instruction"],
        "subtask_list":       s["subtask_list"],
        "current_subtask_hint": s["current_subtask_hint"],
        # curr-only 字段(curr 已经是 idx[i])
        "curr_index":         new_curr_index,
        "curr_obs":           new_curr_obs,
        "curr_image_path":    new_curr_image_path,
        # 元信息
        "selection_reason":   s["selection_reason"],
        "conversion_note":    conversion_note,
    }
    samples_currof.append(new_s)

# 校验 curr_only 版本
errors = []
for s in samples_currof:
    if not s["curr_obs"].strip():
        errors.append(f"sample {s['id']}: curr_obs 为空")
    if not os.path.isfile(s["curr_image_path"]):
        errors.append(f"sample {s['id']}: curr_image 不存在")
if errors:
    print("❌ curr_only 版校验失败:")
    for e in errors[:10]: print(f"  {e}")
    sys.exit(1)

currof_meta = dict(dual_meta)
currof_meta["variant"] = "curr_only"
currof_meta["conversion_rule"] = "non_last: new_curr=idx[i] (was old_prev); last: new_curr=idx[i] (unchanged)"
currof_meta["removed_fields"]  = ["prev_index", "prev_obs", "prev_image_path"]

with open(OUT_CURRONLY, "w") as f:
    json.dump({"_meta": currof_meta, "samples": samples_currof}, f, ensure_ascii=False, indent=2)
print(f"\n[落盘 curr_only 版] {OUT_CURRONLY}")
print(f"  样本数: {len(samples_currof)}")


# ============================================================
# 末尾汇总
# ============================================================
print("\n" + "="*70)
print("抽样总结")
print("="*70)
print(f"实验 ID (TS): {TS}")
print(f"双帧版    : {OUT_DUAL}")
print(f"curr_only : {OUT_CURRONLY}")
print(f"\n各动作分布(归一化后):")
for a in sorted(QUOTA.keys()):
    print(f"  动作 {a} ({get_group(a)}): {len(picked_by_action[a])} 条")
print(f"\n本轮抽到的样本中,被归一化(-1/-2 → 4/5)的: {local_alias_count} 条")

# 排除统计
print(f"\n[排除统计]")
all_excluded = defaultdict(int)
for a in excluded_stats:
    for r, c in excluded_stats[a].items():
        all_excluded[r] += c
if all_excluded:
    for r, c in sorted(all_excluded.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")
else:
    print(f"  (无排除)")

print(f"\n下一步: python3 generate_v1_smoke_curr_only.py --input {OUT_CURRONLY}")
