"""
sample_10.py
============
分层抽样 10 条样本作为 Prompt v1 小样本测试的输入。

抽样配额:
  动作 0/1/2/3/4/5 各 1 条; 动作 8/9 各 2 条; 共 10 条。

落盘:
  - smoke_input_10.json : 10 条样本(每条含完整字段,可直接喂给 generate 脚本)

设计原则:
  - seed=42 可复现
  - 失败补抽,直到每个动作配额凑齐
  - 最小过滤(obs 非空 + 图像文件存在)
  - 记录 selection_reason 和 excluded_stats 便于追溯
"""

import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
OUT_FILE  = os.path.join(OUT_DIR, "smoke_input_10.json")

# 抽样配额
QUOTA = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 8: 2, 9: 2}
TOTAL_TARGET = sum(QUOTA.values())   # 10


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
# 候选枚举: (episode_idx, frame_i, target_action)
# 每个候选是一个"选择某条轨迹的某一步"
# ============================================================
print("[枚举候选]")
candidates_by_action = defaultdict(list)
for ep_idx, ep in enumerate(data):
    acts = ep.get("action", [])
    idx_list = ep.get("index_list", [])
    for i, a in enumerate(acts):
        a_norm = normalize_action(a)        # -1/-2 -> 4/5
        if a_norm not in QUOTA: continue    # 只收 QUOTA 里需要的动作
        candidates_by_action[a_norm].append((ep_idx, i))

for a, lst in sorted(candidates_by_action.items()):
    print(f"  动作 {a}: 候选 {len(lst)} 个")



# ============================================================
# 动作分组(后续 filter 用于按族统计)
# ============================================================
ACTION_GROUPS = {
    0: "stop",
    1: "forward_family", 8: "forward_family", 9: "forward_family",
    2: "turn_family", 3: "turn_family",
    4: "vertical_family", 5: "vertical_family",
    6: "strafe_family", 7: "strafe_family",
}
def _get_group(a):
    return ACTION_GROUPS.get(a, "unknown")


# ============================================================
# 校验单条候选的合法性 (obs 非空 + 图像存在)
# 返回: (是否合法, 排除原因 or 完整字段 dict)
# ============================================================
def validate_and_build(ep_idx, i):
    ep = data[ep_idx]
    image_path = ep["image_path"]
    idx_list = ep["index_list"]
    acts = ep["action"]

    # 取 prev / curr 的 frame id
    # 中间步: prev=idx[i], curr=idx[i+1]
    # 最后步(stop): prev=idx[i-1], curr=idx[i]
    is_last = (i == len(acts) - 1)
    if is_last:
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

    # 图像文件存在
    prev_img = f"{IMG_BASE}/{image_path}/{prev_id}.png"
    curr_img = f"{IMG_BASE}/{image_path}/{curr_id}.png"
    if not os.path.isfile(prev_img): return False, "missing_image", None
    if not os.path.isfile(curr_img): return False, "missing_image", None

    # subtasks
    st_dict = sub_idx.get(image_path, {})
    if not isinstance(st_dict, dict) or not st_dict:
        return False, "missing_subtasks", None
    subtask_list_str = "\n".join(f"{k}. {v}" for k, v in st_dict.items())

    # history actions(已规范化 -1/-2)
    history = [normalize_action(a) for a in acts[:i]]

    return True, "ok", {
        "image_path": image_path,
        "ep_idx": ep_idx,
        "frame_i": i,
        "is_last_step": is_last,
        "target_action": normalize_action(acts[i]),
        "target_group": _get_group(normalize_action(acts[i])),
        "raw_target_action": acts[i],
        "gpt_instruction": ep["gpt_instruction"],
        "subtask_list": subtask_list_str,
        "history_actions": history,
        "current_subtask_hint": "",       # 一律空(拍板 A)
        "prev_index": prev_id,
        "curr_index": curr_id,
        "prev_obs": prev_obs,
        "curr_obs": curr_obs,
        "prev_image_path": prev_img,
        "curr_image_path": curr_img,
    }


# ============================================================
# 分层抽样 + 失败补抽
# ============================================================
random.seed(SEED)

picked_by_action = defaultdict(list)
excluded_stats = defaultdict(lambda: defaultdict(int))   # action -> reason -> count

print(f"\n[抽样开始] seed={SEED}")
for action, quota in QUOTA.items():
    pool = candidates_by_action[action][:]   # 拷贝
    random.shuffle(pool)
    for ep_idx, i in pool:
        if len(picked_by_action[action]) >= quota: break
        ok, reason, sample = validate_and_build(ep_idx, i)
        if not ok:
            excluded_stats[action][reason] += 1
            continue
        sample["selection_reason"] = f"action_{action}_quota_fill_{len(picked_by_action[action])+1}_of_{quota}"
        picked_by_action[action].append(sample)

# 检查是否凑齐
unfilled = [a for a, q in QUOTA.items() if len(picked_by_action[a]) < q]
if unfilled:
    print(f"\n❌ 错误: 以下动作未凑齐配额: {unfilled}")
    for a in unfilled:
        print(f"  动作 {a}: 想要 {QUOTA[a]} 条, 实际 {len(picked_by_action[a])} 条")
        print(f"    排除原因: {dict(excluded_stats[a])}")
    print(f"\n[fail-fast] 配额未凑齐,不落盘,需要扩大候选池或修复数据后重跑")
    import sys
    sys.exit(1)
else:
    print(f"\n✅ 所有配额已凑齐")


# ============================================================
# 整理 + 落盘
# ============================================================
samples = []
sample_id = 0
for action in sorted(QUOTA.keys()):
    for s in picked_by_action[action]:
        s["id"] = sample_id
        samples.append(s)
        sample_id += 1

output = {
    "_meta": {
        "seed": SEED,
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(samples),
        "quota": QUOTA,
        "actually_picked": {a: len(picked_by_action[a]) for a in QUOTA},
        "excluded_stats": {a: dict(v) for a, v in excluded_stats.items()},
        "data_source": DATA,
    },
    "samples": samples,
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n[落盘] {OUT_FILE}")
print(f"  样本数: {len(samples)}")
print(f"  各动作: {dict((a, len(picked_by_action[a])) for a in sorted(QUOTA))}")

# 排除统计汇总
print("\n[排除统计]")
all_excluded = defaultdict(int)
for a in excluded_stats:
    for r, c in excluded_stats[a].items():
        all_excluded[r] += c
if all_excluded:
    for r, c in sorted(all_excluded.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")
else:
    print("  (无排除)")
