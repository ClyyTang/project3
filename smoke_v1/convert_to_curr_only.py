"""
convert_to_curr_only.py
========================
把 smoke_input_10.json 转换成 curr_only 语义版本。

转换规则:
  - 非 last (is_last=False):
      new_curr_index      = old_prev_index    (idx[i])
      new_curr_obs        = old_prev_obs
      new_curr_image_path = old_prev_image_path
  - last (is_last=True):
      new_curr_* = old_curr_*  (idx[i],已经是当前帧)

去掉 prev_* 字段(curr_only 模式不需要)。

输出: smoke_input_10_curr_only.json
"""

import json
import os
import sys
from datetime import datetime

IN_PATH  = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/smoke_input_10.json"
OUT_PATH = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/smoke_input_10_curr_only.json"

with open(IN_PATH) as f:
    data = json.load(f)

new_samples = []
for s in data["samples"]:
    is_last = s.get("is_last_step", False)
    
    if is_last:
        new_curr_index      = s["curr_index"]
        new_curr_obs        = s["curr_obs"]
        new_curr_image_path = s["curr_image_path"]
        conversion_note     = "last_step: curr unchanged"
        # 硬断言:last 步 new_curr 必须等于 old_curr
        assert new_curr_index == s["curr_index"], \
            f"sample {s['id']}: last step 转换错,期望 new_curr=={s['curr_index']},实际 {new_curr_index}"
    else:
        new_curr_index      = s["prev_index"]
        new_curr_obs        = s["prev_obs"]
        new_curr_image_path = s["prev_image_path"]
        conversion_note     = "non_last: new_curr <- old_prev"
        # 硬断言:非 last 步 new_curr 必须等于 old_prev
        assert new_curr_index == s["prev_index"], \
            f"sample {s['id']}: non-last 步转换错,期望 new_curr=={s['prev_index']},实际 {new_curr_index}"
    
    new_s = {
        "id":                 s["id"],
        "image_path":         s["image_path"],
        "ep_idx":             s.get("ep_idx"),
        "frame_i":            s.get("frame_i"),
        "is_last_step":       is_last,
        "target_action":      s["target_action"],
        "raw_target_action":  s.get("raw_target_action"),
        "target_group":       s.get("target_group", "unknown"),  # 师兄补丁 1
        "gpt_instruction":    s["gpt_instruction"],
        "subtask_list":       s["subtask_list"],
        "history_actions":    s["history_actions"],
        "current_subtask_hint": s.get("current_subtask_hint", ""),
        # 新字段(curr_only)
        "curr_index":         new_curr_index,
        "curr_obs":           new_curr_obs,
        "curr_image_path":    new_curr_image_path,
        # 元信息
        "selection_reason":   s.get("selection_reason", ""),
        "conversion_note":    conversion_note,
    }
    new_samples.append(new_s)

# 校验:curr_obs 非空 + 图像存在(fail-fast)
errors = []
for s in new_samples:
    if not s["curr_obs"].strip():
        errors.append(f"sample {s['id']}: curr_obs 为空")
    if not os.path.isfile(s["curr_image_path"]):
        errors.append(f"sample {s['id']}: curr_image 不存在: {s['curr_image_path']}")

if errors:
    print("❌ 校验失败:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)

output = {
    "_meta": {
        "converted_at":       datetime.now().isoformat(),
        "source_file":        IN_PATH,
        "conversion_rule":    "non_last: new_curr=idx[i] (was old_prev); last: new_curr=idx[i] (unchanged)",
        "total_samples":      len(new_samples),
        "removed_fields":     ["prev_index", "prev_obs", "prev_image_path"],
    },
    "samples": new_samples,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成: {OUT_PATH}")
print(f"  总样本: {len(new_samples)}")

# 对照表
print(f"\n{'id':>3} {'i':>3} {'is_last':>7} {'old_prev':>22} {'old_curr':>22} {'new_curr':>22}")
print("-" * 90)
old_idx = {s["id"]: s for s in data["samples"]}
for s in new_samples:
    o = old_idx[s["id"]]
    print(f"{s['id']:>3} {s['frame_i']:>3} {str(s['is_last_step']):>7} {o['prev_index']:>22} {o['curr_index']:>22} {s['curr_index']:>22}")
