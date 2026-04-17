"""
filter_v1.py
============
对 generate_v1_smoke_curr_only_latest.json 做硬规则过滤。

7 条硬规则(全部通过才算 keep):
  R1. xml_parse_ok          : <next_action>X</next_action> 可解析
  R2. action_strict_match   : 解析出的 pred == GT (strict, 1/8/9 不互通)
  R3. has_4_sections        : [Progress]/[Observation]/[Reasoning]/[Decision] 全在
  R4. direction_word_in_reasoning : [Reasoning] 段内含至少 1 个方向词
  R5. distance_word_in_reasoning  : [Reasoning] 段内含至少 1 个距离词
  R6. alternative_line_valid: [Reasoning] 段内恰好 1 行 Alternative
                              格式 'Alternative considered: NAME (ID); Rejected because <reason>.'
                              NAME/ID 在动作表内, NAME 单一(不允许 or/and/,)
                              NAME ≠ Decision, ID ≠ pred
                              <reason> 含至少 1 个方向词或距离词
  R7. decision_next_match   : [Decision] 中的动作名 与 <next_action>X</next_action>
                              对应的动作名一致 (防止"段落对但标签错")

软统计(不卡门槛,只看分布):
  S1. A2/C token Jaccard 相似度
  S2. 否定式距离词出现率 ('not far' / 'no longer far')
  S3. mixed 模式动作匹配(1/8/9 互通,看强 vs 严差距)

输出:
  filter_v1_curr_only_latest.json  + filter_v1_curr_only_{ts}.json
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

# 引入动作语义
sys.path.insert(0, "/home/ubuntu/data1/lyy/full_rlds_project-3")
from action_semantics import actions_match  # 仅用于软统计 mixed

# ============================================================
# 配置
# ============================================================
INPUT_FILE = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/outputs/generate_v1_smoke_curr_only_latest.json"
OUT_DIR    = "/home/ubuntu/data1/lyy/full_rlds_project-3/smoke_v1/outputs"

DIRECTION_WORDS = {
    "left", "right", "ahead", "front", "behind",
    "up", "down", "above", "below",
}
DISTANCE_WORDS = {
    "near", "nearby", "close", "closer",
    "far", "distant", "farther", "further",
    "short-range", "mid-range", "long-range",
    "meter", "meters",
}
NEGATIVE_DIST_PATTERNS = [
    r"\bnot\s+far\b",
    r"\bno\s+longer\s+far\b",
    r"\bnot\s+distant\b",
    r"\bno\s+longer\s+distant\b",
]

# 动作表
ACTION_NAMES = {
    0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right",
    4: "ascend", 5: "descend", 6: "strafe_left", 7: "strafe_right",
    8: "fast_forward", 9: "super_forward",
}
NAME_TO_ID = {v: k for k, v in ACTION_NAMES.items()}

# v3 冲突词表(硬规则,按动作)
CONFLICT_WORDS_BY_ACTION = {
    0: ["far", "distant", "long-range", "long range",
        "continue forward", "proceed toward", "proceed towards",
        "still need to approach", "not yet close"],
    1: ["reached", "arrived", "already at", "goal is reached",
        "stop here", "very far", "long-range", "long range"],
    2: [], 3: [],
    4: ["descend", "downward", "moving down", "go down", "lower"],
    5: ["ascend", "upward", "moving up", "go up", "rise", "higher"],
    6: [], 7: [],
    8: ["reached", "arrived", "already at", "goal is reached",
        "very close", "stop now"],
    9: ["reached", "arrived", "already at", "goal is reached",
        "very close", "stop now"],
}

# v3 混淆替代集
CONFUSION_SET = {
    0: [1, 8], 1: [8, 9], 2: [3], 3: [2],
    4: [5], 5: [4], 6: [7, 2], 7: [6, 3],
    8: [1, 9], 9: [8, 1],
}

# 否定前缀模式(检测 "not reached yet" 等,不能误判为冲突)
NEGATION_PATTERN = re.compile(
    r"\b(?:not|no longer|haven'?t|have not|has not|yet to|still (?:need|have|needs))\b",
    re.IGNORECASE
)


# ============================================================
# 解析工具
# ============================================================
def extract_next_action(text: str):
    """提取 <next_action>X</next_action> 中的整数,失败返回 None"""
    if not text:
        return None
    m = re.search(r'<next_action>\s*(\d+)\s*</next_action>', text)
    return int(m.group(1)) if m else None


def extract_section(text: str, section_name: str):
    """
    从 thinking 中抠出某一段的内容(纯文本)。
    section_name 是 "[Progress]" "[Observation]" 等。
    返回 None 表示没找到。
    
    锚定到行首(避免 thinking 正文里误匹配 [Decision] 等字样)。
    抠到下一个 [XXX] 段或 </thinking> 为止。
    """
    if not text:
        return None
    # (?m)^ 锚定行首,允许前导空白
    pattern = r"(?m)^\s*" + re.escape(section_name) + r"\s*(.*?)(?=^\s*\[(?:Progress|Observation|Reasoning|Decision)\]|^\s*</thinking>|\Z)"
    m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip()


def find_alternative_lines(reasoning_text: str):
    """从 [Reasoning] 段内找所有以 'Alternative considered:' 开头的行(大小写不敏感)"""
    if not reasoning_text:
        return []
    lines = []
    for line in reasoning_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("alternative considered:"):
            lines.append(stripped)
    return lines


def parse_alternative_line(line: str):
    """
    解析 Alternative 行,返回 (action_name, action_id, reason) 或 None。
    格式: "Alternative considered: NAME (ID); Rejected because REASON."
    大小写不敏感(关键字),但 NAME 保留原样供后续校验。
    """
    pattern = r"^Alternative considered:\s+(\S+)\s+\((\d+)\)\s*;\s*Rejected because\s+(.+?)\.?$"
    m = re.match(pattern, line, re.IGNORECASE)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3).strip()


def extract_decision_action_name(decision_text: str):
    """
    从 [Decision] 段抠出动作名(如 "ascend")。
    Decision 段可能写成: "ascend" 或 "The action is ascend." 等。
    我们用宽松匹配:在动作表里找第一个出现的动作名。
    """
    if not decision_text:
        return None
    text_lower = decision_text.lower()
    # 优先匹配两个词的(避免 forward 误中 fast_forward/super_forward)
    long_first = sorted(NAME_TO_ID.keys(), key=lambda x: -len(x))
    for name in long_first:
        if re.search(rf'\b{re.escape(name)}\b', text_lower):
            return name
    return None


# ============================================================
# 7 条硬规则检查
# ============================================================
def check_one_result(r: dict) -> dict:
    """返回 {fail_reasons: [...], rule_results: {R1: bool, ...}, parsed: {...}}"""
    text = r.get("response_text") or ""
    gt   = r.get("target_action")

    fail_reasons = []
    rule_results = {}
    parsed = {}

    # R1: XML 解析
    pred = extract_next_action(text)
    rule_results["R1_xml_parse"] = pred is not None
    if pred is None:
        fail_reasons.append("R1_xml_parse_fail")
    parsed["pred"] = pred

    # R2: strict action match
    if pred is None:
        rule_results["R2_action_strict"] = False
        fail_reasons.append("R2_action_strict_fail(no_pred)")
    else:
        ok2 = (pred == gt)
        rule_results["R2_action_strict"] = ok2
        if not ok2:
            fail_reasons.append(f"R2_action_strict_fail(pred={pred},gt={gt})")

    # R3: 4 段齐全
    sections = {}
    for sec in ["[Progress]", "[Observation]", "[Reasoning]", "[Decision]"]:
        sections[sec] = extract_section(text, sec)
    missing = [s for s, v in sections.items() if not v]
    rule_results["R3_4_sections"] = (len(missing) == 0)
    if missing:
        fail_reasons.append(f"R3_missing_sections({','.join(missing)})")
    parsed["sections"] = {k: (v[:80] + "..." if v and len(v) > 80 else v) for k, v in sections.items()}

    reasoning = sections.get("[Reasoning]") or ""
    decision  = sections.get("[Decision]") or ""

    # R4: direction word in reasoning
    reasoning_lower = reasoning.lower()
    dir_hits = [w for w in DIRECTION_WORDS if re.search(rf'\b{re.escape(w)}\b', reasoning_lower)]
    rule_results["R4_direction_word"] = len(dir_hits) > 0
    if not dir_hits:
        fail_reasons.append("R4_no_direction_word_in_reasoning")
    parsed["direction_hits"] = dir_hits

    # R5: distance word in reasoning
    dist_hits = [w for w in DISTANCE_WORDS if re.search(rf'\b{re.escape(w)}\b', reasoning_lower)]
    rule_results["R5_distance_word"] = len(dist_hits) > 0
    if not dist_hits:
        fail_reasons.append("R5_no_distance_word_in_reasoning")
    parsed["distance_hits"] = dist_hits

    # R6: alternative line valid
    alt_lines = find_alternative_lines(reasoning)
    decision_name = extract_decision_action_name(decision)
    parsed["decision_name"] = decision_name
    parsed["alternative_lines_count"] = len(alt_lines)

    if len(alt_lines) != 1:
        rule_results["R6_alternative_line"] = False
        fail_reasons.append(f"R6_alternative_count({len(alt_lines)},need_exactly_1)")
    else:
        parsed_alt = parse_alternative_line(alt_lines[0])
        if parsed_alt is None:
            rule_results["R6_alternative_line"] = False
            fail_reasons.append("R6_alternative_format_invalid")
        else:
            alt_name, alt_id, alt_reason = parsed_alt
            parsed["alternative"] = {"name": alt_name, "id": alt_id, "reason": alt_reason[:80]}
            errs = []
            # NAME/ID 一致性 + 在动作表内
            if alt_name not in NAME_TO_ID:
                errs.append(f"alt_name_unknown({alt_name})")
            elif NAME_TO_ID[alt_name] != alt_id:
                errs.append(f"alt_name_id_mismatch({alt_name}!={alt_id})")
            # NAME 单一(不能含 or/and/,)
            if re.search(r'\b(or|and)\b|,', alt_name):
                errs.append(f"alt_name_multi({alt_name})")
            # NAME != Decision, ID != pred
            if decision_name and alt_name == decision_name:
                errs.append(f"alt_name_eq_decision({alt_name})")
            if pred is not None and alt_id == pred:
                errs.append(f"alt_id_eq_pred({alt_id})")
            # v3 模式:alt 必须来自 confusion_set(按 gt 校验)
            if gt is not None and gt in CONFUSION_SET:
                allowed = CONFUSION_SET[gt]
                if alt_id not in allowed:
                    errs.append(f"alt_not_in_confusion_set(alt={alt_id},allowed={allowed})")
            # reason 含方向词或距离词
            reason_lower = alt_reason.lower()
            reason_has_dir = any(re.search(rf'\b{re.escape(w)}\b', reason_lower) for w in DIRECTION_WORDS)
            reason_has_dist = any(re.search(rf'\b{re.escape(w)}\b', reason_lower) for w in DISTANCE_WORDS)
            if not (reason_has_dir or reason_has_dist):
                errs.append("alt_reason_no_direction_or_distance")

            if errs:
                rule_results["R6_alternative_line"] = False
                fail_reasons.append(f"R6_alternative_invalid({';'.join(errs)})")
            else:
                rule_results["R6_alternative_line"] = True

    # R7: decision name 与 <next_action> id 一致
    if decision_name is None:
        rule_results["R7_decision_next_match"] = False
        fail_reasons.append("R7_decision_name_not_extractable")
    elif pred is None:
        rule_results["R7_decision_next_match"] = False
        fail_reasons.append("R7_pred_none(cannot_compare)")
    else:
        expected_pred = NAME_TO_ID.get(decision_name)
        if expected_pred is None:
            rule_results["R7_decision_next_match"] = False
            fail_reasons.append(f"R7_decision_name_unknown({decision_name})")
        elif expected_pred != pred:
            rule_results["R7_decision_next_match"] = False
            fail_reasons.append(f"R7_decision_next_mismatch(decision={decision_name}({expected_pred}),next_action={pred})")
        else:
            rule_results["R7_decision_next_match"] = True

    # R8: [Observation] 不允许出现动作名/动作ID(防循环论证)
    obs_text = sections.get("[Observation]") or ""
    obs_lower = obs_text.lower()
    r8_violations = []
    if obs_text:
        # 检查动作名
        for aid, aname in ACTION_NAMES.items():
            if re.search(rf"\b{re.escape(aname)}\b", obs_lower):
                r8_violations.append(aname)
        # 检查动作 ID 模式(如 "action 9" "action: 9")
        if re.search(r"\baction\s*[:\s]\s*\d\b", obs_lower):
            r8_violations.append("action_id_pattern")
    rule_results["R8_obs_no_action"] = (len(r8_violations) == 0)
    if r8_violations:
        fail_reasons.append(f"R8_obs_contains_action({','.join(r8_violations)})")
    parsed["r8_violations"] = r8_violations

    # R9: 冲突词检查(只在 [Reasoning],否定感知)
    conflict_list = CONFLICT_WORDS_BY_ACTION.get(gt, [])
    r9_hits = []
    if reasoning and conflict_list:
        for cw in conflict_list:
            for m_cw in re.finditer(re.escape(cw), reasoning_lower):
                # 往前 30 字符看有没有否定
                context = reasoning_lower[max(0, m_cw.start()-30):m_cw.start()]
                if NEGATION_PATTERN.search(context):
                    continue  # 否定修饰,不算冲突
                r9_hits.append(cw)
    rule_results["R9_no_conflict_words"] = (len(r9_hits) == 0)
    if r9_hits:
        fail_reasons.append(f"R9_conflict_words({','.join(set(r9_hits))})")
    parsed["r9_conflict_hits"] = sorted(set(r9_hits))

    keep = (len(fail_reasons) == 0)
    return {
        "keep": keep,
        "fail_reasons": fail_reasons,
        "rule_results": rule_results,
        "parsed": parsed,
    }


# ============================================================
# 软统计(不卡门槛,只看分布)
# ============================================================
def tokenize(text: str) -> set:
    """简单分词:小写 + 取所有 \w+"""
    if not text:
        return set()
    return set(re.findall(r'\w+', text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def soft_stats(results: list, filter_outcomes: list) -> dict:
    """计算软统计:Jaccard, 否定距离词, mixed match"""
    # 按 sample_id 分组,看 A2 vs C
    by_sid = defaultdict(dict)
    for r in results:
        by_sid[r["sample_id"]][r["mode"]] = r

    jaccards = []
    for sid, modes in by_sid.items():
        if "A2" in modes and "C" in modes:
            ta = tokenize(modes["A2"].get("response_text") or "")
            tc = tokenize(modes["C"].get("response_text") or "")
            jaccards.append((sid, jaccard(ta, tc)))

    # 否定距离词
    neg_dist_count = 0
    for r in results:
        text = (r.get("response_text") or "").lower()
        for pat in NEGATIVE_DIST_PATTERNS:
            if re.search(pat, text):
                neg_dist_count += 1
                break

    # mixed match
    strict_hits = mixed_hits = 0
    for r in results:
        pred = extract_next_action(r.get("response_text") or "")
        gt = r.get("target_action")
        if pred is None: continue
        if pred == gt: strict_hits += 1
        if actions_match(pred, gt, mode="mixed"): mixed_hits += 1

    return {
        "a2_c_jaccard": {
            "per_sample": [(sid, round(j, 3)) for sid, j in jaccards],
            "mean": round(sum(j for _, j in jaccards) / max(1, len(jaccards)), 3),
        },
        "negative_distance_word_count": neg_dist_count,
        "strict_match_count": strict_hits,
        "mixed_match_count": mixed_hits,
        "strict_vs_mixed_gap": mixed_hits - strict_hits,
    }


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=INPUT_FILE,
                        help="输入文件,默认 INPUT_FILE 常量")
    parser.add_argument("--min-results", type=int, default=20,
                        help="期望最少结果条数,少于则警告/退出")
    parser.add_argument("--allow-fewer", action="store_true",
                        help="允许少于 min-results 时继续(否则 exit)")
    args = parser.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"❌ 输入文件不存在: {in_path}")
        sys.exit(1)
    print(f"[读取] {in_path}")
    data = json.load(open(in_path))
    results = data["results"]
    print(f"  共 {len(results)} 条")

    # 输入条数保护
    if len(results) < args.min_results:
        msg = f"⚠️ 输入条数 {len(results)} < 期望 {args.min_results}"
        if args.allow_fewer:
            print(f"{msg} (allow-fewer 模式,继续)")
        else:
            print(f"{msg}")
            print(f"   如果你确实想跑这个文件,加 --allow-fewer")
            print(f"   或用 --input 指定 timestamp 版本(目录: {OUT_DIR})")
            sys.exit(1)

    # 跑硬规则
    filter_outcomes = []
    for r in results:
        outcome = check_one_result(r)
        merged = {**r, **outcome}
        filter_outcomes.append(merged)

    # 统计
    keep_count = sum(1 for x in filter_outcomes if x["keep"])
    print(f"\n[硬规则结果] {keep_count}/{len(filter_outcomes)} keep")

    # 按 mode 拆
    for mode in ["A2", "C"]:
        sub = [x for x in filter_outcomes if x["mode"] == mode]
        sub_keep = sum(1 for x in sub if x["keep"])
        print(f"  {mode}: {sub_keep}/{len(sub)}")

    # 失败原因分布(按规则)
    rule_fail = Counter()
    for x in filter_outcomes:
        if x["keep"]: continue
        for reason in x["fail_reasons"]:
            # 取规则前缀(R1/R2/...)
            rule_id = reason.split("_", 1)[0]
            rule_fail[rule_id] += 1
    print(f"\n[失败原因分布(按规则)]")
    for rid in sorted(rule_fail.keys()):
        print(f"  {rid}: {rule_fail[rid]} 条")

    # 详细失败原因
    print(f"\n[详细失败原因]")
    detail_fail = Counter()
    for x in filter_outcomes:
        for reason in x["fail_reasons"]:
            detail_fail[reason] += 1
    for reason, cnt in detail_fail.most_common():
        print(f"  {cnt:>3}x  {reason}")

    # 软统计
    print(f"\n[软统计]")
    soft = soft_stats(results, filter_outcomes)
    print(f"  A2/C Jaccard 均值: {soft['a2_c_jaccard']['mean']}")
    print(f"  否定距离词出现条数: {soft['negative_distance_word_count']}")
    print(f"  strict match: {soft['strict_match_count']} / mixed match: {soft['mixed_match_count']}  (gap={soft['strict_vs_mixed_gap']})")

    # 落盘
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "_meta": {
            "timestamp": ts,
            "input_file": in_path,
            "input_meta": data.get("_meta", {}),
            "total_results": len(filter_outcomes),
            "keep_count": keep_count,
            "by_mode": {
                mode: {
                    "total": sum(1 for x in filter_outcomes if x["mode"] == mode),
                    "keep":  sum(1 for x in filter_outcomes if x["mode"] == mode and x["keep"]),
                } for mode in ["A2", "C"]
            },
            "rule_fail_count": dict(rule_fail),
            "detail_fail_count": dict(detail_fail),
            "soft_stats": soft,
        },
        "results": filter_outcomes,
    }
    ts_path = os.path.join(OUT_DIR, f"filter_v1_curr_only_{ts}.json")
    latest = os.path.join(OUT_DIR, "filter_v1_curr_only_latest.json")
    with open(ts_path, "w") as f: json.dump(out, f, ensure_ascii=False, indent=2)
    with open(latest, "w") as f: json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[落盘]")
    print(f"  {ts_path}")
    print(f"  {latest}")


if __name__ == "__main__":
    main()
