#!/usr/bin/env python3
"""
merge_shards.py - Stage2 Round0 分片合并 + 去重 + 审计

用法:
  cd /home/ubuntu/data1/lyy/full_rlds_project-3/stage2
  python3 merge_shards.py --dry_run
  python3 merge_shards.py --id_mode evenodd

关键参数:
  --id_mode evenodd     # 适用于 i % 2 分片（你们当前 shard 脚本）
  --id_mode contiguous  # 适用于前半/后半分片
  --id_mode none        # 如果 sample_id 已是全局 ID，不做映射
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =========================
# 默认配置
# =========================
TOTAL_SAMPLES = 4500
NUM_SHARDS = 2
ROUND_ID = 0

GATE_FALLBACK_MAX_PCT = 2.0
GATE_UNKNOWN_MAX_PCT = 15.0
GATE_SMALL_GAP_MAX_PCT = 20.0
GATE_SMALL_GAP_THRESHOLD = 0.01
GATE_DUP_MAX = 0

# =========================
# 工具函数
# =========================


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_records(obj: Any) -> List[Dict[str, Any]]:
    # 兼容 list / dict(records|samples|data|items|pairs|candidates|all_candidates)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["records", "samples", "data", "items", "pairs", "candidates", "all_candidates"]:
            v = obj.get(k)
            if isinstance(v, list):
                return v
        for _, v in obj.items():
            if isinstance(v, list):
                return v
    raise ValueError("JSON 结构无法解析为 records list")


def to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        return int(str(x))
    except Exception:
        return None


def first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def get_sample_id(rec: Dict[str, Any]) -> Optional[int]:
    return to_int(first_not_none(rec.get("sample_id"), rec.get("sample_idx"), rec.get("id")))


def set_sample_id(rec: Dict[str, Any], sid: int) -> None:
    if "sample_id" in rec:
        rec["sample_id"] = sid
    elif "sample_idx" in rec:
        rec["sample_idx"] = sid
    else:
        rec["sample_id"] = sid

    # 若两个字段都存在，保持一致
    if "sample_id" in rec and "sample_idx" in rec:
        rec["sample_idx"] = sid


def get_frame_id(rec: Dict[str, Any]) -> Optional[str]:
    fid = first_not_none(rec.get("frame_idx"), rec.get("frame_id"), rec.get("index"))
    if fid is None:
        return None
    return str(fid)


def map_local_to_global(local_id: int, shard_id: int, id_mode: str, shard_size: int) -> int:
    if id_mode == "none":
        return local_id
    if id_mode == "contiguous":
        return local_id + shard_id * shard_size
    if id_mode == "evenodd":
        # shard0: 0,2,4...
        # shard1: 1,3,5...
        return local_id * NUM_SHARDS + shard_id
    raise ValueError(f"unknown id_mode={id_mode}")


def extract_error_type(rec: Dict[str, Any]) -> str:
    et = rec.get("error_type")
    if et is None and isinstance(rec.get("diagnosis"), dict):
        et = rec["diagnosis"].get("error_type")
    if et is None:
        return "missing"
    et = str(et).strip().lower()
    return et if et else "missing"


def extract_fallback(rec: Dict[str, Any]) -> bool:
    if rec.get("fallback") is True:
        return True
    d = rec.get("diagnosis")
    if isinstance(d, dict):
        if d.get("_is_fallback") is True:
            return True
        if d.get("is_fallback") is True:
            return True
    return False


def extract_gap(rec: Dict[str, Any]) -> Optional[float]:
    g = rec.get("score_gap")
    if g is not None:
        try:
            return float(g)
        except Exception:
            return None
    cs = rec.get("chosen_score")
    rs = rec.get("rejected_score")
    try:
        if cs is not None and rs is not None:
            return float(cs) - float(rs)
    except Exception:
        return None
    return None


def make_dedupe_key(rec: Dict[str, Any], fallback_idx: int) -> Tuple[Any, Any]:
    sid = get_sample_id(rec)
    fid = get_frame_id(rec)
    if sid is None:
        return ("NO_SID", fallback_idx)
    if fid is None:
        return (sid, None)
    return (sid, fid)


@dataclass
class MergeResult:
    merged: List[Dict[str, Any]]
    dup_removed: int


def merge_and_dedupe(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> MergeResult:
    out: List[Dict[str, Any]] = []
    seen = set()
    dup = 0

    for idx, rec in enumerate(a + b):
        key = make_dedupe_key(rec, idx)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        out.append(rec)

    # 稳定排序（先按 sample_id，再按 frame）
    def sort_key(r: Dict[str, Any]):
        sid = get_sample_id(r)
        if sid is None:
            sid = 10**18
        fid = get_frame_id(r)
        return (sid, "" if fid is None else fid)

    out.sort(key=sort_key)
    return MergeResult(merged=out, dup_removed=dup)


def audit(candidates: List[Dict[str, Any]], diagnosis: List[Dict[str, Any]], dup_total: int, total_samples: int) -> Dict[str, Any]:
    total_c = len(candidates)
    total_d = len(diagnosis)

    fb_count = sum(1 for r in diagnosis if extract_fallback(r))
    fb_pct = (fb_count / total_d * 100.0) if total_d > 0 else 0.0

    et_counter = Counter(extract_error_type(r) for r in diagnosis)
    unknown_count = et_counter.get("unknown", 0) + et_counter.get("missing", 0)
    unknown_pct = (unknown_count / total_d * 100.0) if total_d > 0 else 0.0

    gaps: List[float] = []
    neg_gap = 0
    for r in diagnosis:
        g = extract_gap(r)
        if g is None or not math.isfinite(g):
            continue
        gaps.append(g)
        if g < 0:
            neg_gap += 1

    small_gap = sum(1 for g in gaps if g < GATE_SMALL_GAP_THRESHOLD)
    small_gap_pct = (small_gap / len(gaps) * 100.0) if gaps else 0.0

    unique_diag_sid = {get_sample_id(r) for r in diagnosis if get_sample_id(r) is not None}
    coverage_pct = (len(unique_diag_sid) / total_samples * 100.0) if total_samples > 0 else 0.0

    gates = {
        "fallback_le_2pct": {
            "value": round(fb_pct, 3),
            "threshold": GATE_FALLBACK_MAX_PCT,
            "pass": fb_pct <= GATE_FALLBACK_MAX_PCT,
        },
        "unknown_le_15pct": {
            "value": round(unknown_pct, 3),
            "threshold": GATE_UNKNOWN_MAX_PCT,
            "pass": unknown_pct <= GATE_UNKNOWN_MAX_PCT,
        },
        "small_gap_le_20pct": {
            "value": round(small_gap_pct, 3),
            "threshold": GATE_SMALL_GAP_MAX_PCT,
            "pass": small_gap_pct <= GATE_SMALL_GAP_MAX_PCT,
        },
        "dup_eq_0": {
            "value": dup_total,
            "threshold": GATE_DUP_MAX,
            "pass": dup_total <= GATE_DUP_MAX,
        },
    }

    report = {
        "timestamp": now_str(),
        "counts": {
            "candidates": total_c,
            "diagnosis": total_d,
            "cand_diag_match": total_c == total_d,
            "unique_diag_sample_ids": len(unique_diag_sid),
            "expected_total_samples": total_samples,
            "coverage_pct": round(coverage_pct, 3),
        },
        "fallback": {"count": fb_count, "pct": round(fb_pct, 3)},
        "error_type_distribution": dict(et_counter.most_common()),
        "unknown": {"count": unknown_count, "pct": round(unknown_pct, 3)},
        "score_gap": {
            "with_gap_count": len(gaps),
            "small_gap_count": small_gap,
            "small_gap_pct": round(small_gap_pct, 3),
            "negative_gap_count": neg_gap,
            "mean": round(sum(gaps) / len(gaps), 6) if gaps else None,
            "min": round(min(gaps), 6) if gaps else None,
            "max": round(max(gaps), 6) if gaps else None,
            "threshold": GATE_SMALL_GAP_THRESHOLD,
        },
        "gates": gates,
        "all_gates_pass": all(g["pass"] for g in gates.values()),
    }
    return report


def print_report(r: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("Round 0 Audit")
    print("=" * 60)
    print(f"counts: cand={r['counts']['candidates']}, diag={r['counts']['diagnosis']}, "
          f"match={r['counts']['cand_diag_match']}, coverage={r['counts']['coverage_pct']}%")
    print(f"fallback: {r['fallback']['count']} ({r['fallback']['pct']}%)")
    print(f"unknown: {r['unknown']['count']} ({r['unknown']['pct']}%)")
    sg = r["score_gap"]
    print(f"gap: with={sg['with_gap_count']}, small={sg['small_gap_count']} ({sg['small_gap_pct']}%), "
          f"neg={sg['negative_gap_count']}, mean={sg['mean']}")
    print("gates:")
    for k, v in r["gates"].items():
        print(f"  {k}: {v['value']} <= {v['threshold']} -> {'PASS' if v['pass'] else 'FAIL'}")
    print(f"overall: {'PASS' if r['all_gates_pass'] else 'FAIL'}")
    print("=" * 60 + "\n")


# =========================
# 主流程
# =========================


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_base = script_dir

    parser = argparse.ArgumentParser(description="Merge Stage2 shard outputs with audit")
    parser.add_argument("--round_id", type=int, default=ROUND_ID)
    parser.add_argument("--total_samples", type=int, default=TOTAL_SAMPLES)
    parser.add_argument("--id_mode", choices=["evenodd", "contiguous", "none"], default="evenodd")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--shard0_dir", type=str, default=str(default_base / "checkpoints_shard0"))
    parser.add_argument("--shard1_dir", type=str, default=str(default_base / "checkpoints_shard1"))
    parser.add_argument("--output_dir", type=str, default=str(default_base / "checkpoints"))
    args = parser.parse_args()

    rid = args.round_id
    shard0_dir = Path(args.shard0_dir)
    shard1_dir = Path(args.shard1_dir)
    out_dir = Path(args.output_dir)

    cand_name = f"round_{rid}_candidates.json"
    diag_name = f"round_{rid}_diagnosis_records.json"
    prog_name = f"round_{rid}_progress.json"

    for d in [shard0_dir, shard1_dir]:
        if not d.exists():
            raise FileNotFoundError(f"missing dir: {d}")

    c0p = shard0_dir / cand_name
    d0p = shard0_dir / diag_name
    p0p = shard0_dir / prog_name

    c1p = shard1_dir / cand_name
    d1p = shard1_dir / diag_name
    p1p = shard1_dir / prog_name

    for p in [c0p, d0p, p0p, c1p, d1p, p1p]:
        if not p.exists():
            raise FileNotFoundError(f"missing file: {p}")

    # 读取
    cand0 = unwrap_records(load_json(c0p))
    cand1 = unwrap_records(load_json(c1p))
    diag0 = unwrap_records(load_json(d0p))
    diag1 = unwrap_records(load_json(d1p))
    prog0 = load_json(p0p)
    prog1 = load_json(p1p)

    print(f"load shard0: cand={len(cand0)}, diag={len(diag0)}, completed={prog0.get('completed_samples')}")
    print(f"load shard1: cand={len(cand1)}, diag={len(diag1)}, completed={prog1.get('completed_samples')}")

    # 深拷贝，避免改原对象
    cand0 = copy.deepcopy(cand0)
    cand1 = copy.deepcopy(cand1)
    diag0 = copy.deepcopy(diag0)
    diag1 = copy.deepcopy(diag1)

    shard_size = args.total_samples // NUM_SHARDS

    # ID 映射
    def remap(records: List[Dict[str, Any]], shard_id: int):
        miss = 0
        for r in records:
            sid = get_sample_id(r)
            if sid is None:
                miss += 1
                continue
            gid = map_local_to_global(sid, shard_id=shard_id, id_mode=args.id_mode, shard_size=shard_size)
            set_sample_id(r, gid)
            r["_shard_id"] = shard_id
        return miss

    miss_c0 = remap(cand0, 0)
    miss_c1 = remap(cand1, 1)
    miss_d0 = remap(diag0, 0)
    miss_d1 = remap(diag1, 1)

    print(f"remap mode={args.id_mode}, missing sid: cand({miss_c0+miss_c1}), diag({miss_d0+miss_d1})")

    # 合并去重
    mc = merge_and_dedupe(cand0, cand1)
    md = merge_and_dedupe(diag0, diag1)
    dup_total = mc.dup_removed + md.dup_removed

    # 审计
    report = audit(mc.merged, md.merged, dup_total=dup_total, total_samples=args.total_samples)
    report["merge_info"] = {
        "round_id": rid,
        "id_mode": args.id_mode,
        "shard_size": shard_size,
        "shard0_candidates": len(cand0),
        "shard1_candidates": len(cand1),
        "shard0_diagnosis": len(diag0),
        "shard1_diagnosis": len(diag1),
        "dup_candidates_removed": mc.dup_removed,
        "dup_diagnosis_removed": md.dup_removed,
        "missing_sample_id_candidates": miss_c0 + miss_c1,
        "missing_sample_id_diagnosis": miss_d0 + miss_d1,
    }

    print_report(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / f"round_{rid}_audit_report.json"
    atomic_write_json(audit_path, report)
    print(f"write: {audit_path}")

    if args.dry_run:
        print("dry_run=true, skip writing merged files.")
        return 0

    # 输出清理（去掉内部字段）
    def clean(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in records:
            rc = {k: v for k, v in r.items() if not k.startswith("_")}
            out.append(rc)
        return out

    merged_cand = clean(mc.merged)
    merged_diag = clean(md.merged)

    out_cand = out_dir / f"round_{rid}_candidates.json"
    out_diag_records = out_dir / f"round_{rid}_diagnosis_records.json"
    out_diag_alias = out_dir / f"round_{rid}_diagnosis.json"
    out_prog = out_dir / f"round_{rid}_progress.json"

    atomic_write_json(out_cand, merged_cand)
    atomic_write_json(out_diag_records, merged_diag)
    atomic_write_json(out_diag_alias, merged_diag)

    progress = {
        "completed_samples": len(merged_cand),
        "total_pairs": len(merged_cand),
        "total_diagnosis": len(merged_diag),
        "timestamp": datetime.now().isoformat(),
        "merged_from": [str(shard0_dir), str(shard1_dir)],
        "id_mode": args.id_mode,
        "is_complete": len(merged_cand) >= args.total_samples,
    }
    atomic_write_json(out_prog, progress)

    print(f"write: {out_cand} ({len(merged_cand)})")
    print(f"write: {out_diag_records} ({len(merged_diag)})")
    print(f"write: {out_diag_alias} ({len(merged_diag)})")
    print(f"write: {out_prog}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
