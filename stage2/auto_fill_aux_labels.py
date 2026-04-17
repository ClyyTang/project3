#!/usr/bin/env python3
from pathlib import Path
import argparse, json, datetime, shutil

def unwrap(obj):
    if isinstance(obj, list): return obj, None, None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list): return v, k, obj
    raise RuntimeError("unsupported json shape")

def wrap(records, key, tpl):
    if key is None: return records
    out = dict(tpl) if isinstance(tpl, dict) else {}
    out[key] = records
    return out

def f01(x, d=0.5):
    try: x = float(x)
    except: x = d
    return max(0.0, min(1.0, x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--root", type=str, default="/home/ubuntu/data1/lyy/full_rlds_project-3")
    args = ap.parse_args()

    root = Path(args.root)
    r = args.round
    cand_p = root / "stage2" / "checkpoints" / f"round_{r}_candidates.json"
    diag_p = root / "stage2" / "checkpoints" / f"round_{r}_diagnosis_records.json"
    aux_p  = root / "data" / f"auxiliary_labels_round{r}.json"

    if not cand_p.exists():
        print(f"[aux_hook] skip: no candidates file: {cand_p}")
        return
    if not diag_p.exists():
        print(f"[aux_hook] skip: no diagnosis file: {diag_p}")
        return

    cand_obj = json.load(open(cand_p, "r", encoding="utf-8"))
    diag_obj = json.load(open(diag_p, "r", encoding="utf-8"))
    pairs, pkey, ptpl = unwrap(cand_obj)
    diags, _, _ = unwrap(diag_obj)

    total = len(pairs)
    before = sum(1 for p in pairs if isinstance(p, dict) and p.get("aux_labels") is not None)
    if total > 0 and before == total:
        print(f"[aux_hook] round{r}: already complete {before}/{total}")
        return

    diag_map = {}
    for d in diags:
        if isinstance(d, dict):
            sid = d.get("sample_id", d.get("sample_idx"))
            if sid is not None:
                diag_map[str(sid)] = d

    emap = {"unknown":0, "perception":1, "comprehension":2, "reasoning":3, "decision":4}

    for p in pairs:
        if not isinstance(p, dict): 
            continue
        if p.get("aux_labels") is not None:
            continue
        sid = p.get("sample_id", p.get("sample_idx"))
        d = diag_map.get(str(sid), {})
        et = str(d.get("error_type", "unknown")).lower()
        direction = emap.get(et, 0)
        score = f01(d.get("chosen_score", p.get("chosen_score", 0.5)))
        gap = abs(float(d.get("score_gap", p.get("score_gap", 0.0) or 0.0)))
        kw = [0.0] * 34
        kw[direction] = 1.0
        kw[10] = 1.0 if gap < 0.01 else 0.0
        kw[11] = 1.0 if et == "unknown" else 0.0
        p["aux_labels"] = {
            "keywords": kw,
            "direction": int(direction),
            "cot_quality": score,
            "action_validity": f01(0.4 + 0.6 * min(gap, 1.0)),
        }

    after = sum(1 for p in pairs if isinstance(p, dict) and p.get("aux_labels") is not None)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    arc = root / "stage2" / "checkpoints" / f"_archive_auxhook_round{r}_{ts}"
    arc.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cand_p, arc / cand_p.name)

    json.dump(wrap(pairs, pkey, ptpl), open(cand_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    aux_map = {}
    for p in pairs:
        if isinstance(p, dict):
            sid = p.get("sample_id", p.get("sample_idx"))
            if sid is not None and p.get("aux_labels") is not None:
                aux_map[str(sid)] = p["aux_labels"]
    aux_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(aux_map, open(aux_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[aux_hook] round{r}: {before}/{total} -> {after}/{total}")
    print(f"[aux_hook] wrote: {aux_p}")

if __name__ == "__main__":
    main()
