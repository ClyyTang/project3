#!/usr/bin/env python3
from pathlib import Path
import json, argparse, datetime, shutil

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
    ap.add_argument("--root", default="/home/ubuntu/data1/lyy/full_rlds_project-3")
    args = ap.parse_args()

    root = Path(args.root)
    r = args.round
    cand_candidates = [
        root / "stage3" / "checkpoints" / f"round_{r}_candidates.json",
        root / "checkpoints" / "stage3_final" / f"round_{r}_candidates.json",
        root / "stage2" / "checkpoints" / f"round_{r}_candidates.json",
    ]
    cand_p = next((x for x in cand_candidates if x.exists()), cand_candidates[0])
    diag_p = cand_p.with_name(f"round_{r}_diagnosis_records.json")
    aux_p  = root / "data" / f"auxiliary_labels_round{r}.json"

    if not cand_p.exists():
        print(f"[stage3_aux_hook] skip, no candidates: {cand_p}")
        return

    cand_obj = json.load(open(cand_p, "r", encoding="utf-8"))
    pairs, pkey, ptpl = unwrap(cand_obj)

    diag_map = {}
    if diag_p.exists():
        diag_obj = json.load(open(diag_p, "r", encoding="utf-8"))
        diags, _, _ = unwrap(diag_obj)
        for d in diags:
            if isinstance(d, dict):
                sid = d.get("sample_id", d.get("sample_idx"))
                if sid is not None:
                    diag_map[str(sid)] = d

    aux_map = {}
    if aux_p.exists():
        aux_map = json.load(open(aux_p, "r", encoding="utf-8"))

    emap = {"unknown":0, "perception":1, "comprehension":2, "reasoning":3, "decision":4}

    def fallback(rec, diag):
        et = str((diag or {}).get("error_type", "unknown")).lower()
        direction = emap.get(et, 0)
        score = f01((diag or {}).get("chosen_score", rec.get("chosen_score", 0.5)))
        gap = abs(float((diag or {}).get("score_gap", rec.get("score_gap", 0.0) or 0.0)))
        kw = [0.0] * 34
        kw[direction] = 1.0
        kw[10] = 1.0 if gap < 0.01 else 0.0
        kw[11] = 1.0 if et == "unknown" else 0.0
        return {
            "keywords": kw,
            "direction": int(direction),
            "cot_quality": score,
            "action_validity": f01(0.4 + 0.6 * min(gap, 1.0)),
        }

    def sanitize(aux):
        if not isinstance(aux, dict): aux = {}
        kw = aux.get("keywords", [0.0] * 34)
        if not isinstance(kw, list): kw = [0.0] * 34
        kw = [(float(x) if isinstance(x, (int, float)) else 0.0) for x in kw[:34]]
        if len(kw) < 34: kw += [0.0] * (34 - len(kw))
        try: direction = int(aux.get("direction", 0))
        except: direction = 0
        return {
            "keywords": kw,
            "direction": direction,
            "cot_quality": f01(aux.get("cot_quality", 0.5)),
            "action_validity": f01(aux.get("action_validity", 0.5)),
        }

    before = sum(1 for p in pairs if isinstance(p, dict) and p.get("aux_labels") is not None)

    for p in pairs:
        if not isinstance(p, dict): 
            continue
        sid = p.get("sample_id", p.get("sample_idx"))
        key = str(sid) if sid is not None else None
        diag = diag_map.get(key, {})
        aux = None
        if key is not None and key in aux_map:
            aux = aux_map[key]
        elif key is not None and sid in aux_map:
            aux = aux_map[sid]
        if aux is None:
            aux = fallback(p, diag)
        p["aux_labels"] = sanitize(aux)

    after = sum(1 for p in pairs if isinstance(p, dict) and p.get("aux_labels") is not None)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    arc = root / "stage3" / "checkpoints" / f"_archive_auxfix_round{r}_{ts}"
    arc.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cand_p, arc / cand_p.name)

    json.dump(wrap(pairs, pkey, ptpl), open(cand_p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[stage3_aux_hook] round{r}: {before}/{len(pairs)} -> {after}/{len(pairs)}")
    print(f"[stage3_aux_hook] backup: {arc}")

if __name__ == "__main__":
    main()
