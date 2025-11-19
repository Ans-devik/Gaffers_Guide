from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / 'model3_output'
OUTPUT.mkdir(parents=True, exist_ok=True)

# cd /Users/nishoosingh/Downloads/model

# # overwrite the adaptive script with a fixed, robust version that detects attacking_focal_value etc.
# cat > model3_adaptive_thresholds.py <<'PY'
#!/usr/bin/env python3
# """
# model3_adaptive_thresholds.py (fixed)
# Robust re-classifier that supports attacking_focal_value / attacking_presence keys.
# Writes model3_results_adaptive.json and .csv
# """
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime



def build_adaptive_thresholds(features_df, policy=None, n_matches=None):
    if policy is None:
        policy = {
            'attacking_score': {'method':'quantile', 'q':0.66, 'tune_min_frac': None},
            'midfield_presence': {'method':'range_quantile', 'q_low':0.33, 'q_high':0.67},
            'defensive_presence': {'method':'quantile', 'q':0.33, 'tune_min_frac': None}
        }
    thr = {}
    if n_matches is None:
        n_matches = len(features_df)
    min_one_match_frac = 1.0 / max(1, n_matches)
    for feat, opts in policy.items():
        if feat not in features_df.columns:
            raise KeyError(f"Feature '{feat}' missing from features_df columns: {list(features_df.columns)}")
        if opts['method'] == 'quantile':
            q = float(opts.get('q', 0.66))
            base_thr = features_df[feat].quantile(q)
            thr_val = base_thr
            tune_min_frac = opts.get('tune_min_frac', None)
            if tune_min_frac is None:
                tune_min_frac = max(min_one_match_frac, 0.05)
            qs = np.linspace(0.5, 0.99, 40)
            found = False
            for qtest in qs:
                candidate = features_df[feat].quantile(qtest)
                frac = (features_df[feat] >= candidate).mean()
                if frac >= tune_min_frac:
                    thr_val = candidate
                    found = True
                    break
            if not found:
                thr_val = base_thr
            thr[feat] = float(thr_val)
        elif opts['method'] == 'range_quantile':
            ql = float(opts.get('q_low', 0.33)); qh = float(opts.get('q_high', 0.67))
            low = float(features_df[feat].quantile(ql)); high = float(features_df[feat].quantile(qh))
            thr[feat] = (low, high)
        else:
            raise ValueError(f"Unknown method {opts['method']} for feature {feat}")
    return thr

def label_match(row, thr):
    reasons = []
    a_val = float(row.get('attacking_score', np.nan))
    a_thr = thr['attacking_score']
    if np.isnan(a_val):
        attacking_label = "Unknown Attacking"; reasons.append("attacking_score missing")
    elif a_val >= a_thr:
        attacking_label = "Central Creativity (Strong)|Left Half-Space Creativity"
        reasons.append(f"attacking_score {a_val:.2f} >= {a_thr:.2f} (dominant)")
    else:
        attacking_label = "Less Creative Attacking"
        reasons.append(f"attacking_score {a_val:.2f} < {a_thr:.2f}")

    m_val = float(row.get('midfield_presence', np.nan))
    mid_low, mid_high = thr['midfield_presence']
    if np.isnan(m_val):
        midfield_label = "Unknown Midfield"; reasons.append("midfield_presence missing")
    else:
        if m_val < mid_low:
            midfield_label = "Low Midfield Presence"; reasons.append(f"midfield_presence {m_val:.2f} < {mid_low:.2f} -> low")
        elif m_val < mid_high:
            midfield_label = "Wing-Midfield Progression"; reasons.append(f"midfield_presence {m_val:.2f} between {mid_low:.2f} and {mid_high:.2f}")
        else:
            midfield_label = "Central Midfield Dominant"; reasons.append(f"midfield_presence {m_val:.2f} >= {mid_high:.2f}")

    d_val = float(row.get('defensive_presence', np.nan))
    d_thr = thr['defensive_presence']
    if np.isnan(d_val):
        defensive_label = "Unknown Defence"; reasons.append("defensive_presence missing")
    elif d_val < d_thr:
        defensive_label = "Solid Defence / Opponent Kept Out"; reasons.append(f"defensive_presence {d_val:.2f} < {d_thr:.2f}")
    else:
        defensive_label = "Open Defence / Vulnerable"; reasons.append(f"defensive_presence {d_val:.2f} >= {d_thr:.2f}")

    if "Left Half-Space" in attacking_label or "Central Creativity" in attacking_label:
        match_style = "Creative Half-Space Team"; reasons.append("attacking focal in half-space -> high-quality box entries")
    else:
        match_style = "Balanced/Other"
    return attacking_label, midfield_label, defensive_label, match_style, reasons

def build_features_df_from_results(results):
    """
    Robust extractor: detects multiple possible keys for attacking_score:
      - attacking_score, attack_score, attack
      - attacking_focal_value (preferred)
      - attacking_presence (if focal_value missing we may use presence scaled)
    Also normalizes midfield_presence / defensive_presence to percentages if values in 0..1.
    """
    rows = []
    for m in results:
        match_id = m.get('match_id') or m.get('id') or m.get('match') or m.get('matchId')
        # helper to search many plausible keys and nested places
        def find_val(keys):
            for k in keys:
                if k in m:
                    return m[k]
                if isinstance(m.get('features'), dict) and k in m['features']:
                    return m['features'][k]
                if isinstance(m.get('summary'), dict) and k in m['summary']:
                    return m['summary'][k]
                # also check nested under 'features' with alternatives
                if isinstance(m.get('features'), dict):
                    alt = m['features'].get(k + "_pct") or m['features'].get(k + "_percent")
                    if alt is not None:
                        return alt
            return None

        # prefer focal value if present
        a = find_val(['attacking_focal_value','attacking_score','attack_score','attack','attacking_presence'])
        mval = find_val(['midfield_presence','midfield','midfield_presence_pct','midfield_pct'])
        dval = find_val(['defensive_presence','defence_presence','defensive_presence_pct','defence_pct','defensive_pct'])

        def parse_num(v):
            if v is None:
                return np.nan
            if isinstance(v, str):
                s = v.strip()
                if s.endswith('%'):
                    try:
                        return float(s[:-1])
                    except:
                        pass
                try:
                    return float(s)
                except:
                    return np.nan
            if isinstance(v, (int, float)):
                return float(v)
            return np.nan

        a_val = parse_num(a); m_val = parse_num(mval); d_val = parse_num(dval)
        rows.append({'match_id': match_id, 'attacking_score_raw': a_val, 'midfield_presence_raw': m_val, 'defensive_presence_raw': d_val, 'raw_obj': m})

    df = pd.DataFrame(rows)

    # If attacking_score_raw seems to be a focal value between 0..1, convert to a 0..1 or 0..100 scale consistently.
    # We'll keep attacking_score as the focal value (0..1) if max <=1.1, otherwise use as-is.
    amax = df['attacking_score_raw'].max(skipna=True)
    if not pd.isna(amax) and amax <= 1.1:
        # keep attacking_score as-is (0..1); many thresholds were designed for values like 0.65
        df['attacking_score'] = df['attacking_score_raw']
    else:
        # if values look like big numbers, keep them as-is
        df['attacking_score'] = df['attacking_score_raw']

    # convert midfield/defence to percentages if they are fractions
    for col in ['midfield_presence_raw', 'defensive_presence_raw']:
        colmax = df[col].max(skipna=True)
        if pd.isna(colmax):
            df[col.replace('_raw','')] = df[col]
        else:
            if colmax <= 1.1:
                df[col.replace('_raw','')] = df[col] * 100.0
            else:
                df[col.replace('_raw','')] = df[col]

    out = df[['match_id','attacking_score','midfield_presence','defensive_presence','raw_obj']].copy()
    return out

def main(results_path=None, min_frac=None, out_prefix="model3_results_adaptive"):
    if results_path is None:
        results_path = Path.cwd() / "model3_results.json"
    else:
        results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"results file not found: {results_path}")
    with open(results_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and 'matches' in data:
        results_list = data['matches']
    elif isinstance(data, list):
        results_list = data
    else:
        if isinstance(data, dict):
            cand = [v for v in data.values() if isinstance(v, dict) and ('match_id' in v or 'features' in v or 'attacking_focal_value' in v)]
            if cand:
                results_list = cand
            else:
                raise ValueError("Unrecognized model3_results.json structure; expected list or {'matches': [...]} or dict with match-like values.")
        else:
            raise ValueError("Unrecognized model3_results.json structure; expected list or dict.")
    features_df = build_features_df_from_results(results_list)
    n_matches = len(features_df); print(f"Loaded {n_matches} matches from {results_path}")
    policy = {
        'attacking_score': {'method':'quantile','q':0.66,'tune_min_frac': (min_frac if min_frac is not None else None)},
        'midfield_presence': {'method':'range_quantile','q_low':0.33,'q_high':0.67},
        'defensive_presence': {'method':'quantile','q':0.33,'tune_min_frac': (min_frac if min_frac is not None else None)}
    }
    thr = build_adaptive_thresholds(features_df[['attacking_score','midfield_presence','defensive_presence']], policy=policy, n_matches=n_matches)
    print("Computed adaptive thresholds:")
    for k,v in thr.items(): print(" ", k, ":", v)
    reclassified=[]; counts = Counter()
    for idx, row in features_df.iterrows():
        att_lab, mid_lab, def_lab, style_lab, reasons = label_match(row, thr)
        counts['attacking:' + att_lab] += 1; counts['midfield:' + mid_lab] += 1; counts['defensive:' + def_lab] += 1; counts['style:' + style_lab] += 1
        entry = dict(row['raw_obj']) if isinstance(row['raw_obj'], dict) else {}
        entry.update({
            'match_id': row['match_id'],
            'attacking_label_adaptive': att_lab,
            'midfield_label_adaptive': mid_lab,
            'defensive_label_adaptive': def_lab,
            'match_style_adaptive': style_lab,
            'classification_reasons': reasons
        })
        reclassified.append(entry)
    print("\nReclassification counts (summary):")
    for k,v in counts.items(): print(f"  {k:40s} : {v}")
    out_json = Path(f"{out_prefix}.json"); out_csv = Path(f"{out_prefix}.csv")
    with open(out_json, "w", encoding="utf-8") as fh: json.dump(reclassified, fh, indent=2, ensure_ascii=False)
    print("Saved reclassified JSON to", out_json)
    rows = []
    for r in reclassified:
        rows.append({
            'match_id': r.get('match_id'),
            'attacking_label_adaptive': r.get('attacking_label_adaptive'),
            'midfield_label_adaptive': r.get('midfield_label_adaptive'),
            'defensive_label_adaptive': r.get('defensive_label_adaptive'),
            'match_style_adaptive': r.get('match_style_adaptive')
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved CSV summary to", out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reclassify model3 results using adaptive thresholds")
    p.add_argument("--results", help="Path to model3_results.json (default: ./model3_results.json)")
    p.add_argument("--min_frac", type=float, default=None, help="Minimum fraction for tuned thresholds (e.g. 0.10). If not set, defaults to max(1/n_matches,0.05).")
    p.add_argument("--out_prefix", default="model3_results_adaptive", help="Prefix for output JSON/CSV files")
    args = p.parse_args()
    main(results_path=args.results, min_frac=args.min_frac, out_prefix=args.out_prefix)
# PY

# # run the fixed adaptive re-classifier (adjust min_frac if you want, e.g. --min_frac 0.166)
# /Users/nishoosingh/Downloads/model/venv/bin/python /Users/nishoosingh/Downloads/model/model3_adaptive_thresholds.py




from pathlib import Path
from typing import Optional, Dict, Any
import traceback

def _call_entrypoint_for_model3(match_path: Optional[str] = None):
    """
    Safely call a real entrypoint:
    - If match_path is a directory: try to find a JSON results file inside it.
    - Prefer: run_on_matches -> run -> main
    - NEVER call model_main recursively.
    """
    from pathlib import Path

    resolved_path = match_path  # always defined

    try:
        p = Path(match_path) if match_path is not None else None

        # If it's a directory, search for a results file
        if p and p.exists() and p.is_dir():
            print(f"[model3] Inspecting directory {p} for JSON result files...")

            # search patterns
            patterns = [
                "model3_results.json",
                "model3_results.jsonl",
                "*_results.json",
                "*_results.jsonl",
                "*results.json",
                "*results.jsonl",
                "*_match.json",
                "*match.json",
                "*.json",
                "*.jsonl",
            ]

            selected = None
            for pat in patterns:
                for file in sorted(p.glob(pat)):
                    if file.is_file():
                        selected = str(file.resolve())
                        print(f"[model3] Selected results file: {selected}")
                        resolved_path = selected
                        break
                if selected:
                    break

            # If still nothing found — fallback: search recursive
            if not selected:
                deep_files = sorted([x for x in p.rglob("*.json") if x.is_file()])
                if deep_files:
                    selected = str(deep_files[0].resolve())
                    print(f"[model3] Fallback: picked {selected}")
                    resolved_path = selected
                else:
                    print(f"[model3] No JSON found inside {p}. Passing directory through.")

    except Exception as e:
        print(f"[model3] Exception while scanning directory {match_path}: {e}")
        # resolved_path stays as original match_path

    # ---- Preferred entrypoints ----
    # 1. run_on_matches / run_matches / run_multi
    fn = globals().get("run_on_matches") or globals().get("run_matches") or globals().get("run_multi")
    if callable(fn):
        try:
            if resolved_path is None:
                return fn()
            # try list-first
            try:
                return fn([resolved_path])
            except TypeError:
                return fn(resolved_path)
        except Exception:
            raise

    # 2. run(...)
    fn = globals().get("run")
    if callable(fn):
        try:
            return fn(resolved_path) if resolved_path else fn()
        except TypeError:
            return fn()

    # 3. main(...)
    fn = globals().get("main")
    if callable(fn):
        try:
            return fn(resolved_path) if resolved_path else fn()
        except TypeError:
            return fn()

    # Nothing found
    print("[model3] No valid entrypoint found.")
    return {}
def model_main(match_path: Optional[str]) -> Dict[str, Any]:
    """
    Public wrapper expected by app.py. Calls the safe entrypoint above, collects outputs
    into static/plots/<run_id>/ and returns {"plots": [...], "output_folder": "...", "summary_line": ...}.
    """
    from datetime import datetime
    from pathlib import Path as _P
    import json
    import pandas as _pd

    repo_root = Path(__file__).resolve().parents[1]

    # base static plots folder
    static_base = repo_root / "static" / "plots"
    static_base.mkdir(parents=True, exist_ok=True)

    # prepare run id & dest dir
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest_dir = static_base / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _copy(fp: _P):
        try:
            tgt = dest_dir / fp.name
            tgt.write_bytes(fp.read_bytes())
            return f"plots/{run_id}/{fp.name}"
        except Exception as e:
            print(f"[model3] copy failed {fp}: {e}")
            return None

    def _extract_summary_from_csv(fp: _P):
        try:
            df = _pd.read_csv(fp)
            row = df.iloc[0]
            return f"{row.get('match_id')},{row.get('attacking_label_adaptive')},{row.get('midfield_label_adaptive')},{row.get('defensive_label_adaptive')},{row.get('match_style_adaptive')}"
        except Exception:
            return None

    def _extract_summary_from_json(fp: _P):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "matches" in data:
                rec = data["matches"][0] if data["matches"] else None
            elif isinstance(data, list) and data:
                rec = data[0]
            elif isinstance(data, dict):
                # try pick first dict-like value
                vals = [v for v in data.values() if isinstance(v, dict)]
                rec = vals[0] if vals else None
            else:
                rec = None
            if not rec:
                return None
            return f"{rec.get('match_id')},{rec.get('attacking_label_adaptive')},{rec.get('midfield_label_adaptive')},{rec.get('defensive_label_adaptive')},{rec.get('match_style_adaptive')}"
        except Exception:
            return None

    summary_line = None

    # ========== FAST PATH: check match_path, repo_root, /mnt/data ==========
    try:
        mp = _P(match_path) if match_path else None
        # if file passed directly
        if mp and mp.exists() and mp.is_file() and mp.suffix.lower() in (".json", ".csv"):
            rel = _copy(mp)
            if rel:
                if mp.suffix.lower() == ".csv":
                    summary_line = _extract_summary_from_csv(mp)
                else:
                    summary_line = _extract_summary_from_json(mp)
                return {"plots": [rel], "output_folder": str(mp.parent.resolve()), "summary_line": summary_line}
        # if directory passed
        if mp and mp.exists() and mp.is_dir():
            for name in ("model3_results_adaptive.csv", "model3_results_adaptive.json",
                         "model3_results.csv", "model3_results.json"):
                cand = mp / name
                if cand.exists() and cand.is_file():
                    rel = _copy(cand)
                    if rel:
                        if cand.suffix.lower() == ".csv":
                            summary_line = _extract_summary_from_csv(cand)
                        else:
                            summary_line = _extract_summary_from_json(cand)
                        return {"plots": [rel], "output_folder": str(mp.resolve()), "summary_line": summary_line}
            # fallback: any single json/csv or prefer adaptive
            files = sorted([x for x in mp.glob("*.json")] + [x for x in mp.glob("*.csv")])
            if files:
                chosen = next((x for x in files if "adaptive" in x.name.lower() or "model3" in x.name.lower() or "results" in x.name.lower()), files[0])
                rel = _copy(_P(chosen))
                if rel:
                    if chosen.suffix.lower() == ".csv":
                        summary_line = _extract_summary_from_csv(_P(chosen))
                    else:
                        summary_line = _extract_summary_from_json(_P(chosen))
                    return {"plots": [rel], "output_folder": str(mp.resolve()), "summary_line": summary_line}
        # check repo root and /mnt/data
        for d in (repo_root, _P("/mnt/data")):
            if d and d.exists() and d.is_dir():
                for name in ("model3_results_adaptive.csv", "model3_results_adaptive.json",
                             "model3_results.csv", "model3_results.json"):
                    cand = d / name
                    if cand.exists() and cand.is_file():
                        rel = _copy(cand)
                        if rel:
                            if cand.suffix.lower() == ".csv":
                                summary_line = _extract_summary_from_csv(cand)
                            else:
                                summary_line = _extract_summary_from_json(cand)
                            return {"plots": [rel], "output_folder": str(d.resolve()), "summary_line": summary_line}
                files = sorted([x for x in d.glob("*.json")] + [x for x in d.glob("*.csv")])
                if files:
                    chosen = next((x for x in files if "adaptive" in x.name.lower() or "model3" in x.name.lower() or "results" in x.name.lower()), files[0])
                    rel = _copy(_P(chosen))
                    if rel:
                        if chosen.suffix.lower() == ".csv":
                            summary_line = _extract_summary_from_csv(_P(chosen))
                        else:
                            summary_line = _extract_summary_from_json(_P(chosen))
                        return {"plots": [rel], "output_folder": str(d.resolve()), "summary_line": summary_line}
    except Exception as e:
        print(f"[model3] fast-path scan error: {e}")

    # ========== CALL ENTRYPOINT and copy outputs ==========
    try:
        ret = _call_entrypoint_for_model3(match_path)
    except Exception:
        traceback.print_exc()
        raise

    # build candidate output folders
    out_candidates = []
    if isinstance(ret, dict) and ret.get("output_folder"):
        out_candidates.append(Path(ret["output_folder"]))
    out_candidates.extend([
        repo_root / "model3_output_multi",
        repo_root / "model3_output",
        repo_root / "model3_output_team",
        repo_root / "model3_output_clean_fullpitch",
        repo_root / "model3_output_fullmatch",
        repo_root / "matches",
        Path(match_path) if match_path else None,
    ])

    found_files = []
    used_output = None
    exts = {".png", ".jpg", ".jpeg", ".svg", ".json", ".csv", ".pdf"}

    for cand in out_candidates:
        try:
            if not cand:
                continue
            candp = Path(cand)
            if not candp.exists():
                continue
            for p in sorted(candp.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() in exts:
                    try:
                        target = dest_dir / p.name
                        target.write_bytes(p.read_bytes())
                        rel = f"plots/{run_id}/{p.name}"
                        if rel not in found_files:
                            found_files.append(rel)
                            used_output = str(candp.resolve())
                            # try to extract summary if it's the adaptive file
                            if p.name.lower().endswith(("model3_results_adaptive.csv", "model3_results.csv")) and not summary_line:
                                summary_line = _extract_summary_from_csv(p)
                            if p.name.lower().endswith(("model3_results_adaptive.json", "model3_results.json")) and not summary_line:
                                summary_line = _extract_summary_from_json(p)
                    except Exception:
                        continue
        except Exception:
            continue

    # include any files explicitly returned by entrypoint
    if isinstance(ret, dict) and isinstance(ret.get("plots"), (list, tuple)):
        for p in ret["plots"]:
            try:
                src = Path(p)
                if not src.is_absolute():
                    candidate = (repo_root / p)
                    if candidate.exists():
                        src = candidate
                    elif used_output:
                        candidate2 = Path(used_output) / p
                        if candidate2.exists():
                            src = candidate2
                if src.exists() and src.is_file():
                    target = dest_dir / src.name
                    target.write_bytes(src.read_bytes())
                    rel = f"plots/{run_id}/{src.name}"
                    if rel not in found_files:
                        found_files.append(rel)
                        if src.name.lower().endswith(("model3_results_adaptive.csv", "model3_results.csv")) and not summary_line:
                            summary_line = _extract_summary_from_csv(src)
                        if src.name.lower().endswith(("model3_results_adaptive.json", "model3_results.json")) and not summary_line:
                            summary_line = _extract_summary_from_json(src)
            except Exception:
                if isinstance(p, str) and p not in found_files:
                    found_files.append(p)

    return {"plots": found_files, "output_folder": used_output or str(out_candidates[0]), "summary_line": summary_line}
