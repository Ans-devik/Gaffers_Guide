# models/model3.py
"""
Robust model3 helper used by the Flask app.

Behavior:
 - Finds the repository root automatically (looks for app.py or .git or falls back sensibly)
 - Searches many likely model3 output folders and recursively grabs images (png/jpg/svg/...).
 - Copies images and non-image files into static/plots/<run_id>/ and returns:
    - served_plots: list of image URLs ("/static/plots/...") for the gallery
    - other_files: other static file URLs (csv/json) for preview/download
    - tables, summary_line, etc (as before)
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import traceback
import pandas as pd
import shutil
import os
import fnmatch

# ------------------------------------------------------------------
# Repo root detection (works when placed in models/)
# ------------------------------------------------------------------
_THIS_F = Path(__file__).resolve()
def _find_repo_root(start: Path) -> Path:
    # Walk up until we find 'app.py' or '.git' or stop after 6 parents
    cur = start
    for i in range(6):
        if (cur / 'app.py').exists() or (cur / '.git').exists() or (cur / 'templates').exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # fallback: two levels up (works if file is in models/)
    return start.parents[1]

REPO_ROOT = _find_repo_root(_THIS_F.parent)
STATIC_PLOTS_ROOT = REPO_ROOT / "static" / "plots"
STATIC_PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def _is_pref_file(p: Path) -> bool:
    n = p.name.lower()
    if "model3" in n or "adaptive" in n or "results" in n:
        if "dynamic_events" in n or ("events" in n and "dynamic" in n):
            return False
        return True
    return False

def _find_candidate_results_in_dir(d: Path) -> List[Path]:
    cand = []
    if not d or not d.exists() or not d.is_dir():
        return cand
    for ext in ("*.csv", "*.json", "*.jsonl"):
        cand.extend(sorted(d.glob(ext)))
    return [p for p in cand if p.is_file()]

def _choose_best_result_file(files: List[Path]) -> Optional[Path]:
    if not files:
        return None
    priority = [
        lambda n: "model3_results_adaptive.csv" in n,
        lambda n: "model3_results_adaptive.json" in n,
        lambda n: "model3_results.csv" in n,
        lambda n: "model3_results.json" in n,
        lambda n: "model3" in n and n.endswith(".csv"),
        lambda n: "adaptive" in n and n.endswith(".csv"),
        lambda n: "model3" in n and n.endswith(".json"),
        lambda n: "adaptive" in n and n.endswith(".json"),
        lambda n: "results" in n and (n.endswith(".csv") or n.endswith(".json")),
    ]
    names = [p.name.lower() for p in files]
    for test in priority:
        for i, n in enumerate(names):
            if test(n):
                return files[i]
    for p in files:
        if _is_pref_file(p):
            return p
    for p in files:
        if "dynamic_events" not in p.name.lower():
            return p
    return files[0]

def _extract_summary_from_csv(fp: Path, match_id_guess: Optional[str] = None) -> Optional[str]:
    try:
        df = pd.read_csv(fp, dtype=str)
        if df.empty:
            return None
        row = df.iloc[0]
        if match_id_guess is not None and 'match_id' in [c.lower() for c in df.columns]:
            try:
                matches = df[df.apply(lambda r: str(r.get('match_id','')).strip() == str(match_id_guess).strip(), axis=1)]
                if not matches.empty:
                    row = matches.iloc[0]
            except Exception:
                pass
        def _g(k):
            for cand in (k, k.lower(), k.upper()):
                if cand in row.index:
                    return row.get(cand) or ""
            return ""
        match_id = _g("match_id") or _g("id") or ""
        attacking = _g("attacking_label_adaptive") or _g("attacking_label") or ""
        midfield = _g("midfield_label_adaptive") or _g("midfield_label") or ""
        defensive = _g("defensive_label_adaptive") or _g("defensive_label") or ""
        style = _g("match_style_adaptive") or _g("match_style") or ""
        return f"{str(match_id)},{str(attacking)},{str(midfield)},{str(defensive)},{str(style)}"
    except Exception:
        return None

def _extract_summary_from_json(fp: Path, match_id_guess: Optional[str] = None) -> Optional[str]:
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
        rec = None
        if isinstance(raw, list) and raw:
            if match_id_guess is not None:
                for r in raw:
                    if isinstance(r, dict) and str(r.get("match_id","")) == str(match_id_guess):
                        rec = r
                        break
            if rec is None:
                rec = raw[0]
        elif isinstance(raw, dict):
            if "matches" in raw and isinstance(raw["matches"], list) and raw["matches"]:
                rec = None
                if match_id_guess is not None:
                    for r in raw["matches"]:
                        if isinstance(r, dict) and str(r.get("match_id","")) == str(match_id_guess):
                            rec = r
                            break
                if rec is None:
                    rec = raw["matches"][0]
            else:
                for v in raw.values():
                    if isinstance(v, dict) and ("match_id" in v or "attacking_label_adaptive" in v):
                        rec = v
                        break
                if rec is None and ("match_id" in raw or "attacking_label_adaptive" in raw):
                    rec = raw
        if not rec or not isinstance(rec, dict):
            return None
        match_id = rec.get("match_id") or rec.get("id") or ""
        attacking = rec.get("attacking_label_adaptive") or rec.get("attacking_label") or ""
        midfield = rec.get("midfield_label_adaptive") or rec.get("midfield_label") or ""
        defensive = rec.get("defensive_label_adaptive") or rec.get("defensive_label") or ""
        style = rec.get("match_style_adaptive") or rec.get("match_style") or ""
        return f"{str(match_id)},{str(attacking)},{str(midfield)},{str(defensive)},{str(style)}"
    except Exception:
         return None

# def _copy_file_into_dest(src: Path, dest_dir: Path) -> Optional[str]:
#     try:
#         dest_dir.mkdir(parents=True, exist_ok=True)
#         dst = dest_dir / src.name
#         shutil.copy2(src, dst)
#         return f"plots/{dest_dir.name}/{src.name}"
#     except Exception as e:
#         # log and continue
#         print(f"[model3] copy failed {src}: {e}")
#         return None
def _copy_file_into_dest(src: Path, dest_dir: Path) -> Optional[str]:
    """
    Copy a source file into dest_dir and return the relative path used by templates,
    e.g. 'plots/<run_id>/<filename>' (template expects either '/static/plots/...' or 'plots/...')
    If the source file is already the same path as the destination, avoid copying.
    """
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dst = dest_dir / src.name

        # Resolve absolute paths to detect identical files (avoid SameFileError)
        try:
            src_res = src.resolve()
            dst_res = dst.resolve()
        except Exception:
            # If resolve fails for some reason, fall back to string comparison
            src_res = src
            dst_res = dst

        if src_res == dst_res:
            # Already in dest; return relative path without copying.
            return f"plots/{dest_dir.name}/{src.name}"

        # Otherwise perform copy
        shutil.copy2(src_res, dst)
        return f"plots/{dest_dir.name}/{src.name}"
    except shutil.SameFileError:
        # Defensive: if shutil still raises SameFileError, just return path
        return f"plots/{dest_dir.name}/{src.name}"
    except Exception as e:
        # Lightweight log so console isn't flooded with long tracebacks for every file.
        print(f"[model3] copy failed {src}: {e}")
        return None

def _collect_plots_from_dir(src_dir: Path) -> List[Path]:
    """
    Collect image files from a directory (recursively).
    """
    out = []
    if not src_dir or not src_dir.exists() or not src_dir.is_dir():
        return out

    # prefer shallow matches first, but include recursive rglob too
    exts = ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf", "*.gif", "*.webp"]
    try:
        for ext in exts:
            out.extend(sorted(src_dir.glob(ext)))
        # also search one level deeper and recursively (safe, but avoid extremely deep)
        for ext in exts:
            out.extend(sorted(src_dir.rglob(ext)))
    except Exception:
        pass

    # dedupe and ensure files
    seen = []
    for p in out:
        if p and p.is_file() and p not in seen:
            seen.append(p)
    return seen

def _call_entrypoint_for_model3(match_path: Optional[str] = None) -> Dict[str, Any]:
    # If your repo exposes a programmatic entrypoint for model3 that returns an output_folder, call it here.
    # Keep stubbed to avoid accidental execution.
    return {}

def _load_table_summaries(dest_dir: Path) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    try:
        csvs = sorted(dest_dir.glob("*.csv"))
        for csv in csvs:
            try:
                df = pd.read_csv(csv, dtype=str)
                if df.empty:
                    continue
                cols = list(df.columns)
                rows = df.fillna("").to_dict(orient="records")
                tables.append({
                    "id": csv.stem,
                    "title": f"CSV — {csv.name}",
                    "columns": cols,
                    "rows": rows,
                    "source_file": str(csv.name)
                })
            except Exception:
                continue
    except Exception:
        pass

    try:
        jsons = sorted(dest_dir.glob("*.json"))
        for j in jsons:
            try:
                raw = json.loads(j.read_text(encoding="utf-8"))
                rows = []
                cols = []
                if isinstance(raw, dict):
                    maybe_list = []
                    all_keys = set()
                    for k, v in raw.items():
                        if isinstance(v, dict):
                            row = {"id": k}
                            row.update(v)
                            maybe_list.append(row)
                            all_keys.update(row.keys())
                        else:
                            maybe_list.append({"key": k, "value": v})
                            all_keys.update(["key", "value"])
                    rows = maybe_list
                    cols = list(all_keys)
                elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
                    rows = raw
                    cols = list({k for r in rows for k in r.keys()})
                else:
                    rows = [{"value": json.dumps(raw)}]
                    cols = ["value"]
                tables.append({
                    "id": j.stem,
                    "title": f"JSON — {j.name}",
                    "columns": cols,
                    "rows": rows,
                    "source_file": str(j.name)
                })
            except Exception:
                continue
    except Exception:
        pass

    return tables

# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------
def model_main(match_path: Optional[str]) -> Dict[str, Any]:
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest_dir = STATIC_PLOTS_ROOT / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    repo_root = REPO_ROOT
    summary_line = None
    found_files: List[str] = []
    used_output = None

    # prepare candidate directories to search (in order)
    cand_dirs = []

    mp = Path(match_path) if match_path else None
    if mp and mp.exists() and mp.is_file():
        mp = mp.parent
    if mp and mp.exists() and mp.is_dir():
        cand_dirs.append(mp)

    # core fallbacks
    cand_dirs.append(repo_root)
    cand_dirs.append(Path("/mnt/data"))

    # add a broader set of likely model3 output folders (so images are found)
    cand_dirs.extend([
        repo_root / "model3_output_multi",
        repo_root / "model3_output",
        repo_root / "model3_output_team",
        repo_root / "model3_output_clean_fullpitch",
        repo_root / "model3_output_fullmatch",
        repo_root / "model3_images",
        repo_root / "model3_visuals",
        repo_root / "model3_plots",
        repo_root / "model3_png",
        repo_root / "matches",
    ])

    try:
        ret = _call_entrypoint_for_model3(match_path)
    except Exception:
        ret = {}
    if isinstance(ret, dict) and ret.get("output_folder"):
        try:
            cand_dirs.insert(0, Path(ret["output_folder"]))
        except Exception:
            pass

    chosen_result_fp: Optional[Path] = None
    chosen_dir_for_plots: Optional[Path] = None

    for d in cand_dirs:
        try:
            if not d:
                continue
            dpath = Path(d)
            if not dpath.exists() or not dpath.is_dir():
                continue
            files = _find_candidate_results_in_dir(dpath)
            if not files:
                continue
            best = _choose_best_result_file(files)
            if best:
                chosen_result_fp = best
                chosen_dir_for_plots = dpath
                used_output = str(dpath.resolve())
                break
        except Exception:
            continue

    if chosen_result_fp is None:
        for d in (repo_root, Path("/mnt/data")):
            try:
                if not d.exists():
                    continue
                files = _find_candidate_results_in_dir(d)
                if not files:
                    continue
                best = _choose_best_result_file(files)
                if best:
                    chosen_result_fp = best
                    chosen_dir_for_plots = d
                    used_output = str(d.resolve())
                    break
            except Exception:
                continue

    match_id_guess = None
    if mp:
        match_id_guess = mp.name

    if chosen_result_fp:
        try:
            if chosen_result_fp.suffix.lower() in (".csv",):
                summary_line = _extract_summary_from_csv(chosen_result_fp, match_id_guess=match_id_guess)
            elif chosen_result_fp.suffix.lower() in (".json", ".jsonl"):
                summary_line = _extract_summary_from_json(chosen_result_fp, match_id_guess=match_id_guess)
            rel = _copy_file_into_dest(chosen_result_fp, dest_dir)
            if rel:
                found_files.append(rel)
        except Exception:
            print("[model3] failed to extract summary from", chosen_result_fp)
            traceback.print_exc()

    copied = set()

    def _copy_images_from_folder(folder: Path):
        if not folder or not folder.exists():
            return
        imgs = _collect_plots_from_dir(folder)
        for img in imgs:
            try:
                rel = _copy_file_into_dest(img, dest_dir)
                if rel and rel not in copied:
                    found_files.append(rel)
                    copied.add(rel)
            except Exception:
                continue

    # copy images from chosen dir (if any)
    if chosen_dir_for_plots:
        _copy_images_from_folder(Path(chosen_dir_for_plots))

    # also search conventional locations for images
    for p in (
        repo_root / "model3_output_team",
        repo_root / "model3_output",
        repo_root / "model3_images",
        repo_root / "model3_visuals",
        repo_root / "model3_plots",
        repo_root / "model3_png",
        repo_root / "model2_output_team",
    ):
        try:
            _copy_images_from_folder(p)
        except Exception:
            pass

    # Also pick up any images already in repo static/plots (helpful when files are pre-generated)
    try:
        _copy_images_from_folder(repo_root / "static" / "plots")
    except Exception:
        pass

    # dedupe & sort
    served_rel_paths = sorted(list(dict.fromkeys(found_files)))

    # ensure returned values are URLs templates can use.
    normalized_all = []
    for p in served_rel_paths:
        if isinstance(p, str) and (p.startswith("/static/") or p.startswith("http")):
            normalized_all.append(p)
        else:
            normalized_all.append("/static/" + p.lstrip("/"))

    # fallback: if CSV exists in dest_dir, try to extract summary from it and include it in listing
    if not summary_line:
        try:
            csvs = sorted(dest_dir.glob("*.csv"))
            chosen = None
            if csvs:
                for c in csvs:
                    if "adaptive" in c.name.lower() or "model3" in c.name.lower() or "results" in c.name.lower():
                        chosen = c
                        break
                if chosen is None:
                    chosen = csvs[0]
                summary_line = _extract_summary_from_csv(chosen, match_id_guess=match_id_guess)
                rel = f"plots/{dest_dir.name}/{chosen.name}"
                if "/static/" + rel not in normalized_all:
                    normalized_all.append("/static/" + rel)
                    served_rel_paths.append(rel)
        except Exception:
            pass

    # Load parsed table summaries (if any) from dest_dir
    try:
        tables = _load_table_summaries(dest_dir)
    except Exception:
        tables = []

    # filter images-only for the gallery
    image_exts = ('.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp')
    images_only = [u for u in normalized_all if any(u.lower().endswith(ext) for ext in image_exts)]
    other_files = [u for u in normalized_all if u not in images_only]

    out = {
        "served_plots": images_only,
        "other_files": other_files,
        "plots": images_only,
        "served_output_dir": str(used_output) if used_output else str(repo_root),
        "served_output_dir_rel": f"plots/{run_id}",
        "output_folder": str(used_output) if used_output else str(repo_root),
        "summary_line": summary_line,
        "tables": tables
    }

    return out

# CLI for manual testing
if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    print("Running models.model3.model_main with match_path=", arg)
    o = model_main(arg)
    print(json.dumps(o, indent=2))
