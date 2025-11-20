# import json
# import pandas as pd
# from pathlib import Path
# from typing import Dict, Any


# BASE = Path("/Users/nishoosingh/Downloads/model/opendata/data/matches/1925299")

# TRACKING_FILE = BASE / "1925299_tracking_extrapolated.jsonl"
# MATCH_FILE = BASE / "1925299_match.json"
# EVENTS_FILE = BASE / "1925299_dynamic_events.csv"
# PHASES_FILE = BASE / "1925299_phases_of_play.csv"


# def load_tracking():
#     print("Loading tracking data JSONL...")

#     frames = []
#     with open(TRACKING_FILE, "r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             frames.append(json.loads(line))

#     print(f"Loaded {len(frames):,} tracking frames.")
#     return frames


# def load_match_metadata():
#     print("Loading match metadata...")
#     with open(MATCH_FILE, "r") as f:
#         meta = json.load(f)
#     print("Match metadata loaded.")
#     return meta


# def load_events():
#     print("Loading dynamic events CSV...")
#     df_events = pd.read_csv(EVENTS_FILE)
#     print(f"Loaded {len(df_events):,} events.")
#     return df_events


# def load_phases():
#     print("Loading phases of play CSV...")
#     df_phases = pd.read_csv(PHASES_FILE)
#     print(f"Loaded {len(df_phases):,} phases.")
#     return df_phases


# if __name__ == "__main__":
#     tracking = load_tracking()
#     meta = load_match_metadata()
#     events = load_events()
#     phases = load_phases()

#     print("\nEverything loaded successfully!")
# def extract_features_from_raw(tracking, events, phases, metadata) -> Dict[str, Any]:
#     """
#     Robust extractor that understands your events CSV layout (x_start/y_start, x_end/y_end).
#     Returns the required feature dict per match.
#     """
#     import numpy as np
#     import pandas as pd

#     def _get_xy_from_events(events_df):
#         if events_df is None:
#             return None, None
#         # Prefer start coordinates, then end coordinates, then fallbacks
#         primary_pairs = [("x_start", "y_start"), ("x_end", "y_end")]
#         fallback_x = ("x", "X", "pos_x", "PosX", "event_x", "ball_x", "ballX")
#         fallback_y = ("y", "Y", "pos_y", "PosY", "event_y", "ball_y", "ballY")

#         # 1) primary pairs
#         for xc, yc in primary_pairs:
#             if xc in events_df.columns and yc in events_df.columns:
#                 xs = pd.to_numeric(events_df[xc], errors="coerce").to_numpy()
#                 ys = pd.to_numeric(events_df[yc], errors="coerce").to_numpy()
#                 mask = (~np.isnan(xs)) & (~np.isnan(ys))
#                 xs, ys = xs[mask], ys[mask]
#                 if len(xs) > 0:
#                     return xs.astype(float), ys.astype(float)

#         # 2) targeted/pass reception coords sometimes named like player_targeted_x_pass / _y_pass
#         for sx in ("player_targeted_x_pass", "player_targeted_x_reception", "player_targeted_x_reception", "player_targeted_xpass"):
#             sy = sx.replace("x", "y")
#             if sx in events_df.columns and sy in events_df.columns:
#                 xs = pd.to_numeric(events_df[sx], errors="coerce").to_numpy()
#                 ys = pd.to_numeric(events_df[sy], errors="coerce").to_numpy()
#                 mask = (~np.isnan(xs)) & (~np.isnan(ys))
#                 xs, ys = xs[mask], ys[mask]
#                 if len(xs) > 0:
#                     return xs.astype(float), ys.astype(float)

#         # 3) fallback candidates
#         for xc in fallback_x:
#             for yc in fallback_y:
#                 if xc in events_df.columns and yc in events_df.columns:
#                     xs = pd.to_numeric(events_df[xc], errors="coerce").to_numpy()
#                     ys = pd.to_numeric(events_df[yc], errors="coerce").to_numpy()
#                     mask = (~np.isnan(xs)) & (~np.isnan(ys))
#                     xs, ys = xs[mask], ys[mask]
#                     if len(xs) > 0:
#                         return xs.astype(float), ys.astype(float)
#         return None, None

#     xs, ys = None, None
#     try:
#         xs, ys = _get_xy_from_events(events)
#     except Exception:
#         xs, ys = None, None

#     # fallback: try ball positions from tracking frames
#     if (xs is None or ys is None or len(xs) == 0) and isinstance(tracking, list) and len(tracking) > 0:
#         bx, by = [], []
#         for fr in tracking:
#             if not isinstance(fr, dict):
#                 continue
#             if "ball" in fr and isinstance(fr["ball"], dict):
#                 b = fr["ball"]
#                 if "x" in b and "y" in b:
#                     try:
#                         bx.append(float(b["x"])); by.append(float(b["y"]))
#                     except Exception:
#                         pass
#             if "ball_x" in fr and "ball_y" in fr:
#                 try:
#                     bx.append(float(fr["ball_x"])); by.append(float(fr["ball_y"]))
#                 except Exception:
#                     pass
#         if bx and by:
#             xs = np.array(bx, dtype=float)
#             ys = np.array(by, dtype=float)

#     # if still nothing, return safe defaults
#     if xs is None or ys is None or len(xs) == 0:
#         return {
#             "attacking_focal": 4,
#             "attacking_focal_value": 0.0,
#             "attacking_presence": 0.0,
#             "midfield_focal": 4,
#             "midfield_presence": 0.0,
#             "defensive_focal": 4,
#             "defensive_presence": 0.0,
#             "match_id": metadata.get("match_id", metadata.get("id", "unknown"))
#         }

#     # Normalize to 0..1 (detect scale)
#     max_x = float(np.nanmax(xs)) if len(xs) > 0 else 1.0
#     max_y = float(np.nanmax(ys)) if len(ys) > 0 else 1.0
#     if max_x > 1.5:
#         xs = xs / max_x
#     if max_y > 1.5:
#         ys = ys / max_y
#     xs = np.clip(xs, 0.0, 1.0)
#     ys = np.clip(ys, 0.0, 1.0)

#     # Build 3x3 global grid
#     grid = np.zeros((3, 3), dtype=float)
#     ix = np.minimum((xs * 3).astype(int), 2)
#     iy = np.minimum((ys * 3).astype(int), 2)
#     for a, b in zip(ix, iy):
#         grid[a, b] += 1.0

#     # For each third, put the counts in the middle row (same indexing scheme as model3 expects)
#     def grid_for_third(full_grid, third_row_index):
#         mat = np.zeros((3, 3), dtype=float)
#         mat[1, :] = full_grid[third_row_index, :]
#         return mat

#     attacking_grid = grid_for_third(grid, 0)
#     midfield_grid = grid_for_third(grid, 1)
#     defensive_grid = grid_for_third(grid, 2)

#     def focal_and_presence(mat):
#         flat = mat.reshape(-1)
#         total = float(flat.sum())
#         if total == 0.0:
#             return 4, 0.0, 0.0
#         idx = int(flat.argmax())
#         focal_value = float(flat[idx] / total)
#         presence_pct = float(min(100.0, (total / len(xs)) * 100.0))
#         return idx, focal_value, presence_pct

#     af_idx, af_val, ap = focal_and_presence(attacking_grid)
#     mf_idx, mf_val, mp = focal_and_presence(midfield_grid)
#     df_idx, df_val, dp = focal_and_presence(defensive_grid)

#     return {
#         "attacking_focal": int(af_idx),
#         "attacking_focal_value": float(af_val),
#         "attacking_presence": float(ap),
#         "midfield_focal": int(mf_idx),
#         "midfield_presence": float(mp),
#         "defensive_focal": int(df_idx),
#         "defensive_presence": float(dp),
#         "match_id": metadata.get("match_id", metadata.get("id", "unknown"))
#     }
# # ---------------------
# # Batch helpers & discovery
# # ---------------------
# from typing import List

# # point at the folder containing many match subfolders (adjust if needed)
# MATCH_BASE = Path("/Users/nishoosingh/Downloads/model/opendata/data/matches")

# def _load_tracking_file(path: Path):
#     frames = []
#     with path.open("r") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 frames.append(json.loads(line))
#             except Exception:
#                 # skip malformed lines
#                 continue
#     return frames

# def _load_json_file(path: Path):
#     with path.open("r") as f:
#         return json.load(f)

# def _load_csv_file(path: Path):
#     try:
#         return pd.read_csv(path, low_memory=False)
#     except Exception:
#         return pd.read_csv(path, dtype=object)

# def get_all_matches() -> List[Dict[str, Any]]:
#     """
#     Discover all match subfolders under MATCH_BASE, load each match's files,
#     run extract_features_from_raw(...) and return a list of feature dicts.
#     Skips match folders missing required files.
#     """
#     matches: List[Dict[str, Any]] = []

#     if not MATCH_BASE.exists():
#         print(f"[model0] MATCH_BASE not found: {MATCH_BASE}")
#         return matches

#     match_dirs = sorted([p for p in MATCH_BASE.iterdir() if p.is_dir()])
#     print(f"[model0] Found {len(match_dirs)} match folders under {MATCH_BASE}")

#     for mdir in match_dirs:
#         match_id = mdir.name
#         print(f"[model0] Processing match folder: {match_id}")

#         # expected file patterns in each folder
#         try:
#             tracking_file = next(mdir.glob("*tracking_extrapolated.jsonl"))
#             meta_file = next(mdir.glob("*match.json"))
#             events_file = next(mdir.glob("*dynamic_events.csv"))
#             phases_file = next(mdir.glob("*phases_of_play.csv"))
#         except StopIteration:
#             print(f"[model0] Skipped {match_id}: missing one or more expected files")
#             continue

#         # load by path (do NOT call the single-match load_* functions that use BASE)
#         try:
#             tracking = _load_tracking_file(tracking_file)
#             metadata = _load_json_file(meta_file)
#             events = _load_csv_file(events_file)
#             phases = _load_csv_file(phases_file)
#         except Exception as e:
#             print(f"[model0] Error loading files for {match_id}: {e}")
#             continue

#         try:
#             feats = extract_features_from_raw(tracking, events, phases, metadata)
#             feats["match_id"] = match_id
#             matches.append(feats)
#         except Exception as e:
#             print(f"[model0] Error extracting features for {match_id}: {e}")
#             continue

#     # expose convenience globals
#     global FEATURES_LIST, FEATURES
#     FEATURES_LIST = matches
#     FEATURES = matches

#     print(f"[model0] Extracted features for {len(matches)} matches.")
#     return matches

# model0_load_data.py
"""
Unified loader for local match data.

- Discovers match folders under MATCH_BASE
- For each match folder loads:
    * <match>_tracking_extrapolated.jsonl
    * <match>_match.json
    * <match>_dynamic_events.csv
    * <match>_phases_of_play.csv
- Extracts simple tactical features via extract_features_from_raw(...)
- Exposes get_all_matches() and FEATURES_LIST
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np


# === Configure this to point at the folder that contains match subfolders ===
MATCH_BASE = Path("/Users/nishoosingh/Downloads/model/opendata/data/matches")

# ---------------------
# File loaders (by path)
# ---------------------
def _load_tracking_file(path: Path) -> List[Dict[str, Any]]:
    frames = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except Exception:
                # if a line fails to parse, skip it
                continue
    return frames

def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)

def _load_csv_file(path: Path):
    # use low_memory=False to reduce DtypeWarning for large CSVs
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        # last-resort read
        return pd.read_csv(path, dtype=object)

# ---------------------
# Feature extraction
# ---------------------
def extract_features_from_raw(tracking, events, phases, metadata) -> Dict[str, Any]:
    """
    Event-based extractor — builds simple 3x3 heatmaps per third using event (x,y) positions.
    Returns dict with keys expected by model3.classify_all.

    If events DataFrame contains x,y-like columns this uses them.
    Otherwise tries to extract ball x/y from tracking frames as a fallback.
    """
    # helper: robustly find x,y columns in events DF
    def _get_xy_from_events(events_df):
        if events_df is None:
            return None, None
        # candidate column name pairs (common variants)
        x_candidates = ("x_start","x_end","x","X","pos_x","PosX","ball_x","ballX","event_x")
        y_candidates = ("y_start","y_end","y","Y","pos_y","PosY","ball_y","ballY","event_y")
        # prefer start / end pairs first
        for xc, yc in (("x_start","y_start"), ("x_end","y_end")):
            if xc in events_df.columns and yc in events_df.columns:
                xs = pd.to_numeric(events_df[xc], errors="coerce").to_numpy()
                ys = pd.to_numeric(events_df[yc], errors="coerce").to_numpy()
                mask = (~np.isnan(xs)) & (~np.isnan(ys))
                xs, ys = xs[mask], ys[mask]
                if len(xs) > 0:
                    return xs.astype(float), ys.astype(float)
        # fallback search
        for xc in x_candidates:
            for yc in y_candidates:
                if xc in events_df.columns and yc in events_df.columns:
                    xs = pd.to_numeric(events_df[xc], errors="coerce").to_numpy()
                    ys = pd.to_numeric(events_df[yc], errors="coerce").to_numpy()
                    mask = (~np.isnan(xs)) & (~np.isnan(ys))
                    xs, ys = xs[mask], ys[mask]
                    if len(xs) > 0:
                        return xs.astype(float), ys.astype(float)
        return None, None

    xs, ys = None, None
    try:
        xs, ys = _get_xy_from_events(events)
    except Exception:
        xs, ys = None, None

    # Fallback: extract ball positions from tracking frames (best-effort)
    if (xs is None or ys is None or len(xs) == 0) and isinstance(tracking, list) and len(tracking) > 0:
        bx, by = [], []
        for fr in tracking:
            if not isinstance(fr, dict):
                continue
            if "ball" in fr and isinstance(fr["ball"], dict):
                b = fr["ball"]
                if "x" in b and "y" in b:
                    try:
                        bx.append(float(b["x"])); by.append(float(b["y"]))
                    except Exception:
                        pass
            if "ball_x" in fr and "ball_y" in fr:
                try:
                    bx.append(float(fr["ball_x"])); by.append(float(fr["ball_y"]))
                except Exception:
                    pass
        if bx and by:
            xs = np.array(bx, dtype=float)
            ys = np.array(by, dtype=float)

    # If still no coordinates, return a safe default feature dict
    if xs is None or ys is None or len(xs) == 0:
        return {
            "attacking_focal": 4,
            "attacking_focal_value": 0.0,
            "attacking_presence": 0.0,
            "midfield_focal": 4,
            "midfield_presence": 0.0,
            "defensive_focal": 4,
            "defensive_presence": 0.0,
            "match_id": metadata.get("match_id", metadata.get("id", "unknown"))
        }

    # Normalize coordinates to 0..1 (detect scale automatically)
    max_x = float(np.nanmax(xs)) if len(xs) > 0 else 1.0
    max_y = float(np.nanmax(ys)) if len(ys) > 0 else 1.0
    if max_x > 1.5:
        xs = xs / max_x
    if max_y > 1.5:
        ys = ys / max_y
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)

    # Build a 3x3 count grid across the pitch (rows = pitch-long slices, cols = lateral slices)
    grid = np.zeros((3, 3), dtype=float)
    ix = np.minimum((xs * 3).astype(int), 2)
    iy = np.minimum((ys * 3).astype(int), 2)
    for a, b in zip(ix, iy):
        grid[a, b] += 1.0

    # Helper: produce a 3x3 matrix for each third (we place counts in the middle row)
    def grid_for_third(full_grid, third_row_index):
        mat = np.zeros((3, 3), dtype=float)
        mat[1, :] = full_grid[third_row_index, :]
        return mat

    attacking_grid = grid_for_third(grid, 0)
    midfield_grid = grid_for_third(grid, 1)
    defensive_grid = grid_for_third(grid, 2)

    # Compute focal index, focal_value, and presence % (events_in_third / total_events *100)
    def focal_and_presence(mat):
        flat = mat.reshape(-1)
        total = float(flat.sum())
        if total == 0.0:
            return 4, 0.0, 0.0
        idx = int(flat.argmax())
        focal_value = float(flat[idx] / total)
        presence_pct = float(min(100.0, (total / len(xs)) * 100.0))
        return idx, focal_value, presence_pct

    af_idx, af_val, ap = focal_and_presence(attacking_grid)
    mf_idx, mf_val, mp = focal_and_presence(midfield_grid)
    df_idx, df_val, dp = focal_and_presence(defensive_grid)

    return {
        "attacking_focal": int(af_idx),
        "attacking_focal_value": float(af_val),
        "attacking_presence": float(ap),
        "midfield_focal": int(mf_idx),
        "midfield_presence": float(mp),
        "defensive_focal": int(df_idx),
        "defensive_presence": float(dp),
        "match_id": metadata.get("match_id", metadata.get("id", "unknown"))
    }

# ---------------------
# Batch discovery: iterate match folders and run extractor
# ---------------------
def get_all_matches() -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []

    if not MATCH_BASE.exists():
        print(f"[model0] MATCH_BASE not found: {MATCH_BASE}")
        return matches

    match_dirs = sorted([p for p in MATCH_BASE.iterdir() if p.is_dir()])
    print(f"[model0] Found {len(match_dirs)} match folders under {MATCH_BASE}")

    for mdir in match_dirs:
        match_id = mdir.name
        print(f"[model0] Processing match: {match_id}")

        try:
            tracking_file = next(mdir.glob("*tracking_extrapolated.jsonl"))
            meta_file = next(mdir.glob("*match.json"))
            events_file = next(mdir.glob("*dynamic_events.csv"))
            phases_file = next(mdir.glob("*phases_of_play.csv"))
        except StopIteration:
            print(f"[model0] Skipped {match_id}: missing one or more expected files")
            continue

        # Load files for this match (using file paths)
        try:
            tracking = _load_tracking_file(tracking_file)
            metadata = _load_json_file(meta_file)
            events = _load_csv_file(events_file)
            phases = _load_csv_file(phases_file)
        except Exception as e:
            print(f"[model0] Error loading files for {match_id}: {e}")
            continue

        try:
            feats = extract_features_from_raw(tracking, events, phases, metadata)
            feats["match_id"] = match_id
            matches.append(feats)
        except Exception as e:
            print(f"[model0] Error extracting features for {match_id}: {e}")
            continue

    print(f"[model0] Extracted features for {len(matches)} matches.")
    # convenience aliases
    global FEATURES_LIST, FEATURES
    FEATURES_LIST = matches
    FEATURES = matches
    return matches

# Convenience alias: run immediately if module executed as script
if __name__ == "__main__":
    ms = get_all_matches()
    print("Done. Matches extracted:", len(ms))

