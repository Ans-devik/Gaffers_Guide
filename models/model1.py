# # model1_multi_matches.py
# # Extended model1 to handle multiple matches (default: 6)
# # - robust loader that accepts .jsonl, *_tracking*.json, structured_data.json etc.
# # - defensive-line logic selects defenders relative to opponent per frame

# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import pandas as pd
# import os
# import glob
# # ======= re-entrancy guard (add near top of file) =======
# _model1_running = False
# # =======================================================

# # Try importing your existing data loader if available
# try:
#     from model0_load_data import load_tracking, load_match_metadata
# except Exception:
#     load_tracking = None
#     load_match_metadata = None

# FPS = 25  # default frames-per-second used for minute conversion

# # -------------------------
# # Helpers (same logic as in model1)
# # -------------------------
# def frame_index_to_minute(frame_idx):
#     return frame_idx / (FPS * 60.0)

# def minute_to_bucket(minute):
#     if minute < 0:
#         return "unknown"
#     if minute < 30:
#         return "deep"
#     if minute < 60:
#         return "normal"
#     if minute < 90:
#         return "high"
#     return "aggressive"

# def map_players_to_teams(frames, meta):
#     home_team_id = meta.get("home_team", {}).get("id") if isinstance(meta, dict) else None
#     away_team_id = meta.get("away_team", {}).get("id") if isinstance(meta, dict) else None

#     all_pids = []
#     for f in frames:
#         for p in f.get("player_data", []):
#             pid = p.get("player_id") or p.get("id")
#             if pid is not None:
#                 all_pids.append(pid)
#     all_pids = list(dict.fromkeys(all_pids))

#     home_players = set()
#     away_players = set()
#     found_teamfield = False
#     for f in frames:
#         for p in f.get("player_data", []):
#             pid = p.get("player_id") or p.get("id")
#             if pid is None:
#                 continue
#             if p.get("team_id") is not None:
#                 found_teamfield = True
#                 if p.get("team_id") == home_team_id:
#                     home_players.add(pid)
#                 elif p.get("team_id") == away_team_id:
#                     away_players.add(pid)
#             elif p.get("team") is not None:
#                 found_teamfield = True
#                 if p.get("team") == meta.get("home_team", {}).get("name") or p.get("team") == home_team_id:
#                     home_players.add(pid)
#                 else:
#                     away_players.add(pid)

#     if not found_teamfield or (not home_players and not away_players):
#         # fallback: split list of unique player ids in half
#         mid = len(all_pids) // 2
#         home_players = set(all_pids[:mid])
#         away_players = set(all_pids[mid:])

#     return home_players, away_players, home_team_id, away_team_id

# def compute_defensive_line_height(frames, team_players=None, opponent_players=None, n_defenders=4):
#     """
#     For each frame:
#       - collect team player x,y positions
#       - collect opponent player x positions to determine relative orientation
#       - if team mean x < opponent mean x -> team is to the left -> deepest players are smallest x
#         else deepest players are largest x
#       - pick n_defenders deepest players and return mean Y of those players for that frame
#     Returns list of mean Y (one per frame)
#     """
#     heights = []
#     for f in frames:
#         # gather team and opponent positions
#         team_pos = []
#         opp_pos = []
#         for p in f.get("player_data", []):
#             pid = p.get("player_id") or p.get("id")
#             x = p.get("x")
#             y = p.get("y")
#             if x is None or y is None:
#                 continue
#             if team_players is not None and pid in team_players:
#                 team_pos.append({"x": x, "y": y, "id": pid})
#             elif opponent_players is not None and pid in opponent_players:
#                 opp_pos.append({"x": x, "y": y, "id": pid})
#             else:
#                 # if team/opponent sets not provided, collect for team_pos (fallback)
#                 if team_players is None and opponent_players is None:
#                     team_pos.append({"x": x, "y": y, "id": pid})

#         # If we don't have explicit opponent players but have team players, infer opponent as others
#         if opponent_players is None and team_players is not None:
#             for p in f.get("player_data", []):
#                 pid = p.get("player_id") or p.get("id")
#                 x = p.get("x")
#                 y = p.get("y")
#                 if x is None or y is None:
#                     continue
#                 if pid not in team_players:
#                     opp_pos.append({"x": x, "y": y, "id": pid})

#         # If team_pos empty, append nan and continue
#         if len(team_pos) == 0:
#             heights.append(np.nan)
#             continue

#         # Decide defensive side by comparing team mean x to opponent mean x (if available)
#         try:
#             team_mean_x = np.mean([p["x"] for p in team_pos]) if team_pos else None
#             opp_mean_x = np.mean([p["x"] for p in opp_pos]) if opp_pos else None
#         except Exception:
#             team_mean_x = None
#             opp_mean_x = None

#         # Default: assume deeper players are those with smaller x (left) if opponent unknown
#         pick_lows = True
#         if team_mean_x is not None and opp_mean_x is not None:
#             pick_lows = team_mean_x < opp_mean_x

#         # sort by x ascending
#         players_sorted = sorted(team_pos, key=lambda p: p["x"])
#         if pick_lows:
#             # deepest = players with smallest x (left side)
#             def_line = players_sorted[:n_defenders]
#         else:
#             # deepest = players with largest x (right side)
#             def_line = players_sorted[-n_defenders:]

#         # if we have fewer than required, use whatever is available
#         if len(def_line) == 0:
#             heights.append(np.nan)
#             continue

#         avg_y = float(np.mean([p["y"] for p in def_line]))
#         heights.append(avg_y)
#     return heights

# # plotting function (scatter + line) - color coded by bucket
# def plot_defensive_line_time_series(heights, title, save_path):
#     minutes = [frame_index_to_minute(i) for i in range(len(heights))]
#     buckets = [minute_to_bucket(m) for m in minutes]
#     color_map = {"deep": "tab:blue", "normal": "tab:green", "high": "tab:orange", "aggressive": "tab:red", "unknown": "gray"}
#     colors = [color_map.get(b, "gray") for b in buckets]

#     plt.figure(figsize=(12, 4))
#     plt.scatter(minutes, heights, c=colors, s=8)
#     plt.plot(minutes, heights, linewidth=0.7, alpha=0.5)
#     from matplotlib.lines import Line2D
#     handles = [Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=color_map[k], markersize=6) 
#                for k in ["deep", "normal", "high", "aggressive"]]
#     plt.legend(handles=handles, title="Minute bucket")
#     plt.title(title)
#     plt.xlabel("Match minute")
#     plt.ylabel("Defensive line mean Y")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# # compute percent time in each bucket for a heights series
# def percent_time_buckets(heights):
#     mins = [frame_index_to_minute(i) for i in range(len(heights))]
#     buckets = [minute_to_bucket(m) for m in mins]
#     s = pd.Series(buckets)
#     counts = s.value_counts().reindex(["deep","normal","high","aggressive"], fill_value=0)
#     pct = 100.0 * counts / counts.sum() if counts.sum() > 0 else counts
#     return pct.to_dict()

# # -------------------------
# # Robust loader supporting jsonl and common patterns
# # -------------------------
# def _load_jsonl_file(path):
#     frames = []
#     with open(path, "r", encoding="utf-8") as fh:
#         for line in fh:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 obj = json.loads(line)
#             except Exception:
#                 # some .jsonl may be not strictly JSON per line; skip bad lines
#                 continue
#             frames.append(obj)
#     return frames

# def load_match_from_path(path):
#     """
#     Accepts:
#       - path -> JSON file (list or dict of frames)
#       - path -> JSONL file (one JSON/frame per line)
#       - path -> directory containing tracking files (json/jsonl) and metadata
#       - path == None -> fallback to model0 loader if available
#     Returns (frames:list, meta:dict)
#     """
#     # If path is None and loader exists, use it
#     if path is None and load_tracking is not None:
#         frames = load_tracking()
#         meta = load_match_metadata() if load_match_metadata is not None else {}
#         return frames, meta

#     p = Path(path)
#     # Directory case
#     if p.is_dir():
#         # Candidate filenames (ordered)
#         candidates = [
#             "tracking.json", "structured_data.json", "frames.json",
#             f"{p.name}_tracking.json", f"{p.name}_tracking_extrapolated.jsonl",
#             f"{p.name}_tracking.jsonl"
#         ]
#         # Also include any *.jsonl or *tracking*.json* files
#         frames = None
#         meta = {}
#         for c in candidates:
#             fp = p / c
#             if fp.exists():
#                 if fp.suffix == ".jsonl":
#                     frames = _load_jsonl_file(fp)
#                 else:
#                     with open(fp, "r", encoding="utf-8") as fh:
#                         frames = json.load(fh)
#                 break

#         if frames is None:
#             # glob for more permissive patterns
#             # prefer files with "tracking" in name, then any .jsonl, then any .json
#             tracking_candidates = sorted(p.glob("*tracking*.jsonl")) + sorted(p.glob("*tracking*.json")) + sorted(p.glob("*.jsonl")) + sorted(p.glob("*.json"))
#             for fp in tracking_candidates:
#                 try:
#                     if fp.suffix == ".jsonl":
#                         frames = _load_jsonl_file(fp)
#                     else:
#                         with open(fp, "r", encoding="utf-8") as fh:
#                             data = json.load(fh)
#                             # accept either list-of-frames or dict-with-frames
#                             if isinstance(data, dict) and "frames" in data:
#                                 frames = data["frames"]
#                             elif isinstance(data, list):
#                                 frames = data
#                             else:
#                                 # sometimes tracking file is a dict mapping; try to find player_data key
#                                 frames = data if isinstance(data, list) else None
#                     if frames is not None:
#                         break
#                 except Exception:
#                     # try next candidate
#                     frames = None
#                     continue

#         # metadata: look for _match.json or metadata.json
#         for mf in ["metadata.json", "match_metadata.json", "meta.json", f"{p.name}_match.json", f"{p.name}_match.json"]:
#             mp = p / mf
#             if mp.exists():
#                 try:
#                     with open(mp, "r", encoding="utf-8") as fh:
#                         meta = json.load(fh)
#                 except Exception:
#                     meta = {}
#                 break

#         if frames is None:
#             raise FileNotFoundError(f"No tracking file found in folder {p}")
#         return frames, meta

#     # File case
#     if p.is_file():
#         if p.suffix == ".jsonl":
#             frames = _load_jsonl_file(p)
#             return frames, {}
#         else:
#             with open(p, "r", encoding="utf-8") as fh:
#                 data = json.load(fh)
#             if isinstance(data, dict):
#                 if "frames" in data:
#                     return data["frames"], data.get("meta", {}) or data.get("match", {})
#                 # if top-level has player_data then treat as frames list
#                 if "player_data" in data:
#                     return [data], {}
#             if isinstance(data, list):
#                 return data, {}
#             # fallback
#             raise ValueError(f"Unrecognized JSON structure in {p}")

#     raise FileNotFoundError(f"Path {path} not found")

# # -------------------------
# # Run for multiple matches
# # -------------------------
# def run_on_matches(match_paths, output_dir="model1_output_multi"):
#     out_root = Path(output_dir)
#     out_root.mkdir(parents=True, exist_ok=True)

#     aggregated_rows = []
#     agg_teamA_pcts = []
#     agg_teamB_pcts = []
#     match_labels = []

#     for idx, mpath in enumerate(match_paths):
#         try:
#             print(f"Processing match {idx+1}/{len(match_paths)} -> {mpath}")
#             frames, meta = load_match_from_path(mpath)
#         except Exception as e:
#             print(f"Failed to load match {mpath}: {e}")
#             continue

#         match_id = meta.get("match_id") or meta.get("id") or meta.get("match_id", None) or f"match_{idx+1}"
#         # try to normalize common numeric id from folder name if still not present
#         if match_id is None:
#             try:
#                 match_id = Path(mpath).name
#             except Exception:
#                 match_id = f"match_{idx+1}"

#         out_dir = out_root / f"{match_id}"
#         out_dir.mkdir(parents=True, exist_ok=True)

#         # Map players to teams
#         home_players, away_players, home_team_id, away_team_id = map_players_to_teams(frames, meta)

#         # Defensive line heights (pass opponent players for orientation)
#         dlA = compute_defensive_line_height(frames, team_players=home_players, opponent_players=away_players, n_defenders=4)
#         dlB = compute_defensive_line_height(frames, team_players=away_players, opponent_players=home_players, n_defenders=4)

#         # per-match plots
#         plot_defensive_line_time_series(dlA, f"Team {home_team_id or 'A'} Defensive Line - {match_id}", out_dir / f"def_line_A.png")
#         plot_defensive_line_time_series(dlB, f"Team {away_team_id or 'B'} Defensive Line - {match_id}", out_dir / f"def_line_B.png")

#         # percent time in each bucket
#         pctA = percent_time_buckets(dlA)
#         pctB = percent_time_buckets(dlB)

#         # save per-match CSV
#         df_match = pd.DataFrame({
#             "bucket":["deep","normal","high","aggressive"],
#             "teamA_pct":[pctA.get(k,0) for k in ["deep","normal","high","aggressive"]],
#             "teamB_pct":[pctB.get(k,0) for k in ["deep","normal","high","aggressive"]],
#         })
#         csv_path = out_dir / f"summary_{match_id}.csv"
#         df_match.to_csv(csv_path, index=False)
#         print(f"Saved per-match summary to {csv_path}")

#         # accumulate for aggregate
#         agg_teamA_pcts.append([pctA.get(k,0) for k in ["deep","normal","high","aggressive"]])
#         agg_teamB_pcts.append([pctB.get(k,0) for k in ["deep","normal","high","aggressive"]])
#         match_labels.append(str(match_id))
#         aggregated_rows.append((match_id, pctA, pctB))

#     # Create aggregated plots if we processed at least one match
#     if agg_teamA_pcts:
#         aggA = np.array(agg_teamA_pcts)  # shape: (n_matches, 4)
#         aggB = np.array(agg_teamB_pcts)
#         labels = ["deep","normal","high","aggressive"]

#         # create a grouped bar chart across matches for Team A
#         n = aggA.shape[0]
#         x = np.arange(n)
#         width = 0.18
#         plt.figure(figsize=(12,5))
#         for i in range(4):
#             plt.bar(x + i*width, aggA[:,i], width=width, label=labels[i])
#         plt.xticks(x + 1.5*width, match_labels, rotation=45)
#         plt.ylabel("Percent time (%)")
#         plt.title("Team A: Percent time in defensive-line buckets (per match)")
#         plt.legend()
#         plt.tight_layout()
#         aggA_path = out_root / "agg_percent_time_teamA.png"
#         plt.savefig(aggA_path)
#         plt.close()
#         print(f"Saved aggregated Team A plot to {aggA_path}")

#         # Team B
#         plt.figure(figsize=(12,5))
#         for i in range(4):
#             plt.bar(x + i*width, aggB[:,i], width=width, label=labels[i])
#         plt.xticks(x + 1.5*width, match_labels, rotation=45)
#         plt.ylabel("Percent time (%)")
#         plt.title("Team B: Percent time in defensive-line buckets (per match)")
#         plt.legend()
#         plt.tight_layout()
#         aggB_path = out_root / "agg_percent_time_teamB.png"
#         plt.savefig(aggB_path)
#         plt.close()
#         print(f"Saved aggregated Team B plot to {aggB_path}")

#         # Save aggregated CSV
#         agg_rows = []
#         for i, mid in enumerate(match_labels):
#             row = {"match_id": mid}
#             for j, k in enumerate(labels):
#                 row[f"teamA_{k}_pct"] = float(aggA[i,j])
#                 row[f"teamB_{k}_pct"] = float(aggB[i,j])
#             agg_rows.append(row)
#         df_agg = pd.DataFrame(agg_rows)
#         df_agg.to_csv(out_root / "agg_summary.csv", index=False)
#         print(f"Saved aggregated CSV to {out_root / 'agg_summary.csv'}")

#     print("All done. Outputs are in:", out_root)


# # -------------------------
# # CLI / ENTRY POINT
# # -------------------------
# if __name__ == "__main__":
#     """
#     Usage:
#       - Put your 6 match JSONs (or folders) in a directory named 'matches/' and name them any way you like.
#       - Or pass explicit paths in the list below.
#     Example:
#       python model1_multi_matches.py
#     """
#     # Attempt to find up to 6 matches automatically under ./matches/
#     candidate_dir = Path("matches")
#     if candidate_dir.exists() and candidate_dir.is_dir():
#         # pick up to 6 match files/folders inside matches/
#         entries = sorted([str(p) for p in candidate_dir.iterdir() if not p.name.startswith(".")])[:6]
#         if len(entries) == 0 and load_tracking is not None:
#             # fallback to using built-in loader 6 times (if it supports different matchs) - unlikely
#             entries = [None] * 6
#     else:
#         # default fallback: if load_tracking exists, run single match from it
#         if load_tracking is not None:
#             entries = [None]  # process the loaded match
#         else:
#             # nothing found; inform user and exit
#             raise FileNotFoundError("No 'matches/' directory found and no data-loader available. Place up to 6 match JSONs or directories under ./matches/")

#     # If user wants exactly 6, but fewer files exist, script will process whatever it finds.
#     run_on_matches(entries)

# # --- Flask-friendly wrapper (safe, uses globals().get to avoid undefined-name linting/runtime issues) ---
# from pathlib import Path
# from typing import Optional, Dict, Any
# import traceback

# def _call_entrypoint_for_model1(match_path: Optional[str] = None):
#     """
#     Safely call the most-likely entrypoint in this module:
#       - run_on_matches(list_of_paths)
#       - run(path) (if present)
#     Uses globals().get(...) so editors won't flag undefined names.
#     """
#     # prefer model_main if defined
#     model_main_fn = globals().get("model_main")
#     if callable(model_main_fn):
#         return model_main_fn(match_path)

#     # prefer run_on_matches if present
#     run_on_matches_fn = globals().get("run_on_matches")
#     if callable(run_on_matches_fn):
#         # build entries list
#         if match_path is None:
#             repo_root = Path(__file__).resolve().parents[1]
#             candidate_dir = repo_root / "matches"
#             if candidate_dir.exists() and candidate_dir.is_dir():
#                 entries = sorted([str(p) for p in candidate_dir.iterdir() if not p.name.startswith(".")])[:6]
#             else:
#                 opendata_matches = repo_root / "opendata" / "data" / "matches"
#                 if opendata_matches.exists() and opendata_matches.is_dir():
#                     dirs = sorted([d for d in opendata_matches.iterdir() if d.is_dir()])[:6]
#                     entries = [str(d) for d in dirs]
#                 else:
#                     entries = []
#         else:
#             p = Path(match_path)
#             if p.is_dir():
#                 entries = sorted([str(x) for x in p.iterdir() if not x.name.startswith(".")])[:6] or [str(p)]
#             else:
#                 entries = [str(p)]
#         return run_on_matches_fn(entries)

#     # fallback to run(...) if present
#     run_fn = globals().get("run")
#     if callable(run_fn):
#         try:
#             return run_fn(match_path) if match_path is not None else run_fn()
#         except TypeError:
#             return run_fn()

#     # nothing found
#     raise RuntimeError("No entrypoint found for model1. Expected run_on_matches(...) or run(...).")

# def model_main(match_path: Optional[str]) -> Dict[str, Any]:
#     """
#     Wrapper expected by app.py. Runs the model entrypoint and copies PNGs to static/plots.
#     Returns: { "plots": [...], "output_folder": "..." }
#     """
#     repo_root = Path(__file__).resolve().parents[1]
#     static_plots = repo_root / "static" / "plots"
#     static_plots.mkdir(parents=True, exist_ok=True)

#     try:
#         ret = _call_entrypoint_for_model1(match_path)
#     except Exception:
#         traceback.print_exc()
#         raise

#     # collect candidate output folders to copy images from
#     out_candidates = []
#     if "OUTPUT" in globals():
#         try:
#             out_candidates.append(Path(globals()["OUTPUT"]))
#         except Exception:
#             pass
#     out_candidates.extend([
#         repo_root / "model1_output_multi",
#         repo_root / "model1_output",
#         repo_root / "model1_output_team",
#         repo_root / "matches"
#     ])

#     found_plots = []
#     used_output = None
#     for cand in out_candidates:
#         try:
#             if cand and cand.exists():
#                 for p in sorted(cand.rglob("*.png")):
#                     try:
#                         target = static_plots / p.name
#                         target.write_bytes(p.read_bytes())
#                         rel = str(Path("plots") / p.name)
#                         if rel not in found_plots:
#                             found_plots.append(rel)
#                             used_output = str(cand.resolve())
#                     except Exception:
#                         continue
#         except Exception:
#             continue

#     # include any plots returned by the model itself
#     if isinstance(ret, dict):
#         if "plots" in ret and isinstance(ret["plots"], list):
#             for p in ret["plots"]:
#                 if p not in found_plots:
#                     found_plots.append(p)

#     return {"plots": found_plots, "output_folder": used_output or str(out_candidates[0] if out_candidates else repo_root)}
# def model_main(match_path, *args, **kwargs):
#     global _model1_running
#     if _model1_running:
#         # Prevent infinite re-entry — raise an informative exception
#         raise RuntimeError("model_main re-entered (recursive call detected). Aborting.")
#     _model1_running = True
#     try:
#         # --- original body of model_main starts here ---
#         # (KEEP the existing code as-is; nothing else changes)
#         ...
#         # --- original body of model_main ends here ---
#     finally:
#         _model1_running = False

# model1_multi_matches.py
# Extended model1 to handle multiple matches (default: 6)
# - robust loader that accepts .jsonl, *_tracking*.json, structured_data.json etc.
# - defensive-line logic selects defenders relative to opponent per frame

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import glob
# ======= re-entrancy guard (add near top of file) =======
_model1_running = False
# =======================================================

# Try importing your existing data loader if available
try:
    from model0_load_data import load_tracking, load_match_metadata
except Exception:
    load_tracking = None
    load_match_metadata = None

FPS = 25  # default frames-per-second used for minute conversion

# -------------------------
# Helpers (same logic as in model1)
# -------------------------
def frame_index_to_minute(frame_idx):
    return frame_idx / (FPS * 60.0)

def minute_to_bucket(minute):
    if minute < 0:
        return "unknown"
    if minute < 30:
        return "deep"
    if minute < 60:
        return "normal"
    if minute < 90:
        return "high"
    return "aggressive"

def map_players_to_teams(frames, meta):
    home_team_id = meta.get("home_team", {}).get("id") if isinstance(meta, dict) else None
    away_team_id = meta.get("away_team", {}).get("id") if isinstance(meta, dict) else None

    all_pids = []
    for f in frames:
        for p in f.get("player_data", []):
            pid = p.get("player_id") or p.get("id")
            if pid is not None:
                all_pids.append(pid)
    all_pids = list(dict.fromkeys(all_pids))

    home_players = set()
    away_players = set()
    found_teamfield = False
    for f in frames:
        for p in f.get("player_data", []):
            pid = p.get("player_id") or p.get("id")
            if pid is None:
                continue
            if p.get("team_id") is not None:
                found_teamfield = True
                if p.get("team_id") == home_team_id:
                    home_players.add(pid)
                elif p.get("team_id") == away_team_id:
                    away_players.add(pid)
            elif p.get("team") is not None:
                found_teamfield = True
                if p.get("team") == meta.get("home_team", {}).get("name") or p.get("team") == home_team_id:
                    home_players.add(pid)
                else:
                    away_players.add(pid)

    if not found_teamfield or (not home_players and not away_players):
        # fallback: split list of unique player ids in half
        mid = len(all_pids) // 2
        home_players = set(all_pids[:mid])
        away_players = set(all_pids[mid:])

    return home_players, away_players, home_team_id, away_team_id

def compute_defensive_line_height(frames, team_players=None, opponent_players=None, n_defenders=4):
    """
    For each frame:
      - collect team player x,y positions
      - collect opponent player x positions to determine relative orientation
      - if team mean x < opponent mean x -> team is to the left -> deepest players are smallest x
        else deepest players are largest x
      - pick n_defenders deepest players and return mean Y of those players for that frame
    Returns list of mean Y (one per frame)
    """
    heights = []
    for f in frames:
        # gather team and opponent positions
        team_pos = []
        opp_pos = []
        for p in f.get("player_data", []):
            pid = p.get("player_id") or p.get("id")
            x = p.get("x")
            y = p.get("y")
            if x is None or y is None:
                continue
            if team_players is not None and pid in team_players:
                team_pos.append({"x": x, "y": y, "id": pid})
            elif opponent_players is not None and pid in opponent_players:
                opp_pos.append({"x": x, "y": y, "id": pid})
            else:
                # if team/opponent sets not provided, collect for team_pos (fallback)
                if team_players is None and opponent_players is None:
                    team_pos.append({"x": x, "y": y, "id": pid})

        # If we don't have explicit opponent players but have team players, infer opponent as others
        if opponent_players is None and team_players is not None:
            for p in f.get("player_data", []):
                pid = p.get("player_id") or p.get("id")
                x = p.get("x")
                y = p.get("y")
                if x is None or y is None:
                    continue
                if pid not in team_players:
                    opp_pos.append({"x": x, "y": y, "id": pid})

        # If team_pos empty, append nan and continue
        if len(team_pos) == 0:
            heights.append(np.nan)
            continue

        # Decide defensive side by comparing team mean x to opponent mean x (if available)
        try:
            team_mean_x = np.mean([p["x"] for p in team_pos]) if team_pos else None
            opp_mean_x = np.mean([p["x"] for p in opp_pos]) if opp_pos else None
        except Exception:
            team_mean_x = None
            opp_mean_x = None

        # Default: assume deeper players are those with smaller x (left) if opponent unknown
        pick_lows = True
        if team_mean_x is not None and opp_mean_x is not None:
            pick_lows = team_mean_x < opp_mean_x

        # sort by x ascending
        players_sorted = sorted(team_pos, key=lambda p: p["x"])
        if pick_lows:
            # deepest = players with smallest x (left side)
            def_line = players_sorted[:n_defenders]
        else:
            # deepest = players with largest x (right side)
            def_line = players_sorted[-n_defenders:]

        # if we have fewer than required, use whatever is available
        if len(def_line) == 0:
            heights.append(np.nan)
            continue

        avg_y = float(np.mean([p["y"] for p in def_line]))
        heights.append(avg_y)
    return heights

# plotting function (scatter + line) - color coded by bucket
def plot_defensive_line_time_series(heights, title, save_path):
    minutes = [frame_index_to_minute(i) for i in range(len(heights))]
    buckets = [minute_to_bucket(m) for m in minutes]
    color_map = {"deep": "tab:blue", "normal": "tab:green", "high": "tab:orange", "aggressive": "tab:red", "unknown": "gray"}
    colors = [color_map.get(b, "gray") for b in buckets]

    plt.figure(figsize=(12, 4))
    plt.scatter(minutes, heights, c=colors, s=8)
    plt.plot(minutes, heights, linewidth=0.7, alpha=0.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=color_map[k], markersize=6) 
               for k in ["deep", "normal", "high", "aggressive"]]
    plt.legend(handles=handles, title="Minute bucket")
    plt.title(title)
    plt.xlabel("Match minute")
    plt.ylabel("Defensive line mean Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# compute percent time in each bucket for a heights series
def percent_time_buckets(heights):
    mins = [frame_index_to_minute(i) for i in range(len(heights))]
    buckets = [minute_to_bucket(m) for m in mins]
    s = pd.Series(buckets)
    counts = s.value_counts().reindex(["deep","normal","high","aggressive"], fill_value=0)
    pct = 100.0 * counts / counts.sum() if counts.sum() > 0 else counts
    return pct.to_dict()

# -------------------------
# Robust loader supporting jsonl and common patterns
# -------------------------
def _load_jsonl_file(path):
    frames = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # some .jsonl may be not strictly JSON per line; skip bad lines
                continue
            frames.append(obj)
    return frames

def load_match_from_path(path):
    """
    Accepts:
      - path -> JSON file (list or dict of frames)
      - path -> JSONL file (one JSON/frame per line)
      - path -> directory containing tracking files (json/jsonl) and metadata
      - path == None -> fallback to model0 loader if available
    Returns (frames:list, meta:dict)
    """
    # If path is None and loader exists, use it
    if path is None and load_tracking is not None:
        frames = load_tracking()
        meta = load_match_metadata() if load_match_metadata is not None else {}
        return frames, meta

    p = Path(path)
    # Directory case
    if p.is_dir():
        # Candidate filenames (ordered)
        candidates = [
            "tracking.json", "structured_data.json", "frames.json",
            f"{p.name}_tracking.json", f"{p.name}_tracking_extrapolated.jsonl",
            f"{p.name}_tracking.jsonl"
        ]
        # Also include any *.jsonl or *tracking*.json* files
        frames = None
        meta = {}
        for c in candidates:
            fp = p / c
            if fp.exists():
                if fp.suffix == ".jsonl":
                    frames = _load_jsonl_file(fp)
                else:
                    with open(fp, "r", encoding="utf-8") as fh:
                        frames = json.load(fh)
                break

        if frames is None:
            # glob for more permissive patterns
            # prefer files with "tracking" in name, then any .jsonl, then any .json
            tracking_candidates = sorted(p.glob("*tracking*.jsonl")) + sorted(p.glob("*tracking*.json")) + sorted(p.glob("*.jsonl")) + sorted(p.glob("*.json"))
            for fp in tracking_candidates:
                try:
                    if fp.suffix == ".jsonl":
                        frames = _load_jsonl_file(fp)
                    else:
                        with open(fp, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                            # accept either list-of-frames or dict-with-frames
                            if isinstance(data, dict) and "frames" in data:
                                frames = data["frames"]
                            elif isinstance(data, list):
                                frames = data
                            else:
                                # sometimes tracking file is a dict mapping; try to find player_data key
                                frames = data if isinstance(data, list) else None
                    if frames is not None:
                        break
                except Exception:
                    # try next candidate
                    frames = None
                    continue

        # metadata: look for _match.json or metadata.json
        for mf in ["metadata.json", "match_metadata.json", "meta.json", f"{p.name}_match.json", f"{p.name}_match.json"]:
            mp = p / mf
            if mp.exists():
                try:
                    with open(mp, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                except Exception:
                    meta = {}
                break

        if frames is None:
            raise FileNotFoundError(f"No tracking file found in folder {p}")
        return frames, meta

    # File case
    if p.is_file():
        if p.suffix == ".jsonl":
            frames = _load_jsonl_file(p)
            return frames, {}
        else:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                if "frames" in data:
                    return data["frames"], data.get("meta", {}) or data.get("match", {})
                # if top-level has player_data then treat as frames list
                if "player_data" in data:
                    return [data], {}
            if isinstance(data, list):
                return data, {}
            # fallback
            raise ValueError(f"Unrecognized JSON structure in {p}")

    raise FileNotFoundError(f"Path {path} not found")

# -------------------------
# Run for multiple matches
# -------------------------
def run_on_matches(match_paths, output_dir="model1_output_multi"):
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    aggregated_rows = []
    agg_teamA_pcts = []
    agg_teamB_pcts = []
    match_labels = []

    for idx, mpath in enumerate(match_paths):
        try:
            print(f"Processing match {idx+1}/{len(match_paths)} -> {mpath}")
            frames, meta = load_match_from_path(mpath)
        except Exception as e:
            print(f"Failed to load match {mpath}: {e}")
            continue

        match_id = meta.get("match_id") or meta.get("id") or meta.get("match_id", None) or f"match_{idx+1}"
        # try to normalize common numeric id from folder name if still not present
        if match_id is None:
            try:
                match_id = Path(mpath).name
            except Exception:
                match_id = f"match_{idx+1}"

        out_dir = out_root / f"{match_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Map players to teams
        home_players, away_players, home_team_id, away_team_id = map_players_to_teams(frames, meta)

        # Defensive line heights (pass opponent players for orientation)
        dlA = compute_defensive_line_height(frames, team_players=home_players, opponent_players=away_players, n_defenders=4)
        dlB = compute_defensive_line_height(frames, team_players=away_players, opponent_players=home_players, n_defenders=4)

        # per-match plots
        plot_defensive_line_time_series(dlA, f"Team {home_team_id or 'A'} Defensive Line - {match_id}", out_dir / f"def_line_A.png")
        plot_defensive_line_time_series(dlB, f"Team {away_team_id or 'B'} Defensive Line - {match_id}", out_dir / f"def_line_B.png")

        # percent time in each bucket
        pctA = percent_time_buckets(dlA)
        pctB = percent_time_buckets(dlB)

        # save per-match CSV
        df_match = pd.DataFrame({
            "bucket":["deep","normal","high","aggressive"],
            "teamA_pct":[pctA.get(k,0) for k in ["deep","normal","high","aggressive"]],
            "teamB_pct":[pctB.get(k,0) for k in ["deep","normal","high","aggressive"]],
        })
        csv_path = out_dir / f"summary_{match_id}.csv"
        df_match.to_csv(csv_path, index=False)
        print(f"Saved per-match summary to {csv_path}")

        # accumulate for aggregate
        agg_teamA_pcts.append([pctA.get(k,0) for k in ["deep","normal","high","aggressive"]])
        agg_teamB_pcts.append([pctB.get(k,0) for k in ["deep","normal","high","aggressive"]])
        match_labels.append(str(match_id))
        aggregated_rows.append((match_id, pctA, pctB))

    # Create aggregated plots if we processed at least one match
    if agg_teamA_pcts:
        aggA = np.array(agg_teamA_pcts)  # shape: (n_matches, 4)
        aggB = np.array(agg_teamB_pcts)
        labels = ["deep","normal","high","aggressive"]

        # create a grouped bar chart across matches for Team A
        n = aggA.shape[0]
        x = np.arange(n)
        width = 0.18
        plt.figure(figsize=(12,5))
        for i in range(4):
            plt.bar(x + i*width, aggA[:,i], width=width, label=labels[i])
        plt.xticks(x + 1.5*width, match_labels, rotation=45)
        plt.ylabel("Percent time (%)")
        plt.title("Team A: Percent time in defensive-line buckets (per match)")
        plt.legend()
        plt.tight_layout()
        aggA_path = out_root / "agg_percent_time_teamA.png"
        plt.savefig(aggA_path)
        plt.close()
        print(f"Saved aggregated Team A plot to {aggA_path}")

        # Team B
        plt.figure(figsize=(12,5))
        for i in range(4):
            plt.bar(x + i*width, aggB[:,i], width=width, label=labels[i])
        plt.xticks(x + 1.5*width, match_labels, rotation=45)
        plt.ylabel("Percent time (%)")
        plt.title("Team B: Percent time in defensive-line buckets (per match)")
        plt.legend()
        plt.tight_layout()
        aggB_path = out_root / "agg_percent_time_teamB.png"
        plt.savefig(aggB_path)
        plt.close()
        print(f"Saved aggregated Team B plot to {aggB_path}")

        # Save aggregated CSV
        agg_rows = []
        for i, mid in enumerate(match_labels):
            row = {"match_id": mid}
            for j, k in enumerate(labels):
                row[f"teamA_{k}_pct"] = float(aggA[i,j])
                row[f"teamB_{k}_pct"] = float(aggB[i,j])
            agg_rows.append(row)
        df_agg = pd.DataFrame(agg_rows)
        df_agg.to_csv(out_root / "agg_summary.csv", index=False)
        print(f"Saved aggregated CSV to {out_root / 'agg_summary.csv'}")

    print("All done. Outputs are in:", out_root)


# -------------------------
# CLI / ENTRY POINT
# -------------------------
if __name__ == "__main__":
    """
    Usage:
      - Put your 6 match JSONs (or folders) in a directory named 'matches/' and name them any way you like.
      - Or pass explicit paths in the list below.
    Example:
      python model1_multi_matches.py
    """
    # Attempt to find up to 6 matches automatically under ./matches/
    candidate_dir = Path("matches")
    if candidate_dir.exists() and candidate_dir.is_dir():
        # pick up to 6 match files/folders inside matches/
        entries = sorted([str(p) for p in candidate_dir.iterdir() if not p.name.startswith(".")])[:6]
        if len(entries) == 0 and load_tracking is not None:
            # fallback to using built-in loader 6 times (if it supports different matchs) - unlikely
            entries = [None] * 6
    else:
        # default fallback: if load_tracking exists, run single match from it
        if load_tracking is not None:
            entries = [None]  # process the loaded match
        else:
            # nothing found; inform user and exit
            raise FileNotFoundError("No 'matches/' directory found and no data-loader available. Place up to 6 match JSONs or directories under ./matches/")

    # If user wants exactly 6, but fewer files exist, script will process whatever it finds.
    run_on_matches(entries)

# --- Flask-friendly wrapper (safe, uses globals().get to avoid undefined-name linting/runtime issues) ---
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

# def _call_entrypoint_for_model1(match_path: Optional[str] = None):
#     """
#     Safely call the most-likely entrypoint in this module:
#       - run_on_matches(list_of_paths)
#       - run(path) (if present)
#     Uses globals().get(...) so editors won't flag undefined names.
#     """
#     # prefer model_main if defined
#     model_main_fn = globals().get("model_main")
#     if callable(model_main_fn):
#         return model_main_fn(match_path)
# --- Corrected snippet for /Users/nishoosingh/Downloads/model/models/model1.py ---

def _call_entrypoint_for_model1(match_path: Optional[str] = None):
    """
    Safely call the most-likely entrypoint in this module:
      - run_on_matches(list_of_paths)
    """
    
    # 1. Skip model_main check, as it's the wrapper.
    #    Directly check for the actual logic function: run_on_matches

    run_on_matches_fn = globals().get("run_on_matches")
    if callable(run_on_matches_fn):
        # build entries list (existing logic)
        # ... [Keep your existing logic for building the 'entries' list] ...
        
        if match_path is None:
            # ... (logic to find matches directory, etc.)
            repo_root = Path(__file__).resolve().parents[1]
            candidate_dir = repo_root / "matches"
            if candidate_dir.exists() and candidate_dir.is_dir():
                entries = sorted([str(p) for p in candidate_dir.iterdir() if not p.name.startswith(".")])[:6]
            else:
                opendata_matches = repo_root / "opendata" / "data" / "matches"
                if opendata_matches.exists() and opendata_matches.is_dir():
                    dirs = sorted([d for d in opendata_matches.iterdir() if d.is_dir()])[:6]
                    entries = [str(d) for d in dirs]
                else:
                    entries = []
        else:
            p = Path(match_path)
            if p.is_dir():
                entries = sorted([str(x) for x in p.iterdir() if not x.name.startswith(".")])[:6] or [str(p)]
            else:
                entries = [str(p)]

        return run_on_matches_fn(entries) # <-- Call the actual logic function

    # fallback to run(...) if present (keep this fallback)
    # ... [Keep your fallback logic for 'run' or other entry points] ...
    run_fn = globals().get("run")
    if callable(run_fn):
        try:
            return run_fn(match_path) if match_path is not None else run_fn()
        except TypeError:
            return run_fn()

    # nothing found
    raise RuntimeError("No entrypoint found for model1. Expected run_on_matches(...) or run(...).")
    # prefer run_on_matches if present
    run_on_matches_fn = globals().get("run_on_matches")
    if callable(run_on_matches_fn):
        # build entries list
        if match_path is None:
            repo_root = Path(__file__).resolve().parents[1]
            candidate_dir = repo_root / "matches"
            if candidate_dir.exists() and candidate_dir.is_dir():
                entries = sorted([str(p) for p in candidate_dir.iterdir() if not p.name.startswith(".")])[:6]
            else:
                opendata_matches = repo_root / "opendata" / "data" / "matches"
                if opendata_matches.exists() and opendata_matches.is_dir():
                    dirs = sorted([d for d in opendata_matches.iterdir() if d.is_dir()])[:6]
                    entries = [str(d) for d in dirs]
                else:
                    entries = []
        else:
            p = Path(match_path)
            if p.is_dir():
                entries = sorted([str(x) for x in p.iterdir() if not x.name.startswith(".")])[:6] or [str(p)]
            else:
                entries = [str(p)]
        return run_on_matches_fn(entries)

    # fallback to run(...) if present
    run_fn = globals().get("run")
    if callable(run_fn):
        try:
            return run_fn(match_path) if match_path is not None else run_fn()
        except TypeError:
            return run_fn()

    # nothing found
    raise RuntimeError("No entrypoint found for model1. Expected run_on_matches(...) or run(...).")

def model_main(match_path: Optional[str]) -> Dict[str, Any]:
    """
    Wrapper expected by app.py. Runs the model entrypoint and copies PNGs to static/plots.
    Returns: { "plots": [...], "output_folder": "..." }
    """
    global _model1_running
    if _model1_running:
        # Prevent infinite re-entry — raise an informative exception
        raise RuntimeError("model_main re-entered (recursive call detected). Aborting.")
    
    _model1_running = True
    try:
        repo_root = Path(__file__).resolve().parents[1]
        static_plots = repo_root / "static" / "plots"
        static_plots.mkdir(parents=True, exist_ok=True)

        try:
            ret = _call_entrypoint_for_model1(match_path)
        except Exception:
            traceback.print_exc()
            raise

        # collect candidate output folders to copy images from
        out_candidates = []
        if "OUTPUT" in globals():
            try:
                out_candidates.append(Path(globals()["OUTPUT"]))
            except Exception:
                pass
        out_candidates.extend([
            repo_root / "model1_output_multi",
            repo_root / "model1_output",
            repo_root / "model1_output_team",
            repo_root / "matches"
        ])

        found_plots = []
        used_output = None
        for cand in out_candidates:
            try:
                if cand and cand.exists():
                    for p in sorted(cand.rglob("*.png")):
                        try:
                            target = static_plots / p.name
                            target.write_bytes(p.read_bytes())
                            rel = str(Path("plots") / p.name)
                            if rel not in found_plots:
                                found_plots.append(rel)
                                used_output = str(cand.resolve())
                        except Exception:
                            continue
            except Exception:
                continue

        # include any plots returned by the model itself
        if isinstance(ret, dict):
            if "plots" in ret and isinstance(ret["plots"], list):
                for p in ret["plots"]:
                    if p not in found_plots:
                        found_plots.append(p)

        return {"plots": found_plots, "output_folder": used_output or str(out_candidates[0] if out_candidates else repo_root)}
    finally:
        _model1_running = False