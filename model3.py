# # model3_zone_focals_cells.py
# """
# Robust version — per-team per-zone focal points mapped to a 3x3 cell inside the zone.
# - Tries multiple possible player id / x / y keys
# - Filters to valid frames (frames that contain usable player positions)
# - Normalizes metadata IDs to match frame pid types where possible
# - Produces per-team-per-zone PNGs and zone_focals_cells_summary.json

# Usage:
#     python model3_zone_focals_cells.py
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from pathlib import Path
# import json
# from typing import List, Dict, Tuple, Optional, Any

# # Project loaders - must exist
# from model0_load_data import load_tracking, load_match_metadata

# # ---------- configuration ----------
# FIELD_X = (-52.5, 52.5)
# FIELD_Y = (-34.0, 34.0)
# GRID_X = 128
# GRID_Y = 96
# EPS = 0.5

# OUT = Path("model3_output_zone_focals_cells")
# OUT.mkdir(parents=True, exist_ok=True)

# # ---------- utils: tolerant extraction ----------
# def extract_player_xy_and_id(player: dict) -> Tuple[Optional[Any], Optional[float], Optional[float]]:
#     """Try many possible keys to extract player id and x,y coordinates."""
#     # ID candidates
#     pid = None
#     for k in ("player_id", "id", "pid", "playerId", "playerIdInt"):
#         if k in player and player[k] is not None:
#             pid = player[k]
#             break
#     # x candidates
#     x = None
#     for k in ("x","X","pos_x","posX","x_world","x_pos","xf","x_coord","position_x"):
#         if k in player and player[k] is not None:
#             x = player[k]; break
#     # y candidates
#     y = None
#     for k in ("y","Y","pos_y","posY","y_world","y_pos","yf","y_coord","position_y"):
#         if k in player and player[k] is not None:
#             y = player[k]; break
#     # nested position dict?
#     if (x is None or y is None) and isinstance(player.get("position"), dict):
#         pos = player.get("position")
#         if x is None:
#             for k in ("x","X","pos_x","posX"):
#                 if k in pos and pos[k] is not None:
#                     x = pos[k]; break
#         if y is None:
#             for k in ("y","Y","pos_y","posY"):
#                 if k in pos and pos[k] is not None:
#                     y = pos[k]; break
#     # final conversions
#     try:
#         if x is not None:
#             x = float(x)
#     except Exception:
#         x = None
#     try:
#         if y is not None:
#             y = float(y)
#     except Exception:
#         y = None
#     return pid, x, y

# def sample_frame_player_ids(frames: List[dict], limit: int = 200) -> List[Any]:
#     """Return a list of sample player ids observed in frames (first few frames)."""
#     ids = []
#     for f in frames[:10]:
#         pd = f.get("player_data", []) or f.get("players", []) or []
#         for p in pd:
#             pid, x, y = extract_player_xy_and_id(p)
#             if pid is not None and pid not in ids:
#                 ids.append(pid)
#             if len(ids) >= limit:
#                 return ids
#     return ids

# def normalize_metadata_ids(meta_ids: List[Any], sample_pids: List[Any]) -> List[Any]:
#     """
#     Try to convert metadata IDs to types that match sample frame pids.
#     Returns normalized list where possible (best-effort).
#     """
#     if not meta_ids:
#         return []
#     # try to detect sample pid type
#     if not sample_pids:
#         return meta_ids
#     sample_example = sample_pids[0]
#     normalized = []
#     for mid in meta_ids:
#         found = False
#         # try exact match first
#         if mid in sample_pids:
#             normalized.append(mid); continue
#         # try int / str conversion combos
#         try:
#             mid_int = int(mid)
#         except Exception:
#             mid_int = None
#         mid_str = str(mid)
#         if mid_int is not None and mid_int in sample_pids:
#             normalized.append(mid_int); continue
#         # try matching as string
#         if mid_str in sample_pids:
#             normalized.append(mid_str); continue
#         # if sample pids are ints but meta ids are strings of ints, convert
#         converted = None
#         if isinstance(sample_example, int):
#             try:
#                 converted = int(mid)
#             except Exception:
#                 converted = None
#             if converted is not None:
#                 normalized.append(converted); continue
#         # if sample pids are strings and meta are ints, convert to str
#         if isinstance(sample_example, str):
#             normalized.append(str(mid)); continue
#         # fallback append original
#         normalized.append(mid)
#     # dedupe while preserving order
#     seen = set()
#     out = []
#     for x in normalized:
#         if x in seen: continue
#         seen.add(x); out.append(x)
#     return out

# # ---------- frame selection ----------
# def frame_has_usable_player(frame: dict) -> bool:
#     pd = frame.get("player_data", []) or frame.get("players", []) or []
#     for p in pd:
#         # skip players explicitly marked undetected
#         if p.get("is_detected") is False:
#             continue
#         pid, x, y = extract_player_xy_and_id(p)
#         if pid is not None and x is not None and y is not None:
#             return True
#     return False

# def select_valid_frames(frames: List[dict]) -> List[dict]:
#     valid = [f for f in frames if frame_has_usable_player(f)]
#     return valid

# # ---------- grid computation (robust) ----------
# def compute_full_grids(frames: List[dict], home_ids: List[Any], away_ids: List[Any],
#                        gx: int = GRID_X, gy: int = GRID_Y):
#     # select only frames with usable player positions
#     frames_valid = select_valid_frames(frames)
#     print(f"Total frames loaded: {len(frames)}, frames with usable players: {len(frames_valid)}")
#     if len(frames_valid) == 0:
#         print("ERROR: No valid frames found. Check your tracking loader or JSON format.")
#     xs = np.linspace(FIELD_X[0], FIELD_X[1], gx)
#     ys = np.linspace(FIELD_Y[0], FIELD_Y[1], gy)
#     Xc, Yc = np.meshgrid(xs, ys)

#     accumA = np.zeros((gy, gx), dtype=float)
#     accumB = np.zeros((gy, gx), dtype=float)
#     frames_used = 0

#     # convert id lists to sets for speed
#     home_set = set(home_ids or [])
#     away_set = set(away_ids or [])

#     sample_positions = []
#     for frame in frames_valid:
#         pd = frame.get("player_data", []) or frame.get("players", []) or []
#         pos = {}
#         for p in pd:
#             if p.get("is_detected") is False:
#                 continue
#             pid_raw, x, y = extract_player_xy_and_id(p)
#             if pid_raw is None or x is None or y is None:
#                 continue
#             # normalize pid types: try int if possible
#             try:
#                 pid = int(pid_raw)
#             except Exception:
#                 pid = pid_raw
#             pos[pid] = (float(x), float(y))
#         if not pos:
#             continue

#         # record sample
#         if len(sample_positions) < 3:
#             sample_positions.append({k: pos[k] for k in list(pos.keys())[:4]})

#         frames_used += 1
#         frameA = np.zeros_like(accumA)
#         frameB = np.zeros_like(accumB)

#         # determine which player ids belong to each team
#         if not home_set and not away_set:
#             pids = list(pos.keys())
#             half = len(pids)//2
#             home_iter = pids[:half]
#             away_iter = pids[half:]
#         else:
#             # try matching with sets; if types differ (string vs int) previous normalization should help
#             home_iter = [pid for pid in home_set if pid in pos]
#             away_iter = [pid for pid in away_set if pid in pos]
#             # fallback: if no ids matched (maybe meta ids are strings but pos keys ints), try converting
#             if not home_iter and home_set:
#                 # attempt type-flexible match
#                 for pid in pos:
#                     if str(pid) in set(map(str, home_set)):
#                         home_iter.append(pid)
#             if not away_iter and away_set:
#                 for pid in pos:
#                     if str(pid) in set(map(str, away_set)):
#                         away_iter.append(pid)

#         # accumulate per-player influence
#         for pid in home_iter:
#             pp = pos.get(pid)
#             if pp is None:
#                 continue
#             dx = Xc - pp[0]; dy = Yc - pp[1]; d = np.hypot(dx, dy)
#             frameA += 1.0 / (d + EPS)
#         for pid in away_iter:
#             pp = pos.get(pid)
#             if pp is None:
#                 continue
#             dx = Xc - pp[0]; dy = Yc - pp[1]; d = np.hypot(dx, dy)
#             frameB += 1.0 / (d + EPS)

#         accumA += frameA
#         accumB += frameB

#     print("Frames used for averaging:", frames_used)
#     if frames_used > 0:
#         accumA /= float(frames_used)
#         accumB /= float(frames_used)
#     else:
#         print("WARNING: No frames used. Sample positions (if any):", sample_positions)

#     return accumA, accumB, xs, ys

# # ---------- zone & centroid helpers ----------
# def get_zone_bounds():
#     x_min, x_max = FIELD_X
#     total = x_max - x_min
#     w = total / 3.0
#     return [
#         (x_min, x_min + w),        # defensive (left)
#         (x_min + w, x_min + 2*w),  # midfield
#         (x_min + 2*w, x_max)       # attacking
#     ]

# def masked_centroid(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray, x_left: float, x_right: float) -> Tuple[Optional[float], Optional[float], float]:
#     Xc, Yc = np.meshgrid(xs, ys)
#     mask = (Xc >= x_left) & (Xc < x_right)
#     masked = grid * mask
#     total = float(np.sum(masked))
#     if total <= 0.0:
#         return None, None, 0.0
#     cx = float(np.sum(masked * Xc) / total)
#     cy = float(np.sum(masked * Yc) / total)
#     return cx, cy, total

# def compute_zone_cell_stats(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray,
#                             x_left: float, x_right: float) -> Tuple[np.ndarray, List[Tuple[float,float,float,float]]]:
#     x_edges = [x_left + i * ((x_right - x_left) / 3.0) for i in range(4)]
#     y_min, y_max = ys[0], ys[-1]
#     y_edges = [y_min + i * ((y_max - y_min) / 3.0) for i in range(4)]
#     stats = np.zeros((3,3), dtype=float)
#     boxes = []
#     Xc, Yc = np.meshgrid(xs, ys)
#     for row in range(3):
#         y_lo, y_hi = y_edges[row], y_edges[row+1]
#         for col in range(3):
#             x_lo, x_hi = x_edges[col], x_edges[col+1]
#             mask = (Xc >= x_lo) & (Xc < x_hi) & (Yc >= y_lo) & (Yc < y_hi)
#             if np.any(mask):
#                 stats[row, col] = float(np.mean(grid[mask]))
#             else:
#                 stats[row, col] = 0.0
#             boxes.append((x_lo, x_hi, y_lo, y_hi))
#     return stats, boxes

# def find_cell_for_point(cx: float, cy: float, boxes: List[Tuple[float,float,float,float]]) -> Optional[int]:
#     if cx is None or cy is None:
#         return None
#     for idx, (xl, xr, yb, yt) in enumerate(boxes):
#         if (cx >= xl) and (cx < xr) and (cy >= yb) and (cy < yt):
#             return idx + 1
#     return None

# # ---------- plotting ----------
# def plot_zone_cells(grid: np.ndarray, xs: np.ndarray, ys: np.ndarray,
#                     x_left: float, x_right: float, boxes: List[Tuple[float,float,float,float]],
#                     focal: Tuple[Optional[float],Optional[float]], focal_cell: Optional[int],
#                     team_label: str, zone_label: str, out_path: Path, stats: np.ndarray):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     Xc, Yc = np.meshgrid(xs, ys)
#     zone_mask = (Xc >= x_left) & (Xc < x_right)
#     visual = np.copy(grid)
#     visual[~zone_mask] = visual[~zone_mask] * 0.12

#     plt.figure(figsize=(10,7))
#     im = plt.imshow(visual, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap="viridis")
#     plt.colorbar(im, label="Aggregated control (arb. units)")
#     ax = plt.gca()
#     # draw cells
#     for (xl, xr, yb, yt) in boxes:
#         rect = Rectangle((xl, yb), xr - xl, yt - yb, fill=False, edgecolor="white", lw=1.8)
#         ax.add_patch(rect)
#     # annotate means
#     for i, (xl, xr, yb, yt) in enumerate(boxes):
#         row = i // 3; col = i % 3
#         value = stats[row, col]
#         cx_text = (xl + xr) / 2.0
#         cy_text = (yb + yt) / 2.0
#         ax.text(cx_text, cy_text, f"{value:.2e}", color="white", fontsize=9, ha="center", va="center",
#                 bbox=dict(facecolor='black', alpha=0.5, pad=1))
#     # focal marker
#     fx, fy = focal
#     if fx is not None and fy is not None:
#         ax.plot(fx, fy, marker='X', color='red', markersize=12, markeredgecolor='k', zorder=10)
#         cell_label = f"cell {focal_cell}" if focal_cell is not None else "outside"
#         ax.text(fx + 1.0, fy + 1.0, f"focal ({cell_label})", color='white', bbox=dict(facecolor='black', alpha=0.7))
#     else:
#         ax.text(xs[0] + (x_right - x_left)/8.0, ys[-1] - 3.0, "No focal (no control mass)", color='white',
#                 bbox=dict(facecolor='black', alpha=0.7))
#     ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
#     ax.set_xlabel("Pitch X (m)"); ax.set_ylabel("Pitch Y (m)")
#     ax.set_title(f"{team_label} — {zone_label.capitalize()} (3x3 cells; focal cell highlighted)")
#     plt.tight_layout()
#     plt.savefig(str(out_path), dpi=180)
#     plt.close()

# # ---------- main ----------
# def main():
#     frames = load_tracking()
#     meta = load_match_metadata() or {}

#     # sample frame pids for matching
#     sample_pids = sample_frame_player_ids(frames)
#     print("Sample player ids found in frames (first 10):", sample_pids[:10])

#     # read team player lists from metadata (try a few keys)
#     home_ids_raw = meta.get("home_players") or meta.get("home_team_player_ids") or meta.get("home_squad") or []
#     away_ids_raw = meta.get("away_players") or meta.get("away_team_player_ids") or meta.get("away_squad") or []
#     # normalize metadata ids to match frame pid types
#     home_ids = normalize_metadata_ids(home_ids_raw, sample_pids)
#     away_ids = normalize_metadata_ids(away_ids_raw, sample_pids)
#     print("Normalized home_ids (first 10):", home_ids[:10])
#     print("Normalized away_ids (first 10):", away_ids[:10])

#     # team names
#     home_name = meta.get("home_team") or meta.get("home_name") or meta.get("home") or "TeamA"
#     away_name = meta.get("away_team") or meta.get("away_name") or meta.get("away") or "TeamB"

#     # compute averaged grids
#     print("Computing averaged control grids...")
#     teamA_grid, teamB_grid, xs, ys = compute_full_grids(frames, home_ids, away_ids, GRID_X, GRID_Y)

#     zones = get_zone_bounds()
#     zone_names = ["defensive", "midfield", "attacking"]
#     summary = {"teamA": {"name": str(home_name), "zones": {}}, "teamB": {"name": str(away_name), "zones": {}}}

#     # team A
#     for (x_left, x_right), zname in zip(zones, zone_names):
#         cx, cy, tot = masked_centroid(teamA_grid, xs, ys, x_left, x_right)
#         stats, boxes = compute_zone_cell_stats(teamA_grid, xs, ys, x_left, x_right)
#         focal_cell = find_cell_for_point(cx, cy, boxes) if cx is not None else None
#         outfile = OUT / f"teamA_{zname}.png"
#         plot_zone_cells(teamA_grid, xs, ys, x_left, x_right, boxes, (cx, cy), focal_cell, home_name, zname, outfile, stats)
#         summary["teamA"]["zones"][zname] = {
#             "x_left": x_left, "x_right": x_right,
#             "focal_x": None if cx is None else round(cx, 3),
#             "focal_y": None if cy is None else round(cy, 3),
#             "focal_cell": focal_cell,
#             "zone_total_control": round(tot, 8),
#             "cell_means": [[float(round(float(stats[r,c]), 10)) for c in range(3)] for r in range(3)]
#         }

#     # team B
#     for (x_left, x_right), zname in zip(zones, zone_names):
#         cx, cy, tot = masked_centroid(teamB_grid, xs, ys, x_left, x_right)
#         stats, boxes = compute_zone_cell_stats(teamB_grid, xs, ys, x_left, x_right)
#         focal_cell = find_cell_for_point(cx, cy, boxes) if cx is not None else None
#         outfile = OUT / f"teamB_{zname}.png"
#         plot_zone_cells(teamB_grid, xs, ys, x_left, x_right, boxes, (cx, cy), focal_cell, away_name, zname, outfile, stats)
#         summary["teamB"]["zones"][zname] = {
#             "x_left": x_left, "x_right": x_right,
#             "focal_x": None if cx is None else round(cx, 3),
#             "focal_y": None if cy is None else round(cy, 3),
#             "focal_cell": focal_cell,
#             "zone_total_control": round(tot, 8),
#             "cell_means": [[float(round(float(stats[r,c]), 10)) for c in range(3)] for r in range(3)]
#         }

#     # save summary
#     with open(OUT / "zone_focals_cells_summary.json", "w", encoding="utf-8") as fh:
#         json.dump(summary, fh, indent=2)

#     # friendly prints
#     def human_for(team_key):
#         t = summary[team_key]
#         name = t["name"]
#         lines = []
#         cell_human = {
#             1: "top-left", 2: "top-center", 3: "top-right",
#             4: "middle-left",5: "center",6: "middle-right",
#             7: "bottom-left",8: "bottom-center",9: "bottom-right"
#         }
#         for z in zone_names:
#             zinfo = t["zones"][z]
#             fx = zinfo["focal_x"]; fy = zinfo["focal_y"]; cell = zinfo["focal_cell"]
#             if fx is None:
#                 lines.append(f"{name} {z}: no focal point (zero control mass).")
#             else:
#                 lines.append(f"{name} {z} focal ≈ x={fx} m, y={fy} m → {cell_human.get(cell,'cell '+str(cell))} (cell {cell}).")
#         return " ".join(lines)

#     print("\nPlain English focal-by-cell summaries:")
#     print(" -", human_for("teamA"))
#     print(" -", human_for("teamB"))

#     print("\nSaved outputs in", OUT.resolve())
#     for p in sorted(OUT.iterdir()):
#         print(" -", p.name)


# if __name__ == "__main__":
#     main()


# model3.py
"""
model3.py

Integrated classification rules for Attacking / Midfield / Defensive thirds
and Match-wide style labels.

This file tries to import features from a local module that produces the
necessary data (it first attempts model0_load_data, then model0). The
expected interface is either:

 - def get_match_features() -> dict
   or any of these function names: get_match_features, get_features,
   extract_features, load_match, load_features

 - Or a module-level dict named FEATURES or features

The returned dict must contain keys:
    attacking_focal: int         # 0..8 (3x3)
    attacking_focal_value: float # 0..1
    attacking_presence: float    # 0..100

    midfield_focal: int          # 0..8
    midfield_presence: float     # 0..100

    defensive_focal: int         # 0..8
    defensive_presence: float    # 0..100
"""

from typing import Dict, Any, Tuple, List
import importlib
import inspect
import sys

# ---------- Helpers ----------
CELL_NAMES = {
    0: "top-left",
    1: "top-middle",
    2: "top-right",
    3: "center-left",
    4: "center-middle",
    5: "center-right",
    6: "bottom-left",
    7: "bottom-middle",
    8: "bottom-right",
}


# --- Attacking third classification ---
# def classify_attacking_third(attacking_focal: int, attacking_focal_value: float, attacking_presence: float) -> Tuple[List[str], List[str]]:
#     labels: List[str] = []
#     reasons: List[str] = []

#     # Central Creativity
#     if attacking_focal == 4 or attacking_focal_value > 0.25:
#         labels.append("Central Creativity")
#         reasons.append(f"Focal cell {CELL_NAMES.get(attacking_focal)} or central focal value {attacking_focal_value:.2f} > 0.25")

#     # Right Half-Space Creativity
#     if attacking_focal == 5:
#         labels.append("Right Half-Space Creativity")
#         reasons.append("Focal point in right-middle attacking cell → right-sided overloads / cutbacks")

#     # Left Half-Space Creativity
#     if attacking_focal == 3:
#         labels.append("Left Half-Space Creativity")
#         reasons.append("Focal point in left-middle attacking cell → left-sided overloads / underlaps")

#     # Wing Play (Wide Attacking Cells)
#     if attacking_focal in (0, 2, 6, 8):
#         labels.append("Wing Play")
#         reasons.append(f"Focal point in wing cell {CELL_NAMES.get(attacking_focal)} → crossing and wide progression")

#     # Shallow Attacking Presence (closer to midfield than box)
#     if attacking_focal in (6, 7, 8):
#         labels.append("Shallow Attacking Presence")
#         reasons.append("Focal point closer to midfield (bottom row of attacking grid) → reached attacking third but lacked penetration")

#     if not labels:
#         labels.append("No dominant attacking pattern")
#         reasons.append(f"Attacking focal {CELL_NAMES.get(attacking_focal)} with value {attacking_focal_value:.2f} and presence {attacking_presence:.1f}%")

#     return labels, reasons


# # --- Midfield third classification ---
# def classify_midfield_third(midfield_focal: int, midfield_presence: float, attacking_presence: float = None, defensive_presence: float = None) -> Tuple[List[str], List[str]]:
#     labels: List[str] = []
#     reasons: List[str] = []

#     # Controlled Central Build-Up (Positional Play)
#     if midfield_focal == 4 and midfield_presence > 30:
#         labels.append("Controlled Central Build-Up")
#         reasons.append(f"Focal at center-middle (cell 4) and midfield_presence {midfield_presence:.1f}% > 30%")

#     # Wing-Midfield Progression
#     if midfield_focal in (3, 5):
#         labels.append("Wing-Midfield Progression")
#         reasons.append(f"Midfield focal in {CELL_NAMES.get(midfield_focal)} → play routed via wide lanes")

#     # Midfield Bypass / Kick & Rush
#     if (midfield_presence is not None and midfield_presence < 20
#             and attacking_presence is not None and attacking_presence > 30
#             and defensive_presence is not None and defensive_presence > 30):
#         labels.append("Midfield Bypass / Kick & Rush")
#         reasons.append(f"midfield_presence {midfield_presence:.1f}% < 20% and attacking_presence {attacking_presence:.1f}% > 30% and defensive_presence {defensive_presence:.1f}% > 30% → direct vertical play")

#     # Overloaded Midfield Block
#     if midfield_focal in (0, 1, 2) and (attacking_presence is not None and attacking_presence < 20) and (defensive_presence is not None and defensive_presence < 20):
#         labels.append("Overloaded Midfield Block")
#         reasons.append("Midfield focal high with low attacking & defensive presence → team stuck in midfield")

#     if not labels:
#         labels.append("No dominant midfield pattern")
#         reasons.append(f"Midfield focal {CELL_NAMES.get(midfield_focal)} with midfield_presence {midfield_presence:.1f}%")

#     return labels, reasons


# # --- Defensive third classification ---
# def classify_defensive_third(defensive_focal: int, defensive_presence: float) -> Tuple[List[str], List[str]]:
#     labels: List[str] = []
#     reasons: List[str] = []

#     # High Defensive Line / High Press
#     if defensive_focal == 1:
#         labels.append("High Defensive Line / High Press")
#         reasons.append("Focal at top-middle defensive cell → aggressive high line / high press")

#     # Mid Defensive Shape
#     if defensive_focal == 4:
#         labels.append("Mid Defensive Shape")
#         reasons.append("Focal at central defensive cell → mid-block / controlled compact defence")

#     # Low Block / Deep Defence
#     if defensive_focal == 7:
#         labels.append("Low Block / Deep Defence")
#         reasons.append("Focal at bottom-middle defensive cell (near box) → low block / deep defending")

#     # Solid Defence / Opponent Kept Out
#     if defensive_presence < 20:
#         labels.append("Solid Defence / Opponent Kept Out")
#         reasons.append(f"Defensive presence {defensive_presence:.1f}% < 20% → opponent rarely entered defensive third")

#     # Wide Defensive Engagement
#     if defensive_focal in (0, 2, 6, 8):
#         labels.append("Wide Defensive Engagement")
#         reasons.append(f"Focal at wide defensive cell {CELL_NAMES.get(defensive_focal)} → opponent forced wide / channeling play")

#     if not labels:
#         labels.append("No dominant defensive pattern")
#         reasons.append(f"Defensive focal {CELL_NAMES.get(defensive_focal)} with defensive_presence {defensive_presence:.1f}%")

#     return labels, reasons


# # --- Full match-wide style labels ---
# def classify_match_style(midfield_presence: float, attacking_presence: float, defensive_presence: float, attacking_focal: int, midfield_focal: int, defensive_focal: int) -> Tuple[List[str], List[str]]:
#     labels: List[str] = []
#     reasons: List[str] = []

#     # 1. Kick & Rush / Direct Vertical Play
#     if midfield_presence < 20 and attacking_presence > 30 and 20 <= defensive_presence <= 60:
#         labels.append("Kick & Rush / Direct Vertical Play")
#         reasons.append("midfield low + attack high + defence moderate → direct transitions")

#     # 2. Wing Play
#     if attacking_focal in (0, 2, 6, 8) and midfield_focal in (0, 2, 6, 8):
#         labels.append("Wing Play (Match-Wide)")
#         reasons.append("attacking focal wide + midfield focal wide → strong wing-heavy attack")

#     # 3. Positional Play / Controlled Build-Up
#     if midfield_focal == 4 and attacking_focal == 4:
#         labels.append("Positional Play / Controlled Build-Up")
#         reasons.append("midfield central + attacking central → classic controlled buildup")

#     # 4. Low Block Defence
#     if defensive_focal == 7:
#         labels.append("Low Block Defence (Match-Wide)")
#         reasons.append("defensive focal deep → team absorbs pressure")

#     # 5. High Press
#     if defensive_focal == 1:
#         labels.append("High Press (Match-Wide)")
#         reasons.append("defensive focal high → aggressive high line")

#     # 6. Creative Half-Space Team
#     if attacking_focal in (3, 5):
#         labels.append("Creative Half-Space Team")
#         reasons.append("attacking focal in half-space → high-quality box entries")

#     if not labels:
#         labels.append("No clear match-wide style")
#         reasons.append("No combination of thirds matched the style heuristics")

#     return labels, reasons


# # --- Orchestrator ---
# def classify_all(features: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Classify a match given features dict. Returns structured labels & explanations.
#     Required keys are documented at top of file.
#     """
#     required = [
#         "attacking_focal", "attacking_focal_value", "attacking_presence",
#         "midfield_focal", "midfield_presence",
#         "defensive_focal", "defensive_presence",
#     ]
#     for k in required:
#         if k not in features:
#             raise ValueError(f"Missing required feature: {k}")

#     af = int(features["attacking_focal"])
#     afv = float(features["attacking_focal_value"])
#     ap = float(features["attacking_presence"])

#     mf = int(features["midfield_focal"])
#     mp = float(features["midfield_presence"])

#     df = int(features["defensive_focal"])
#     dp = float(features["defensive_presence"])

#     attacking_labels, attacking_reasons = classify_attacking_third(af, afv, ap)
#     midfield_labels, midfield_reasons = classify_midfield_third(mf, mp, ap, dp)
#     defensive_labels, defensive_reasons = classify_defensive_third(df, dp)
#     match_labels, match_reasons = classify_match_style(mp, ap, dp, af, mf, df)

#     return {
#         "attacking": {
#             "labels": attacking_labels,
#             "reasons": attacking_reasons,
#         },
#         "midfield": {
#             "labels": midfield_labels,
#             "reasons": midfield_reasons,
#         },
#         "defensive": {
#             "labels": defensive_labels,
#             "reasons": defensive_reasons,
#         },
#         "match_style": {
#             "labels": match_labels,
#             "reasons": match_reasons,
#         }
#     }


# # --- model0 integration helper ---
# def load_features_from_local_providers(module_names=None) -> Dict[str, Any]:
#     """
#     Try to import a module that provides match features. By default tries:
#       1. model0_load_data
#       2. model0

#     Accepts either a get_match_features()-style function (or several alternative names),
#     or a module-level dict FEATURES / features.
#     """
#     if module_names is None:
#         module_names = ["model0_load_data", "model0"]

#     func_names = [
#         "get_match_features", "get_features", "extract_features", "load_match", "load_features"
#     ]
#     var_names = ["FEATURES", "features", "MATCH_FEATURES", "match_features"]

#     last_err = None
#     for module_name in module_names:
#         try:
#             m = importlib.import_module(module_name)
#         except Exception as e:
#             last_err = e
#             continue

#         # Try functions
#         for name in func_names:
#             if hasattr(m, name) and callable(getattr(m, name)):
#                 fn = getattr(m, name)
#                 try:
#                     sig = inspect.signature(fn)
#                     # call without args, even if signature accepts them (common pattern)
#                     return fn()
#                 except TypeError:
#                     # If signature absolutely requires args, we still attempt a call and let module raise
#                     return fn()
#                 except Exception as e:
#                     last_err = e
#                     # try next provider
#                     continue

#         # Try module-level variables (dicts)
#         for var in var_names:
#             if hasattr(m, var):
#                 val = getattr(m, var)
#                 if isinstance(val, dict):
#                     return val

#         # If we got here, module imported but didn't expose expected items; record and move on
#         last_err = AttributeError(f"Module '{module_name}' imported but no get_match_features() or FEATURES dict found.")

#     # If we exhausted all candidates, raise a helpful error
#     raise ImportError(
#         "No local feature provider found. Tried modules: "
#         f"{', '.join(module_names)}. Last error: {last_err}"
#     )


# # --- __main__ harness ---
# if __name__ == "__main__":
#     try:
#         features = load_features_from_local_providers(["model0_load_data", "model0"])
#         print("Loaded features from local provider:")
#         print(features)
#     except Exception as e:
#         print("WARNING: could not auto-load features from local providers:", e, file=sys.stderr)
#         print("Falling back to example features. To integrate, provide get_match_features() or FEATURES dict in model0_load_data.py or model0.py", file=sys.stderr)
#         features = {
#             "attacking_focal": 4,
#             "attacking_focal_value": 0.35,
#             "attacking_presence": 38.0,
#             "midfield_focal": 4,
#             "midfield_presence": 34.0,
#             "defensive_focal": 1,
#             "defensive_presence": 28.0,
#         }

#     result = classify_all(features)
#     import pprint
#     pprint.pprint(result)
# def classify_all(features):
#     """
#     Hybrid rule-based classifier (relative + soft scoring + tiered thresholds).
#     Expects feature dict keys:
#       attacking_focal (int 0..8), attacking_focal_value (0..1), attacking_presence (%)
#       midfield_focal (int), midfield_presence (%)
#       defensive_focal (int), defensive_presence (%)
#     Returns a dict with keys: attacking, midfield, defensive, match_style
#     Each is a dict with 'labels' (list) and 'reasons' (list).
#     """
#     # defensive programming: safe reads
#     af_idx = int(features.get("attacking_focal", 4))
#     af_val = float(features.get("attacking_focal_value", 0.0))
#     ap = float(features.get("attacking_presence", 0.0))

#     mf_idx = int(features.get("midfield_focal", 4))
#     mp = float(features.get("midfield_presence", 0.0))

#     df_idx = int(features.get("defensive_focal", 4))
#     dp = float(features.get("defensive_presence", 0.0))

#     # Helper builders
#     def mk(): return {"labels": [], "reasons": []}
#     out_att = mk(); out_mid = mk(); out_def = mk(); out_style = mk()

#     # --- Soft scoring ---
#     # Score amplifies focal value by presence (so focal_value=0.7 with 10% presence < same focal with 40% presence)
#     attacking_score = af_val * (1.0 + ap / 100.0)
#     midfield_score = (mp / 100.0)  # already a % -> use 0..1 scale for scoring
#     defensive_score = (dp / 100.0)

#     # --- Attacking labels (tiered + half-space) ---
#     if attacking_score >= 0.65:
#         out_att["labels"].append("Central Creativity (Strong)")
#         out_att["reasons"].append(f"attacking_score {attacking_score:.2f} >= 0.65 (dominant)")
#     elif attacking_score >= 0.35:
#         out_att["labels"].append("Central Creativity")
#         out_att["reasons"].append(f"attacking_score {attacking_score:.2f} >= 0.35")
#     elif attacking_score >= 0.18:
#         out_att["labels"].append("Shallow Attacking Presence")
#         out_att["reasons"].append(f"attacking_score {attacking_score:.2f} < 0.35 and >= 0.18 (shallow)")
#     else:
#         out_att["labels"].append("Distributed / Wide Attack")
#         out_att["reasons"].append(f"attacking_score {attacking_score:.2f} < 0.18")

#     # Interpret focal index for half-space / wing checks
#     # Index layout assumed: 0..8 left→right, top→bottom, center-middle = 4
#     # center-left = 3, center =4, center-right=5, wings might be 2 and 6 (depending on your mapping)
#     if af_idx in (3,):  # center-left
#         out_att["labels"].append("Left Half-Space Creativity")
#         out_att["reasons"].append("Focal cell center-left (index 3)")
#     elif af_idx in (5,):  # center-right
#         out_att["labels"].append("Right Half-Space Creativity")
#         out_att["reasons"].append("Focal cell center-right (index 5)")
#     elif af_idx in (0,2,6,8):  # corners/wings (approx)
#         out_att["labels"].append("Wing Play")
#         out_att["reasons"].append("Focal cell on wide wing cell (wing play)")

#     # Also expose raw numeric summary in reasons (helpful)
#     out_att["reasons"].append(f"attacking_focal_value={af_val:.2f}, attacking_presence={ap:.1f}%")

#     # --- Midfield labels ---
#     # Use relative thresholds but slightly stricter than before for clarity
#     if mp >= 40.0:
#         out_mid["labels"].append("Controlled Central Build-Up (Strong)")
#         out_mid["reasons"].append(f"midfield_presence {mp:.1f}% >= 40%")
#     elif mp >= 25.0:
#         out_mid["labels"].append("Controlled Central Build-Up")
#         out_mid["reasons"].append(f"midfield_presence {mp:.1f}% >= 25%")
#     elif mp >= 12.0:
#         out_mid["labels"].append("Wing-Midfield Progression")
#         out_mid["reasons"].append(f"midfield_presence {mp:.1f}% between 12% and 25% -> routed via wide lanes")
#     else:
#         out_mid["labels"].append("No dominant midfield pattern")
#         out_mid["reasons"].append(f"midfield_presence {mp:.1f}% < 12%")

#     # add focal cell note
#     if mf_idx == 4:
#         out_mid["reasons"].append("midfield focal at center")
#     elif mf_idx in (3,5):
#         out_mid["reasons"].append("midfield focal in wide mid cell")

#     # --- Defensive labels ---
#     # Use lower threshold for 'solid defence' and add mid-block
#     if dp < 12.0:
#         out_def["labels"].append("Solid Defence / Opponent Kept Out")
#         out_def["reasons"].append(f"defensive_presence {dp:.1f}% < 12%")
#     elif dp < 28.0:
#         out_def["labels"].append("Mid Defensive Shape")
#         out_def["reasons"].append(f"defensive_presence {dp:.1f}% < 28%")
#     else:
#         out_def["labels"].append("Low Block / Deep Defence")
#         out_def["reasons"].append(f"defensive_presence {dp:.1f}% >= 28%")

#     out_def["reasons"].append(f"defensive_presence_raw={dp:.1f}%")

#     # --- Match-style composition (combine signals) ---
#     # Positional / controlled build-up
#     if ("Controlled Central Build-Up" in out_mid["labels"] or "Controlled Central Build-Up (Strong)" in out_mid["labels"]) and any("Central Creativity" in s for s in out_att["labels"]):
#         out_style["labels"].append("Positional Play / Controlled Build-Up")
#         out_style["reasons"].append("midfield central + attacking central")

#     # Kick & Rush / Direct vertical
#     if mp < 12.0 and ap > 30.0 and dp > 25.0:
#         out_style["labels"].append("Kick & Rush / Direct Vertical Play")
#         out_style["reasons"].append("midfield low + attack high + defence moderate -> direct transitions")

#     # Half-space creative team
#     if any("Half-Space" in s for s in out_att["labels"]) and any("Controlled Central" in s or "Central Creativity" in s for s in out_att["labels"]):
#         out_style["labels"].append("Creative Half-Space Team")
#         out_style["reasons"].append("attacking focal in half-space -> high-quality box entries")

#     # Fallback style if none appended
#     if not out_style["labels"]:
#         out_style["labels"].append("Balanced / Mixed Style")
#         out_style["reasons"].append("no dominant combined pattern detected")

#     return {
#         "attacking": out_att,
#         "midfield": out_mid,
#         "defensive": out_def,
#         "match_style": out_style
#     }
cd /Users/nishoosingh/Downloads/model

# overwrite the adaptive script with a fixed, robust version that detects attacking_focal_value etc.
cat > model3_adaptive_thresholds.py <<'PY'
#!/usr/bin/env python3
"""
model3_adaptive_thresholds.py (fixed)
Robust re-classifier that supports attacking_focal_value / attacking_presence keys.
Writes model3_results_adaptive.json and .csv
"""
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

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
    print("\\nReclassification counts (summary):")
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
PY

# run the fixed adaptive re-classifier (adjust min_frac if you want, e.g. --min_frac 0.166)
/Users/nishoosingh/Downloads/model/venv/bin/python /Users/nishoosingh/Downloads/model/model3_adaptive_thresholds.py
