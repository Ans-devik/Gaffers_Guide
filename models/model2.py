
"""
Minimal-patch model2.py (team split)
- Keeps original computation/plots exactly
- Adds only the necessary changes:
  * deterministic OUTPUT under repo root
  * fallback to opendata first match when no/corrupt input
  * sanitize folder names
  * model_main wrapper
  * copy PNGs into static/plots for Flask
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------
# FORCE outputs into repo root (minimal change)
# ---------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # project root (/Users/.../Downloads/model)
OUTPUT = REPO_ROOT / "model2_output_team"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Try to import original loader (if present)
# ---------------------------
try:
    from model0_load_data import load_tracking as _orig_load_tracking, load_match_metadata as _orig_load_meta
    _HAS_ORIG_LOADER = True
except Exception:
    _HAS_ORIG_LOADER = False

# Config
FPS = 25
HIGH_INTENSITY_THRESHOLD = 5.0

# ---------------------------
# small helpers (sanitize)
# ---------------------------
import re
def sanitize(name: str) -> str:
    name = str(name)
    name = re.sub(r"[{}\[\]\(\)\:\'\",]", "", name)
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name.strip("_") or "team"

# ---------------------------
# Robust JSONL reader + fallback loader
# ---------------------------
def _read_jsonl_lines_from_folder(match_folder: Path) -> List[dict]:
    candidates = list(match_folder.glob("*tracking_extrapolated.jsonl")) + list(match_folder.glob("*tracking*.jsonl")) + list(match_folder.glob("*tracking*.json"))
    if not candidates:
        raise FileNotFoundError(f"No tracking jsonl found in {match_folder}")
    candidates_sorted = sorted(candidates, key=lambda p: (0 if "extrapolated" in p.name.lower() else 1, p.name))
    chosen = candidates_sorted[0]
    frames = []
    with chosen.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except Exception:
                # skip malformed lines
                continue
    print(f"[fallback-jsonl] loaded {len(frames)} frames from {chosen.name}")
    return frames

def _try_opendata_first_match():
    opendata_matches = REPO_ROOT / "opendata" / "data" / "matches"
    if opendata_matches.exists() and opendata_matches.is_dir():
        match_dirs = sorted([d for d in opendata_matches.iterdir() if d.is_dir()])
        if match_dirs:
            return match_dirs[0]
    return None

def _load_frames_with_fallback(input_path: Optional[str] = None) -> List[dict]:
    # try original loader first (if available)
    frames = None
    if _HAS_ORIG_LOADER:
        try:
            if input_path:
                frames = _orig_load_tracking(input_path)
            else:
                frames = _orig_load_tracking()
            if frames is None:
                print("[loader] original loader returned None")
        except Exception as e:
            print(f"[loader] original load_tracking raised: {e}")
            frames = None

    # if not list, figure folder to read and fallback to JSONL
    if not isinstance(frames, list):
        if input_path:
            p = Path(input_path)
            folder = p if p.is_dir() else p.parent
        else:
            folder = Path.cwd()

        # first try folder; if not found, try opendata first match
        try:
            frames = _read_jsonl_lines_from_folder(folder)
        except FileNotFoundError:
            fallback = _try_opendata_first_match()
            if fallback is not None:
                print(f"[fallback] No tracking in {folder}, trying opendata match {fallback.name}")
                frames = _read_jsonl_lines_from_folder(fallback)
            else:
                raise
    return frames

# ---------------------------
# Tolerant extraction helpers (kept from original)
# ---------------------------
def extract_player_fields(p: dict) -> Tuple[Optional[Any], Optional[float], Optional[float], Optional[str]]:
    pid = None
    for k in ("player_id", "id", "pid", "playerId"):
        if k in p and p[k] is not None:
            pid = p[k]; break

    x = None; y = None
    for k in ("x","X","pos_x","posX","x_world","x_pos"):
        if k in p and p[k] is not None:
            x = p[k]; break
    for k in ("y","Y","pos_y","posY","y_world","y_pos"):
        if k in p and p[k] is not None:
            y = p[k]; break

    if (x is None or y is None) and isinstance(p.get("position"), dict):
        pos = p["position"]
        if x is None:
            for k in ("x","pos_x","x_world"):
                if k in pos and pos[k] is not None:
                    x = pos[k]; break
        if y is None:
            for k in ("y","pos_y","y_world"):
                if k in pos and pos[k] is not None:
                    y = pos[k]; break

    team_label = None
    for k in ("team","side","team_id","teamName"):
        if k in p and p[k] is not None:
            team_label = p[k]; break

    try:
        x = float(x) if x is not None else None
    except Exception:
        x = None
    try:
        y = float(y) if y is not None else None
    except Exception:
        y = None

    return pid, x, y, team_label

def map_meta_player_names(meta: dict) -> Dict[Any, str]:
    name_map = {}
    players_list = meta.get("players") or meta.get("team_players") or None
    if isinstance(players_list, list):
        for p in players_list:
            pid = p.get("player_id") or p.get("id") or p.get("pid")
            name = p.get("name") or p.get("player_name") or p.get("display_name")
            if pid is None and (p.get("first_name") or p.get("last_name")):
                name = " ".join(filter(None, [p.get("first_name"), p.get("last_name")]))
            if pid is not None and name:
                name_map[pid] = name
    for team_key in ("home", "home_team", "home_team_players", "home_team_roster"):
        team_entry = meta.get(team_key)
        if isinstance(team_entry, dict):
            tplayers = team_entry.get("players") if isinstance(team_entry.get("players"), list) else team_entry.get("squad")
            if isinstance(tplayers, list):
                for p in tplayers:
                    pid = p.get("player_id") or p.get("id") or p.get("pid")
                    name = p.get("name") or p.get("player_name")
                    if pid is not None and name:
                        name_map[pid] = name
    return name_map

def compute_distances_and_sprints(frames: List[dict], fps: int) -> Tuple[Dict[Any,float], Dict[Any,int], Dict[Any,str]]:
    dt = 1.0 / fps
    last_pos = {}
    distances = {}
    sprints = {}
    pid_team = {}

    for frame in frames:
        pd = frame.get("player_data", []) or frame.get("players", []) or []
        for p in pd:
            pid, x, y, team_label = extract_player_fields(p)
            if pid is None or x is None or y is None:
                continue
            try:
                pid_key = int(pid)
            except Exception:
                pid_key = pid

            if team_label is not None and pid_key not in pid_team:
                pid_team[pid_key] = team_label

            if pid_key in last_pos:
                dx = x - last_pos[pid_key][0]
                dy = y - last_pos[pid_key][1]
                dist = np.hypot(dx, dy)
                distances[pid_key] = distances.get(pid_key, 0.0) + float(dist)
                speed = dist / dt if dt > 0 else 0.0
                if speed >= HIGH_INTENSITY_THRESHOLD:
                    sprints[pid_key] = sprints.get(pid_key, 0) + 1
            else:
                distances.setdefault(pid_key, 0.0)
                sprints.setdefault(pid_key, 0)

            last_pos[pid_key] = (x, y)

    return distances, sprints, pid_team

def plot_bar(values_map: Dict[Any,float], titlestr: str, ylabel: str, out_path: Path,
             id_to_label: Dict[Any,str]=None, top_n: Optional[int]=None, color=None):
    if not values_map:
        print("No data to plot:", titlestr)
        return
    items = sorted(values_map.items(), key=lambda kv: kv[1], reverse=True)
    if top_n:
        items = items[:top_n]
    ids = [kv[0] for kv in items]
    vals = [kv[1] for kv in items]
    labels = [id_to_label.get(i, str(i)) if id_to_label else str(i) for i in ids]

    plt.figure(figsize=(12,5))
    plt.bar(range(len(vals)), vals, alpha=0.95)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    plt.title(titlestr)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()

# Quarter-by-quarter helpers (optional small add)
import math as _math
def _compute_per_quarter_metrics(frames: List[dict], fps: int):
    total_frames = len(frames)
    if total_frames == 0:
        return [], {}, {}
    q_size = _math.ceil(total_frames / 4)
    per_player_last = {}
    dist_q = {}
    sprint_frames_q = {}
    dt = 1.0 / fps if fps > 0 else 1.0
    for i, frame in enumerate(frames):
        qidx = min(3, i // q_size) if q_size > 0 else 0
        players = frame.get("player_data", []) or frame.get("players", []) or []
        for p in players:
            pid, x, y, _ = extract_player_fields(p)
            if pid is None or x is None or y is None:
                continue
            try:
                pid_key = int(pid)
            except Exception:
                pid_key = pid
            if pid_key not in dist_q:
                dist_q[pid_key] = [0.0, 0.0, 0.0, 0.0]
                sprint_frames_q[pid_key] = [0, 0, 0, 0]
                per_player_last[pid_key] = None
            last = per_player_last.get(pid_key)
            if last is not None:
                dx = x - last[0]
                dy = y - last[1]
                d = _math.hypot(dx, dy)
                dist_q[pid_key][qidx] += float(d)
                speed = d / dt if dt > 0 else 0.0
                if speed >= HIGH_INTENSITY_THRESHOLD:
                    sprint_frames_q[pid_key][qidx] += 1
            per_player_last[pid_key] = (x, y)
    sprint_seconds_q = {pid: [fc * (1.0 / fps) for fc in frames_counts] for pid, frames_counts in sprint_frames_q.items()}
    player_ids = sorted(set(list(dist_q.keys()) + list(sprint_seconds_q.keys())))
    return player_ids, dist_q, sprint_seconds_q

def _plot_lines_by_quarter(player_ids, metric_q: Dict, title: str, ylabel: str, out_path: Path,
                           top_n: int = 8):
    if not player_ids:
        print("[quarter-plots] no players to plot:", title)
        return
    totals = {pid: sum(metric_q.get(pid, [0,0,0,0])) for pid in player_ids}
    top_players = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    top_ids = [kv[0] for kv in top_players]
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    plt.figure(figsize=(13,7))
    for pid in player_ids:
        vals = metric_q.get(pid, [0,0,0,0])
        plt.plot(quarters, vals, color="0.8", linewidth=1.2, alpha=0.6, zorder=1)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, pid in enumerate(top_ids):
        vals = metric_q.get(pid, [0,0,0,0])
        col = colors[i % len(colors)]
        plt.plot(quarters, vals, label=str(pid), color=col, linewidth=2.4, zorder=3 + i)
        plt.scatter(quarters, vals, color=col, s=25, zorder=4 + i)
    plt.title(title)
    plt.xlabel("Match Quarter")
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6)
    plt.legend(title=f"Top players by total", loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()

def generate_quarter_plots(valid_frames: List[dict], fps: int, out_dir: Path, top_n: int = 8):
    player_ids, dist_q, sprint_q = _compute_per_quarter_metrics(valid_frames, fps)
    _plot_lines_by_quarter(player_ids, sprint_q,
                           "Sprint Time (s) by Quarter — one line per player (top {} highlighted)".format(top_n),
                           "Sprint Time (s)",
                           out_dir / "sprints_by_quarter_lines.png",
                           top_n=top_n)
    _plot_lines_by_quarter(player_ids, dist_q,
                           "Distance (m) by Quarter — one line per player (top {} highlighted)".format(top_n),
                           "Distance (m)",
                           out_dir / "distance_by_quarter_lines.png",
                           top_n=top_n)
    print(f"[quarter-plots] Saved quarter-by-quarter line plots to {out_dir.resolve()}")

# ---------------------------
# Main runner (kept from original, minimal edits)
# ---------------------------
def run(input_path: Optional[str] = None, n_minutes: Optional[float] = None):
    frames = _load_frames_with_fallback(input_path)
    meta = {}
    if _HAS_ORIG_LOADER:
        try:
            meta = _orig_load_meta(input_path) if input_path else _orig_load_meta()
        except Exception as e:
            print(f"[loader] load_match_metadata failed: {e}")
            meta = {}
    if not isinstance(meta, dict) or not meta:
        try:
            folder = Path(input_path) if input_path else Path.cwd()
            if folder.is_file():
                folder = folder.parent
            cand = next(folder.glob("*match*.json"), None)
            if cand:
                with cand.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
        except Exception:
            meta = {}

    # FPS detection
    fps_meta = None
    for k in ("fps","frame_rate","frameRate","sample_rate"):
        if meta.get(k):
            try:
                fps_meta = int(meta.get(k))
                break
            except Exception:
                pass
    fps_use = fps_meta or FPS
    print(f"[info] Using FPS = {fps_use}")

    if n_minutes is None:
        valid_frames = [f for f in frames if (f.get("player_data") or f.get("players"))]
    else:
        max_frames = int(n_minutes * 60 * fps_use)
        valid_frames = [f for f in frames if (f.get("player_data") or f.get("players"))][:max_frames]
    print(f"[info] Total frames loaded: {len(frames)} -> frames used for analysis: {len(valid_frames)}")
    if len(valid_frames) == 0:
        raise RuntimeError("No valid frames found. Check your tracking loader output.")

    distances, sprint_frames, pid_team_map_from_frames = compute_distances_and_sprints(valid_frames, fps_use)
    sprint_seconds = {pid: int(frames_count) * (1.0 / fps_use) for pid, frames_count in sprint_frames.items()}

    home_ids_raw = meta.get("home_players") or meta.get("home_team_player_ids") or meta.get("home_squad") or []
    away_ids_raw = meta.get("away_players") or meta.get("away_team_player_ids") or meta.get("away_squad") or []
    def try_int_list(lst):
        out = []
        for v in lst or []:
            try:
                out.append(int(v))
            except Exception:
                out.append(v)
        return out
    home_ids = try_int_list(home_ids_raw)
    away_ids = try_int_list(away_ids_raw)

    home_name = meta.get("home_team") or meta.get("home_name") or meta.get("home") or "TeamA"
    away_name = meta.get("away_team") or meta.get("away_name") or meta.get("away") or "TeamB"
    id_to_name = map_meta_player_names(meta)

    pid_to_team = {}
    if home_ids:
        for pid in home_ids:
            pid_to_team[pid] = "A"
    if away_ids:
        for pid in away_ids:
            pid_to_team[pid] = "B"
    for pid, tlabel in pid_team_map_from_frames.items():
        if pid in pid_to_team:
            continue
        if tlabel in ("home", "Home", 1, "1", "H", "h"):
            pid_to_team[pid] = "A"
        elif tlabel in ("away", "Away", 0, "0", "A", "a"):
            pid_to_team[pid] = "B"
    all_pids = sorted(set(list(distances.keys()) + list(sprint_seconds.keys())))
    if not any(v == "A" for v in pid_to_team.values()) and not any(v == "B" for v in pid_to_team.values()):
        half = len(all_pids)//2
        for i, pid in enumerate(all_pids):
            pid_to_team[pid] = "A" if i < half else "B"

    team_metrics = {
        "A": {"name": str(home_name), "distances": {}, "sprints_seconds": {}, "ids": []},
        "B": {"name": str(away_name), "distances": {}, "sprints_seconds": {}, "ids": []}
    }
    for pid in all_pids:
        team = pid_to_team.get(pid, "A")
        dist_m = distances.get(pid, 0.0)
        sprint_s = sprint_seconds.get(pid, 0.0)
        team_metrics[team]["distances"][pid] = dist_m / 1000.0
        team_metrics[team]["sprints_seconds"][pid] = sprint_s
        team_metrics[team]["ids"].append(pid)

    for team in ("A","B"):
        distances_km = team_metrics[team]["distances"]
        sprints_s = team_metrics[team]["sprints_seconds"]
        fatigue_map = {}
        for pid in distances_km:
            d = distances_km.get(pid, 0.0)
            sp = sprints_s.get(pid, 0.0)
            score = d + (sp * 0.02)
            fatigue_map[pid] = score
        team_metrics[team]["fatigue_score"] = fatigue_map

    # Save per-team plots & JSON summary
    for team in ("A","B"):
        tname = team_metrics[team]["name"]
        safe_tname = sanitize(tname)
        out_dir = OUTPUT / f"team_{team}_{safe_tname}"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_bar(team_metrics[team]["distances"],
                 f"Top players by Total Distance (km) — {tname}",
                 "Distance (km)",
                 out_dir / f"player_total_distance_{safe_tname}.png",
                 id_to_label={pid: id_to_name.get(pid, str(pid)) for pid in team_metrics[team]["distances"].keys()},
                 color="#ff9f9f")

        plot_bar(team_metrics[team]["sprints_seconds"],
                 f"Top players by Sprint Time (s) — {tname}",
                 "Sprint Time (s)",
                 out_dir / f"player_sprint_seconds_{safe_tname}.png",
                 id_to_label={pid: id_to_name.get(pid, str(pid)) for pid in team_metrics[team]["sprints_seconds"].keys()},
                 color="#9fc5ff")

        plot_bar(team_metrics[team]["fatigue_score"],
                 f"Top players by Fatigue Score (higher = more fatigued) — {tname}",
                 "Fatigue score (heuristic)",
                 out_dir / f"player_fatigue_{safe_tname}.png",
                 id_to_label={pid: id_to_name.get(pid, str(pid)) for pid in team_metrics[team]["fatigue_score"].keys()},
                 color="#ffd39f")

        summary = {
            "team_name": tname,
            "num_players_analyzed": len(team_metrics[team]["ids"]),
            "distances_km": {str(pid): team_metrics[team]["distances"][pid] for pid in team_metrics[team]["distances"]},
            "sprint_seconds": {str(pid): team_metrics[team]["sprints_seconds"][pid] for pid in team_metrics[team]["sprints_seconds"]},
            "fatigue_score": {str(pid): team_metrics[team]["fatigue_score"][pid] for pid in team_metrics[team]["fatigue_score"]},
            "id_to_name": {str(pid): id_to_name.get(pid, None) for pid in team_metrics[team]["ids"]}
        }
        with open(out_dir / f"summary_{safe_tname}.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        print(f"Saved outputs for team {tname} in {out_dir.resolve()}")
        # ---------------------------
    # Side-by-side comparison plots (one figure with Team A left, Team B right)
    # ---------------------------
    def plot_side_by_side(metric_key: str, title_root: str, ylabel: str, out_name: str, top_n: int = 12):
        """
        metric_key: one of 'distances', 'sprints_seconds', 'fatigue_score' (these are dicts per team)
        title_root: human title prefix
        out_name: filename to save (under OUTPUT)
        """
        a_map = team_metrics["A"][metric_key]
        b_map = team_metrics["B"][metric_key]
        # pick top_n per team
        def top_items(m):
            return sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        a_items = top_items(a_map)
        b_items = top_items(b_map)

        a_ids = [kv[0] for kv in a_items]
        a_vals = [kv[1] for kv in a_items]
        b_ids = [kv[0] for kv in b_items]
        b_vals = [kv[1] for kv in b_items]

        a_labels = [id_to_name.get(pid, str(pid)) for pid in a_ids]
        b_labels = [id_to_name.get(pid, str(pid)) for pid in b_ids]

        fig, axes = plt.subplots(1, 2, figsize=(18,6), sharey=True)
        # Team A (left) - keep default color
        axes[0].bar(range(len(a_vals)), a_vals, alpha=0.95)
        axes[0].set_xticks(range(len(a_vals)))
        axes[0].set_xticklabels(a_labels, rotation=45, ha="right")
        axes[0].set_title(f"{title_root} — {team_metrics['A']['name']}")
        axes[0].set_ylabel(ylabel)

        # Team B (right) - distinct color
        axes[1].bar(range(len(b_vals)), b_vals, alpha=0.95, color="#2a9df4")
        axes[1].set_xticks(range(len(b_vals)))
        axes[1].set_xticklabels(b_labels, rotation=45, ha="right")
        axes[1].set_title(f"{title_root} — {team_metrics['B']['name']}")

        plt.tight_layout()
        outp = OUTPUT / out_name
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(outp), dpi=150)
        plt.close()
        print(f"[compare] saved {outp.resolve()}")

    # create comparison figures (distance, sprint time, fatigue)
    try:
        plot_side_by_side("distances", "Top players by Total Distance (km)", "Distance (km)", f"compare_total_distance_{sanitize(home_name)}_vs_{sanitize(away_name)}.png", top_n=12)
        plot_side_by_side("sprints_seconds", "Top players by Sprint Time (s)", "Sprint Time (s)", f"compare_sprint_seconds_{sanitize(home_name)}_vs_{sanitize(away_name)}.png", top_n=12)
        plot_side_by_side("fatigue_score", "Top players by Fatigue Score (higher = more fatigued)", "Fatigue score (heuristic)", f"compare_fatigue_{sanitize(home_name)}_vs_{sanitize(away_name)}.png", top_n=12)
    except Exception as _e:
        print("[compare] failed to make comparison plots:", _e)


    # quarter plots
    try:
        generate_quarter_plots(valid_frames, fps_use, OUTPUT, top_n=8)
    except Exception:
        pass

    # copy PNGs to static/plots for Flask convenience
    try:
        STATIC = REPO_ROOT / "static" / "plots"
        STATIC.mkdir(parents=True, exist_ok=True)
        for p in OUTPUT.rglob("*.png"):
            target = STATIC / p.name
            target.write_bytes(p.read_bytes())
    except Exception:
        pass

    combined = {
        "fps_used": fps_use,
        "teams": {
            "A": {"name": team_metrics["A"]["name"], "num_players": len(team_metrics["A"]["ids"])},
            "B": {"name": team_metrics["B"]["name"], "num_players": len(team_metrics["B"]["ids"])}
        }
    }
    with open(OUTPUT / "combined_summary.json", "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2)

    print("Model2 team-split complete. Outputs are in:", OUTPUT.resolve())
    return combined, team_metrics

# Minimal wrapper for Flask
def model_main(input_path: Optional[str]):
    """
    Runs the model and returns a JSON-serializable summary with plot paths (relative to static/plots).
    """
    run(input_path)
    plots = []
    for p in (REPO_ROOT / "static" / "plots").glob("*.png"):
        plots.append(str(Path("plots") / p.name))
    return {"plots": plots, "output_folder": str(OUTPUT)}

# CLI entrypoint
if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        run(arg, n_minutes=None)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
