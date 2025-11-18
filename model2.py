# model2_team_split.py
"""
Model 2 (team split)
- Computes total distance and high-intensity sprint time per player for the full match (n_minutes=None)
- Splits metrics by team (home vs away) using match metadata when available
- Saves per-team plots and JSON summaries
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Any

from model0_load_data import load_tracking, load_match_metadata

# Config
FPS = 25  # default if metadata doesn't provide fps
HIGH_INTENSITY_THRESHOLD = 5.0  # m/s threshold for "sprint"
OUTPUT = Path("model2_output_team")
OUTPUT.mkdir(parents=True, exist_ok=True)


# --- Helpers for tolerant extraction of id/x/y/team/name from frames & metadata ---
def extract_player_fields(p: dict) -> Tuple[Optional[Any], Optional[float], Optional[float], Optional[str]]:
    """
    Try multiple common keys to get player id, x, y and team label (if present in frame).
    Returns (pid, x, y, team_label)
    """
    # id candidates
    pid = None
    for k in ("player_id", "id", "pid", "playerId"):
        if k in p and p[k] is not None:
            pid = p[k]; break

    # position candidates
    x = None; y = None
    for k in ("x","X","pos_x","posX","x_world","x_pos"):
        if k in p and p[k] is not None:
            x = p[k]; break
    for k in ("y","Y","pos_y","posY","y_world","y_pos"):
        if k in p and p[k] is not None:
            y = p[k]; break
    # nested position
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

    # team label in frame
    team_label = None
    for k in ("team","side","team_id","teamName"):
        if k in p and p[k] is not None:
            team_label = p[k]; break

    # convert coords to floats if possible
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
    """
    If metadata contains a 'players' list or mapping, attempt to build id->name map.
    A variety of keys tried for robustness.
    """
    name_map = {}
    # many datasets include 'players' as a list of dicts
    players_list = meta.get("players") or meta.get("team_players") or None
    if isinstance(players_list, list):
        for p in players_list:
            # try id and name keys
            pid = p.get("player_id") or p.get("id") or p.get("pid")
            name = p.get("name") or p.get("player_name") or p.get("display_name")
            if pid is None and (p.get("first_name") or p.get("last_name")):
                name = " ".join(filter(None, [p.get("first_name"), p.get("last_name")]))
            if pid is not None and name:
                name_map[pid] = name
    # some metadata store per-team player lists under home/away team dicts
    # check nested metadata fields:
    for team_key in ("home", "home_team", "home_team_players", "home_team_roster"):
        team_entry = meta.get(team_key)
        if isinstance(team_entry, dict):
            # try team_entry.get("players")
            tplayers = team_entry.get("players") if isinstance(team_entry.get("players"), list) else team_entry.get("squad")
            if isinstance(tplayers, list):
                for p in tplayers:
                    pid = p.get("player_id") or p.get("id") or p.get("pid")
                    name = p.get("name") or p.get("player_name")
                    if pid is not None and name:
                        name_map[pid] = name
    return name_map


# --- Core computation: distances, sprint frames ---
def compute_distances_and_sprints(frames: List[dict], fps: int) -> Tuple[Dict[Any,float], Dict[Any,int], Dict[Any,str]]:
    """
    Returns:
      distances: pid -> total distance (meters)
      sprints: pid -> sprint-frame count (frames where speed >= threshold)
      pid_to_team_label: pid -> team label discovered from frames (if present)
    """
    dt = 1.0 / fps
    # store last position per pid to compute incremental distance
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
            # normalize pid type: prefer int if possible else keep as-is
            try:
                pid_key = int(pid)
            except Exception:
                pid_key = pid

            # set team label if present and not already set
            if team_label is not None and pid_key not in pid_team:
                pid_team[pid_key] = team_label

            # compute incremental distance
            if pid_key in last_pos:
                dx = x - last_pos[pid_key][0]
                dy = y - last_pos[pid_key][1]
                dist = np.hypot(dx, dy)
                distances[pid_key] = distances.get(pid_key, 0.0) + float(dist)
                # speed (m/s) = dist / dt
                speed = dist / dt if dt > 0 else 0.0
                if speed >= HIGH_INTENSITY_THRESHOLD:
                    sprints[pid_key] = sprints.get(pid_key, 0) + 1
            else:
                # initialize
                distances.setdefault(pid_key, 0.0)
                sprints.setdefault(pid_key, 0)

            last_pos[pid_key] = (x, y)

    return distances, sprints, pid_team


# --- Utility plotting functions ---
def plot_bar(values_map: Dict[Any,float], titlestr: str, ylabel: str, out_path: Path,
             id_to_label: Dict[Any,str]=None, top_n: Optional[int]=None, color="#FF9999"):
    """
    values_map: pid -> numeric
    id_to_label: optional mapping from pid to human label (name)
    top_n: if provided, sort descending and display top_n only
    """
    if not values_map:
        print("No data to plot:", titlestr)
        return
    # sort descending
    items = sorted(values_map.items(), key=lambda kv: kv[1], reverse=True)
    if top_n:
        items = items[:top_n]
    ids = [kv[0] for kv in items]
    vals = [kv[1] for kv in items]
    labels = [id_to_label.get(i, str(i)) if id_to_label else str(i) for i in ids]

    plt.figure(figsize=(12,5))
    plt.bar(range(len(vals)), vals, color=color, alpha=0.95)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    plt.title(titlestr)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


# --- Main runner ---
def run(n_minutes: Optional[float] = None):
    frames = load_tracking()
    meta = load_match_metadata() or {}

    # FPS: try to get from metadata first
    fps_meta = None
    for k in ("fps","frame_rate","frameRate","sample_rate"):
        if meta.get(k):
            try:
                fps_meta = int(meta.get(k))
                break
            except Exception:
                pass
    fps_use = fps_meta or FPS
    print(f"Using FPS = {fps_use}")

    # select valid frames (if n_minutes is None -> full match)
    if n_minutes is None:
        valid_frames = [f for f in frames if (f.get("player_data") or f.get("players"))]
    else:
        max_frames = int(n_minutes * 60 * fps_use)
        valid_frames = [f for f in frames if (f.get("player_data") or f.get("players"))][:max_frames]
    print(f"Total frames loaded: {len(frames)} -> frames used for analysis: {len(valid_frames)}")

    if len(valid_frames) == 0:
        raise RuntimeError("No valid frames found. Check your tracking loader output.")

    # compute per-player distances & sprint frames
    distances, sprint_frames, pid_team_map_from_frames = compute_distances_and_sprints(valid_frames, fps_use)

    # convert sprint frames to sprint seconds
    sprint_seconds = {pid: int(frames_count) * (1.0 / fps_use) for pid, frames_count in sprint_frames.items()}

    # try to get team player id lists and team names from metadata
    home_ids_raw = meta.get("home_players") or meta.get("home_team_player_ids") or meta.get("home_squad") or []
    away_ids_raw = meta.get("away_players") or meta.get("away_team_player_ids") or meta.get("away_squad") or []
    # normalize numeric strings in metadata to int where possible (best effort)
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

    # team names
    home_name = meta.get("home_team") or meta.get("home_name") or meta.get("home") or "TeamA"
    away_name = meta.get("away_team") or meta.get("away_name") or meta.get("away") or "TeamB"

    # id -> player name mapping (from metadata)
    id_to_name = map_meta_player_names(meta)

    # Build mapping pid -> team (A/B)
    pid_to_team = {}
    # assign from metadata lists if they exist
    if home_ids:
        for pid in home_ids:
            pid_to_team[pid] = "A"
    if away_ids:
        for pid in away_ids:
            pid_to_team[pid] = "B"

    # fallback: use team labels discovered in frames
    for pid, tlabel in pid_team_map_from_frames.items():
        if pid in pid_to_team:
            continue
        if tlabel in ("home", "Home", 1, "1", "H", "h"):
            pid_to_team[pid] = "A"
        elif tlabel in ("away", "Away", 0, "0", "A", "a"):
            pid_to_team[pid] = "B"
        else:
            # unknown label, don't set yet
            pass

    # final fallback: split players into halves
    all_pids = sorted(set(list(distances.keys()) + list(sprint_seconds.keys())))
    if not any(v == "A" for v in pid_to_team.values()) and not any(v == "B" for v in pid_to_team.values()):
        half = len(all_pids)//2
        for i, pid in enumerate(all_pids):
            pid_to_team[pid] = "A" if i < half else "B"

    # Now group metrics per team
    team_metrics = {
        "A": {"name": str(home_name), "distances": {}, "sprints_seconds": {}, "ids": []},
        "B": {"name": str(away_name), "distances": {}, "sprints_seconds": {}, "ids": []}
    }
    for pid in all_pids:
        team = pid_to_team.get(pid, "A")
        dist_m = distances.get(pid, 0.0)
        sprint_s = sprint_seconds.get(pid, 0.0)
        team_metrics[team]["distances"][pid] = dist_m / 1000.0  # convert to km for plot
        team_metrics[team]["sprints_seconds"][pid] = sprint_s
        team_metrics[team]["ids"].append(pid)

    # compute a simple fatigue score per player: (distance normalized drop) + (sprint seconds scaled)
    # Here we use a heuristic: fatigue_score = (distance_km) + (sprint_seconds * 0.02)
    # (You can change scaling to suit.)
    for team in ("A","B"):
        distances_km = team_metrics[team]["distances"]
        sprints_s = team_metrics[team]["sprints_seconds"]
        # build fatigue map
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
        out_dir = OUTPUT / f"team_{team}_{tname.replace(' ','_')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # distance plot (km)
        plot_bar(team_metrics[team]["distances"],
                 f"Top players by Total Distance (km) — {tname}",
                 "Distance (km)",
                 out_dir / f"player_total_distance_{tname.replace(' ','_')}.png",
                 id_to_label={pid: id_to_name.get(pid, str(pid)) for pid in team_metrics[team]["distances"].keys()},
                 color="#ff9f9f")

        # sprint seconds plot
        plot_bar(team_metrics[team]["sprints_seconds"],
                 f"Top players by Sprint Time (s) — {tname}",
                 "Sprint Time (s)",
                 out_dir / f"player_sprint_seconds_{tname.replace(' ','_')}.png",
                 id_to_label={pid: id_to_name.get(pid, str(pid)) for pid in team_metrics[team]["sprints_seconds"].keys()},
                 color="#9fc5ff")

        # fatigue score
        plot_bar(team_metrics[team]["fatigue_score"],
                 f"Top players by Fatigue Score (higher = more fatigued) — {tname}",
                 "Fatigue score (heuristic)",
                 out_dir / f"player_fatigue_{tname.replace(' ','_')}.png",
                 id_to_label={pid: id_to_name.get(pid, str(pid)) for pid in team_metrics[team]["fatigue_score"].keys()},
                 color="#ffd39f")

        # Save JSON summary
        summary = {
            "team_name": tname,
            "num_players_analyzed": len(team_metrics[team]["ids"]),
            "distances_km": {str(pid): team_metrics[team]["distances"][pid] for pid in team_metrics[team]["distances"]},
            "sprint_seconds": {str(pid): team_metrics[team]["sprints_seconds"][pid] for pid in team_metrics[team]["sprints_seconds"]},
            "fatigue_score": {str(pid): team_metrics[team]["fatigue_score"][pid] for pid in team_metrics[team]["fatigue_score"]},
            "id_to_name": {str(pid): id_to_name.get(pid, None) for pid in team_metrics[team]["ids"]}
        }
        with open(out_dir / f"summary_{tname.replace(' ','_')}.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        print(f"Saved outputs for team {tname} in {out_dir.resolve()}")

    # Also save combined summary in root output folder
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


if __name__ == "__main__":
    # full match by default
    run(n_minutes=None)
