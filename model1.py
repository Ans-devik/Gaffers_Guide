# # model1.py - Dynamic Tactical Shape Analysis (5-minute window)
# # Uses model0_load_data.py for loading

# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from scipy.spatial import ConvexHull
# from model0_load_data import load_tracking, load_match_metadata

# FPS = 25  # SkillCorner default

# # ------------------------------------------------
# # Select valid frames and limit to first N minutes
# # ------------------------------------------------
# def select_valid_frames(frames, n_minutes=5):
#     valid_frames = [f for f in frames if f.get("player_data") and len(f["player_data"]) > 0]
#     max_frames = int(n_minutes * 60 * FPS)
#     return valid_frames[:max_frames]

# # ------------------------------------------------
# # Team Compactness (Convex Hull area)
# # ------------------------------------------------
# # ------------------------------------------------
# # Team Compactness (Convex Hull area)
# # ------------------------------------------------
# def compute_team_compactness(frames, team_id=None):
#     compactness = []

#     for f in frames:
#         # Use all players, ignore team_id if not present
#         players = [[p["x"], p["y"]] for p in f["player_data"] if p.get("x") is not None]

#         if len(players) < 3:
#             compactness.append(np.nan)
#             continue

#         try:
#             hull = ConvexHull(np.array(players))
#             compactness.append(hull.area)
#         except:
#             compactness.append(np.nan)

#     return compactness


# # ------------------------------------------------
# # Defensive Line Height
# # ------------------------------------------------
# def compute_defensive_line_height(frames, team_id=None):
#     heights = []

#     for f in frames:
#         # Use all players, ignore team_id
#         players = [p for p in f["player_data"] if p.get("x") is not None]

#         if len(players) == 0:
#             heights.append(np.nan)
#             continue

#         players_sorted = sorted(players, key=lambda p: p["x"])
#         def_line = players_sorted[:4]  # take first 4 as defensive line
#         avg_y = np.mean([p["y"] for p in def_line]) if def_line else np.nan
#         heights.append(avg_y)

#     return heights


# # ------------------------------------------------
# # Plotting
# # ------------------------------------------------
# def plot_metric(metric, title, save_path):
#     plt.figure(figsize=(10, 4))
#     plt.plot(metric, linewidth=2)
#     plt.title(title)
#     plt.xlabel("Frame")
#     plt.ylabel(title)
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.close()

# # ------------------------------------------------
# # Main Runner
# # ------------------------------------------------
# def run_model1(frames, meta, out_dir, n_minutes=5):
#     out_dir.mkdir(parents=True, exist_ok=True)

#     frames = select_valid_frames(frames, n_minutes)
#     print(f"Using {len(frames)} valid frames.")

#     teamA = meta["home_team"]["id"]
#     teamB = meta["away_team"]["id"]

#     print("Computing compactness...")
#     compA = compute_team_compactness(frames, teamA)
#     compB = compute_team_compactness(frames, teamB)

#     plot_metric(compA, f"Team {teamA} Compactness", out_dir / "compactness_A.png")
#     plot_metric(compB, f"Team {teamB} Compactness", out_dir / "compactness_B.png")

#     print("Computing defensive line height...")
#     dlA = compute_defensive_line_height(frames, teamA)
#     dlB = compute_defensive_line_height(frames, teamB)

#     plot_metric(dlA, f"Team {teamA} Defensive Line Height", out_dir / "def_line_A.png")
#     plot_metric(dlB, f"Team {teamB} Defensive Line Height", out_dir / "def_line_B.png")

#     print("Model 1 complete! Outputs saved in:", out_dir)

# # ------------------------------------------------
# # CLI
# # ------------------------------------------------
# if __name__ == "__main__":
#     frames = load_tracking()
#     meta = load_match_metadata()
#     OUTPUT = Path("model1_output")
#     run_model1(frames, meta, OUTPUT, n_minutes=5)
# model1.py - Dynamic Tactical Shape Analysis (full match + defensive-line buckets)
# Updated to: full 90+ minutes, team-specific defensive line, and minute-bucket categories
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
from model0_load_data import load_tracking, load_match_metadata

FPS = 25  # SkillCorner default

# ------------------------------------------------
# Select valid frames and optionally limit to N minutes
# ------------------------------------------------
def select_valid_frames(frames, n_minutes=None):
    """
    If n_minutes is None -> return all valid frames (full match).
    Otherwise return up to n_minutes of frames.
    """
    valid_frames = [f for f in frames if f.get("player_data") and len(f["player_data"]) > 0]
    if n_minutes is None:
        return valid_frames
    max_frames = int(n_minutes * 60 * FPS)
    return valid_frames[:max_frames]

# ------------------------------------------------
# Helpers: map players to teams (robust to different JSON formats)
# ------------------------------------------------
def map_players_to_teams(frames, meta):
    """
    Try to map player IDs to home/away using metadata first.
    Fallback: split unique player IDs into two halves (heuristic).
    Returns (home_players_set, away_players_set, home_team_id, away_team_id)
    """
    # Attempt 1: Use metadata if it includes player lists with IDs
    home_team_id = meta.get("home_team", {}).get("id")
    away_team_id = meta.get("away_team", {}).get("id")

    # gather all player ids
    all_pids = []
    for f in frames:
        for p in f.get("player_data", []):
            pid = p.get("player_id") or p.get("id")
            if pid is not None:
                all_pids.append(pid)
    all_pids = list(dict.fromkeys(all_pids))  # keep order, unique

    # Attempt 2: If frames contain explicit 'team_id' entries for players, build sets directly
    home_players = set()
    away_players = set()
    found_teamfield = False
    for f in frames:
        for p in f.get("player_data", []):
            pid = p.get("player_id") or p.get("id")
            if pid is None:
                continue
            # common possible keys: 'team_id', 'team'
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

    # Fallback heuristic: split unique IDs into two equal sets
    if not found_teamfield or (not home_players and not away_players):
        # simple split
        mid = len(all_pids) // 2
        home_players = set(all_pids[:mid])
        away_players = set(all_pids[mid:])

    return home_players, away_players, home_team_id, away_team_id

# ------------------------------------------------
# Team Compactness (Convex Hull area)
# ------------------------------------------------
def compute_team_compactness(frames, team_players=None):
    compactness = []

    for f in frames:
        if team_players is None:
            players = [[p["x"], p["y"]] for p in f["player_data"] if p.get("x") is not None]
        else:
            players = []
            for p in f["player_data"]:
                pid = p.get("player_id") or p.get("id")
                if pid in team_players and p.get("x") is not None:
                    players.append([p["x"], p["y"]])

        if len(players) < 3:
            compactness.append(np.nan)
            continue

        try:
            hull = ConvexHull(np.array(players))
            compactness.append(hull.area)
        except:
            compactness.append(np.nan)

    return compactness

# ------------------------------------------------
# Defensive Line Height for a specific team
# ------------------------------------------------
def compute_defensive_line_height(frames, team_players=None):
    """
    For each frame, filter to players in team_players (set of player IDs).
    Sort players by x (longitudinal) and take the 4 deepest (lowest x depending on coordinate convention).
    Returns a list of per-frame defensive-line 'y' (or mean of first 4 player's y).
    """
    heights = []

    for f in frames:
        players = []
        for p in f["player_data"]:
            pid = p.get("player_id") or p.get("id")
            if p.get("x") is None or p.get("y") is None:
                continue
            if team_players is not None:
                if pid not in team_players:
                    continue
            # keep points as dict with x,y
            players.append({"x": p["x"], "y": p["y"], "id": pid})

        if len(players) == 0:
            heights.append(np.nan)
            continue

        # sort by x ascending (left-to-right). Defensive line assumed to be the 4 players with smallest x
        players_sorted = sorted(players, key=lambda p: p["x"])
        def_line = players_sorted[:4]  # take first 4 as defensive line
        if not def_line:
            heights.append(np.nan)
            continue
        avg_y = np.mean([p["y"] for p in def_line])
        heights.append(avg_y)

    return heights

# ------------------------------------------------
# Map frame index to match minute and category
# ------------------------------------------------
def frame_index_to_minute(frame_idx):
    return frame_idx / (FPS * 60.0)

def minute_to_bucket(minute):
    """
    Buckets defined by user:
      0-30 -> deep
      30-60 -> normal
      60-90 -> high
      above 90 -> aggressive
    """
    if minute < 0:
        return "unknown"
    if minute < 30:
        return "deep"
    if minute < 60:
        return "normal"
    if minute < 90:
        return "high"
    return "aggressive"

# ------------------------------------------------
# Plotting: time-series with bucket color-coding
# ------------------------------------------------
def plot_defensive_line_time_series(heights, title, save_path):
    minutes = [frame_index_to_minute(i) for i in range(len(heights))]
    buckets = [minute_to_bucket(m) for m in minutes]

    # map buckets to colors
    color_map = {"deep": "tab:blue", "normal": "tab:green", "high": "tab:orange", "aggressive": "tab:red", "unknown": "gray"}
    colors = [color_map.get(b, "gray") for b in buckets]

    plt.figure(figsize=(12, 4))
    # Plot as scatter colored by bucket for clarity across minute ranges
    plt.scatter(minutes, heights, c=colors, s=8)
    # also plot a thin line
    plt.plot(minutes, heights, linewidth=0.7, alpha=0.5)

    # create legend handles
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
    print(f"Saved {save_path}")

# ------------------------------------------------
# Main Runner
# ------------------------------------------------
def run_model1(frames, meta, out_dir, n_minutes=None):
    """
    n_minutes=None -> analyze full match (all frames).
    Otherwise limit to n_minutes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = select_valid_frames(frames, n_minutes)
    print(f"Using {len(frames)} valid frames.")

    # build team player sets
    home_players, away_players, home_team_id, away_team_id = map_players_to_teams(frames, meta)

    # compute compactness per team (optional)
    print("Computing team compactness (per team)...")
    compA = compute_team_compactness(frames, team_players=home_players)
    compB = compute_team_compactness(frames, team_players=away_players)
    # We keep the compactness plotting if useful
    try:
        plot_defensive_line_time_series(compA, f"Team {home_team_id} Compactness (per frame)", out_dir / "compactness_A.png")
        plot_defensive_line_time_series(compB, f"Team {away_team_id} Compactness (per frame)", out_dir / "compactness_B.png")
    except Exception as e:
        print("Couldn't plot compactness time series:", e)

    # compute defensive line height per team
    print("Computing defensive line height (team A/home)...")
    dlA = compute_defensive_line_height(frames, team_players=home_players)
    print("Computing defensive line height (team B/away)...")
    dlB = compute_defensive_line_height(frames, team_players=away_players)

    # plot two graphs color-coded by minute bucket
    plot_defensive_line_time_series(dlA, f"Team {home_team_id} Defensive Line Height (minutes bucketed)", out_dir / "def_line_A.png")
    plot_defensive_line_time_series(dlB, f"Team {away_team_id} Defensive Line Height (minutes bucketed)", out_dir / "def_line_B.png")

    print("Model 1 complete! Outputs saved in:", out_dir)

# ------------------------------------------------
# CLI
# ------------------------------------------------
if __name__ == "__main__":
    frames = load_tracking()
    meta = load_match_metadata()
    OUTPUT = Path("model1_output")
    # n_minutes=None -> full match (includes >90 for stoppage time)
    run_model1(frames, meta, OUTPUT, n_minutes=None)
