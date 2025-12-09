import pandas as pd
import numpy as np
from config import N_PLAYERS, SEASONS, WEEKS_PER_SEASON

def generate_players():
    positions = ["QB", "RB", "WR", "TE", "K"]
    probs = [0.10, 0.30, 0.40, 0.15, 0.05]

    players = []
    for i in range(N_PLAYERS):
        pos = np.random.choice(positions, p=probs)
        players.append({
            "player_id": f"P{i:04d}",
            "player_name": f"Player_{i}",
            "position": pos
        })

    return pd.DataFrame(players)


def generate_weekly_stats(players):
    rows = []

    for season in SEASONS:
        for week in range(1, WEEKS_PER_SEASON + 1):
            for _, row in players.iterrows():

                pos = row["position"]

                carries = np.random.poisson(10) if pos == "RB" else np.random.poisson(1)
                targets = np.random.poisson(8) if pos in ["WR", "TE", "RB"] else 0
                pass_att = np.random.poisson(35) if pos == "QB" else 0

                rows.append({
                    "player_id": row["player_id"],
                    "player_name": row["player_name"],
                    "position": pos,
                    "season": season,
                    "week": week,
                    "carries": carries,
                    "targets": targets,
                    "pass_att": pass_att,
                    "rush_yds": carries * np.random.randint(2, 6),
                    "rec_yds": targets * np.random.randint(5, 12),
                    "routes": np.random.randint(5, 35),
                    "snaps": np.random.randint(20, 70),
                    "fantasy_points": np.random.uniform(0, 25)
                })

    return pd.DataFrame(rows)


def generate_injuries(players):
    rows = []

    for season in SEASONS:
        for week in range(1, WEEKS_PER_SEASON + 1):
            for _, player in players.iterrows():

                base_prob = {
                    "RB": 0.12,
                    "WR": 0.08,
                    "TE": 0.10,
                    "QB": 0.05,
                    "K": 0.02
                }[player["position"]]

                injury = np.random.rand() < base_prob

                rows.append({
                    "player_id": player["player_id"],
                    "season": season,
                    "week": week,
                    "injured": 1 if injury else 0
                })

    return pd.DataFrame(rows)
