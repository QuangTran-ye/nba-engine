"""Pregame feature engine (streaming, point-in-time).

Walks game dates in order. For date D the information cutoff is D 09:00 ET. Before computing
features for D's games, the engine folds in every box score whose `available_at` <= cutoff,
so a game's own result (and anything later) can never reach its features. Schedule-derived
features (rest, travel) use only the published schedule.
"""
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .config import ARENAS, SEATTLE

CUTOFF_TIME = "09:00"
DEFAULT_PARAMS = {"halflife": 12.0, "season_carry": 0.35, "elo_k": 20.0, "elo_home": 100.0,
                  "elo_revert": 0.25}
STATS = ["off_rtg", "def_rtg", "net", "pace", "efg", "tov", "orb", "ftr", "fg3ar",
         "opp_efg", "opp_tov", "opp_orb", "opp_ftr", "margin", "adj_net"]
LEAGUE_PRIOR = {"off_rtg": 108, "def_rtg": 108, "net": 0, "pace": 96, "efg": .51, "tov": .13,
                "orb": .25, "ftr": .2, "fg3ar": .33, "opp_efg": .51, "opp_tov": .13,
                "opp_orb": .25, "opp_ftr": .2, "margin": 0, "adj_net": 0}


def cutoff_for(date: str) -> str:
    return f"{date} {CUTOFF_TIME}"


def game_stats(team_box: pd.DataFrame) -> pd.DataFrame:
    """Per team-game box-score rates (needs both sides of each game)."""
    t = team_box.copy()
    o = t[["game_id", "team_id", "pts", "fga", "fgm", "fg3m", "oreb", "dreb", "tov", "fta", "ftm"]] \
        .add_prefix("o_").rename(columns={"o_game_id": "game_id", "o_team_id": "opp_id"})
    t = t.merge(o, on=["game_id", "opp_id"])
    poss = 0.5 * ((t.fga - t.oreb + t.tov + 0.44 * t.fta) + (t.o_fga - t.o_oreb + t.o_tov + 0.44 * t.o_fta))
    t["poss"] = poss
    t["off_rtg"] = 100 * t.pts / poss
    t["def_rtg"] = 100 * t.o_pts / poss
    t["net"] = t.off_rtg - t.def_rtg
    t["pace"] = poss * 48 / (t.minutes / 5)
    t["efg"] = (t.fgm + 0.5 * t.fg3m) / t.fga
    t["tov"] = t.tov / poss
    t["orb"] = t.oreb / (t.oreb + t.o_dreb)
    t["ftr"] = t.ftm / t.fga
    t["fg3ar"] = t.fg3a / t.fga
    t["opp_efg"] = (t.o_fgm + 0.5 * t.o_fg3m) / t.o_fga
    t["opp_tov"] = t.o_tov / poss
    t["opp_orb"] = t.o_oreb / (t.o_oreb + t.dreb)
    t["opp_ftr"] = t.o_ftm / t.o_fga
    t["margin"] = t.pts - t.o_pts
    return t


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def arena(team_id, season):
    # ponytail: current arena coords for all years (Seattle only for 2007-08).
    return SEATTLE if team_id == 1610612760 and season == 2008 else ARENAS[team_id]


class TeamState:
    def __init__(self):
        self.sw = 0.0
        self.swx = {k: 0.0 for k in STATS}
        self.elo = 1505.0
        self.season = None
        self.gp = 0
        self.season_net = 0.0
        self.last5 = deque(maxlen=5)

    def mean(self, k):
        # weak prior (1 game of league average) keeps early values finite
        return (self.swx[k] + LEAGUE_PRIOR[k]) / (self.sw + 1.0)


def build_features(schedule: pd.DataFrame, team_box: pd.DataFrame, params=None) -> pd.DataFrame:
    p = {**DEFAULT_PARAMS, **(params or {})}
    decay = 0.5 ** (1 / p["halflife"])
    gs = game_stats(team_box)
    # queue of results, released strictly by available_at
    by_game = {gid: g for gid, g in gs.groupby("game_id")}
    release = gs.groupby("game_id").available_at.first().sort_values()
    rel_ids, rel_times, ri = list(release.index), list(release.values), 0

    sched = schedule.sort_values(["game_date", "game_id"])
    neutral = set(sched.game_id[sched.neutral == 1])
    st = defaultdict(TeamState)
    team_dates = defaultdict(list)           # schedule-derived (public in advance)
    for r in sched.itertuples():
        team_dates[r.home_id].append((r.game_date, r.home_id, r.season))
        team_dates[r.away_id].append((r.game_date, r.home_id, r.season))
    sched_idx = {tid: {d: i for i, (d, _, _) in enumerate(v)} for tid, v in team_dates.items()}

    def new_season(tid, season):
        s = st[tid]
        if s.season is not None and s.season != season:
            s.elo = 1505 + (1 - p["elo_revert"]) * (s.elo - 1505)
            s.sw *= p["season_carry"]
            for k in STATS: s.swx[k] *= p["season_carry"]
        if s.season != season:
            s.season, s.gp, s.season_net = season, 0, 0.0
            s.last5.clear()

    def apply_result(gid):
        g = by_game[gid]
        rows = {r.team_id: r for r in g.itertuples()}
        (t1, r1), (t2, r2) = rows.items()
        # opponent-adjusted net uses the opponent's strength *before* this game
        adj = {t1: r1.net + st[t2].mean("net"), t2: r2.net + st[t1].mean("net")}
        home = t1 if r1.is_home else t2
        away = t2 if home == t1 else t1
        rh = rows[home]
        hca = 0.0 if gid in neutral else p["elo_home"]
        # Elo (538-style MOV multiplier)
        eh, ea = st[home].elo, st[away].elo
        exp_h = 1 / (1 + 10 ** (-(eh + hca - ea) / 400))
        win_h = 1.0 if rh.margin > 0 else 0.0
        wdiff = (eh + hca - ea) if win_h else (ea - eh - hca)
        mult = (abs(rh.margin) + 3) ** 0.8 / (7.5 + 0.006 * wdiff)
        delta = p["elo_k"] * mult * (win_h - exp_h)
        st[home].elo += delta
        st[away].elo -= delta
        for t, r in rows.items():
            s = st[t]
            s.sw = s.sw * decay + 1
            for k in STATS:
                x = adj[t] if k == "adj_net" else getattr(r, k)
                s.swx[k] = s.swx[k] * decay + x
            s.gp += 1
            s.season_net += r.net
            s.last5.append(r.net)

    out = []
    for date, day in sched.groupby("game_date", sort=True):
        cutoff = cutoff_for(date)
        while ri < len(rel_ids) and rel_times[ri] <= cutoff:
            apply_result(rel_ids[ri]); ri += 1
        for r in day.itertuples():
            row = {"game_id": r.game_id, "season": r.season, "game_date": date, "cutoff": cutoff,
                   "home_id": r.home_id, "away_id": r.away_id, "neutral": r.neutral}
            for side, tid in (("h", r.home_id), ("a", r.away_id)):
                new_season(tid, r.season)
                s = st[tid]
                for k in STATS:
                    row[f"{side}_{k}"] = s.mean(k)
                row[f"{side}_elo"] = s.elo
                row[f"{side}_gp"] = s.gp
                # no games yet this season -> fall back to the decayed long-run rating
                row[f"{side}_season_net"] = s.season_net / s.gp if s.gp else s.mean("net")
                row[f"{side}_last5_net"] = float(np.mean(s.last5)) if s.last5 else s.mean("net")
                # schedule: rest, back-to-back, density, travel
                i = sched_idx[tid][date]
                prev = team_dates[tid][i - 1] if i > 0 else None
                same_season = prev is not None and prev[2] == r.season
                rest = (pd.Timestamp(date) - pd.Timestamp(prev[0])).days if same_season else 7
                row[f"{side}_rest"] = min(rest, 7)
                row[f"{side}_b2b"] = int(rest == 1)
                lo = pd.Timestamp(date) - pd.Timedelta(days=7)
                row[f"{side}_g7"] = sum(1 for d, _, _ in team_dates[tid][max(0, i - 6):i] if pd.Timestamp(d) > lo)
                # ponytail: neutral sites use the designated home arena for travel
                row[f"{side}_travel"] = haversine(arena(prev[1], prev[2]), arena(r.home_id, r.season)) if same_season else 0.0
            out.append(row)
    f = pd.DataFrame(out)
    f["elo_diff"] = f.h_elo + p["elo_home"] * (1 - f.neutral) - f.a_elo
    f["elo_p"] = 1 / (1 + 10 ** (-f.elo_diff / 400))
    f["elo_margin"] = f.elo_diff / 28.0
    for k in STATS + ["season_net", "last5_net", "rest", "travel"]:
        f[f"d_{k}"] = f[f"h_{k}"] - f[f"a_{k}"]
    f["pace_sum"] = f.h_pace + f.a_pace
    f["exp_total"] = (f.h_off_rtg + f.a_def_rtg + f.a_off_rtg + f.h_def_rtg) / 2 * f.pace_sum / 200
    return f


def feature_columns(f: pd.DataFrame):
    skip = {"game_id", "season", "game_date", "cutoff", "home_id", "away_id", "inj_published", "tip_et"}
    return [c for c in f.columns if c not in skip]
