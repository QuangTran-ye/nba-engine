"""Data validation. Hard errors raise; soft issues are counted in outputs/data_validation.json.

  python -m nba.validate
"""
import json
import sqlite3

import pandas as pd

from .config import DB_PATH, OUT


def main():
    con = sqlite3.connect(DB_PATH)
    sch = pd.read_sql("select * from schedule", con)
    tb = pd.read_sql("select * from team_box", con)
    res = pd.read_sql("select * from results", con)
    odds = pd.read_sql("select * from odds", con)
    con.close()
    rep = {}

    # hard checks
    assert sch.game_id.is_unique and res.game_id.is_unique
    per = tb.groupby("game_id").is_home.agg(["size", "sum"])
    assert (per["size"] == 2).all() and (per["sum"] == 1).all(), "each game needs one home + one away row"
    assert set(sch.game_id) == set(res.game_id) == set(tb.game_id)
    h = tb[tb.is_home == 1].set_index("game_id").pts
    assert (res.set_index("game_id").home_pts == h.loc[res.game_id].values).all()
    assert (tb.available_at > tb.game_date + " 23:59").all(), "box score available before game ended"
    assert not (res.home_pts == res.away_pts).any(), "ties"
    dup = pd.concat([sch[["game_date", "home_id"]].rename(columns={"home_id": "t"}),
                     sch[["game_date", "away_id"]].rename(columns={"away_id": "t"})]).duplicated().sum()
    assert dup == 0, "team scheduled twice on one date"

    # soft checks
    rep["games_per_season"] = sch.groupby("season").size().to_dict()
    rep["box_minutes_not_240_or_OT"] = tb[~tb.minutes.isin([240, 265, 290, 315, 340, 365, 390])][
        ["game_id", "team_id", "minutes"]].to_dict("records")
    rep["totals_outside_150_300"] = int(((res.home_pts + res.away_pts) < 150).sum() + ((res.home_pts + res.away_pts) > 300).sum())
    o = odds.merge(sch[["game_id", "season"]], on="game_id")
    cov = o.groupby(["season", "source", "line_type"]).agg(
        games=("game_id", "nunique"), spread=("home_spread", "count"), total=("total", "count"),
        ml=("ml_home", "count")).reset_index()
    rep["odds_coverage"] = cov.to_dict("records")
    rep["odds_spread_out_of_range"] = int((o.home_spread.abs() > 25).sum())
    rep["odds_total_out_of_range"] = int(((o.total < 170) | (o.total > 280)).sum())
    fav_ml = o.dropna(subset=["ml_home", "home_spread"])
    fav_ml = fav_ml[fav_ml.home_spread.abs() >= 2]
    rep["ml_spread_sign_conflicts"] = int(((fav_ml.home_spread < 0) != (fav_ml.ml_home < fav_ml.ml_away)).sum())
    rep["ml_spread_checked"] = len(fav_ml)

    # Is the Kaggle/SBR line an opener or a closer? Compare with ESPN open/close where both exist.
    k = o[o.source == "kaggle_sbr"].set_index("game_id")
    for lt in ("open", "close"):
        e = o[(o.source == "espn") & (o.line_type == lt)].set_index("game_id")
        j = k.merge(e, left_index=True, right_index=True, suffixes=("", "_e"))
        rep[f"kaggle_vs_espn_{lt}"] = {
            "games": len(j),
            "spread_mean_abs_diff": float((j.home_spread - j.home_spread_e).abs().mean()),
            "spread_exact_match": float((j.home_spread == j.home_spread_e).mean()),
            "total_mean_abs_diff": float((j.total - j.total_e).abs().mean()),
            "total_exact_match": float((j.total == j.total_e).mean())}
    OUT.mkdir(exist_ok=True)
    (OUT / "data_validation.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "odds_coverage"}, indent=2, default=str))


if __name__ == "__main__":
    main()
