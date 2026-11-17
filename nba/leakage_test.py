"""Automated temporal-leakage checks.

  python -m nba.leakage_test

1. Future-scramble test: for sampled dates D, randomize every box score on/after D and rebuild
   features from scratch. Features for games on D must be bit-identical to the clean build.
2. Truncation test: rebuild with box scores strictly before D only; features on D identical.
3. Locked predictions contain no outcome columns, and every row was trained only on games
   dated before it.
4. No market-odds column is a model input.
"""
import numpy as np
import pandas as pd

from .backtest import load_db, registry
from .config import OUT
from .features import build_features, feature_columns


def main(n_dates=4):
    db = load_db()
    params = registry()[-1]["feature_params"] if registry() else {}
    sch, tb = db["schedule"], db["team_box"]
    full = build_features(sch, tb, params).set_index("game_id")
    cols = feature_columns(full.reset_index())
    rng = np.random.default_rng(1)
    dates = sorted(sch[sch.season >= 2010].game_date.unique())
    picks = sorted(rng.choice(dates, n_dates, replace=False))
    stat_cols = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov", "ast"]
    for d in picks:
        on_d = sch[sch.game_date == d].game_id
        # scramble the future (including D itself)
        fut = tb.game_date >= d
        tb2 = tb.copy()
        for c in stat_cols:
            tb2.loc[fut, c] = rng.permutation(tb2.loc[fut, c].to_numpy())
        tb2.loc[fut, "fga"] = tb2.loc[fut, ["fga", "fgm"]].max(axis=1) + 1
        f2 = build_features(sch, tb2, params).set_index("game_id")
        pd.testing.assert_frame_equal(full.loc[on_d, cols], f2.loc[on_d, cols])
        # truncate the future entirely
        f3 = build_features(sch, tb[~fut], params).set_index("game_id")
        pd.testing.assert_frame_equal(full.loc[on_d, cols], f3.loc[on_d, cols])
        print(f"ok  {d}: {len(on_d)} games unchanged under future scramble + truncation")

    odds_like = [c for c in cols if any(s in c for s in ("spread", "ml_", "odds", "total_line", "close", "open"))]
    assert not odds_like, f"odds columns used as features: {odds_like}"
    print("ok  no market columns among", len(cols), "features")

    p = OUT / "predictions.csv"
    if p.exists():
        pred = pd.read_csv(p, dtype={"game_id": str}, nrows=5)
        bad = [c for c in pred.columns if c in ("home_pts", "away_pts") or c.startswith("actual")]
        assert not bad, f"outcome columns in locked predictions: {bad}"
        pred = pd.read_csv(p, usecols=["game_date", "train_through", "cutoff"])
        assert (pred.train_through < pred.game_date).all(), "model trained on a same-day/future game"
        assert (pred.cutoff.str[:10] == pred.game_date).all()
        print(f"ok  {len(pred)} locked predictions: no outcome columns, all trained strictly on prior dates")
    entry = registry()[-1] if registry() else {}
    if entry.get("injuries"):
        injury_checks(db, rng)
    print("LEAKAGE TESTS PASSED")


def injury_checks(db, rng):
    import sqlite3
    from .config import DB_PATH
    from .injuries import LEAD, player_logs, player_values
    con = sqlite3.connect(DB_PATH)
    ch = pd.read_sql("select * from injury_report_choice where published is not null", con)
    ent = pd.read_sql("select game_id, published from injury_entries", con)
    tb = pd.read_sql("select game_id, team_id, game_date from team_box", con)
    con.close()
    pub, tip = pd.to_datetime(ch.published), pd.to_datetime(ch.tip_et)
    assert (pub <= tip - LEAD).all(), "injury report published less than 60 min before tip"
    assert (pub.dt.strftime("%Y-%m-%d") <= ch.game_date).all()
    e = ent.merge(ch[["game_id", "published"]], on="game_id", suffixes=("", "_chosen"))
    assert (e.published == e.published_chosen).all(), "injury entry from a report other than the chosen one"
    print(f"ok  {len(ch)} games: every injury report used was published >= 60 min before tip")
    pl = player_logs()
    dates = sorted(rng.choice(sorted(ch.game_date.unique()), 3, replace=False))
    full = player_values(pl, tb, dates)
    for d in dates:
        trunc = player_values(pl[pl.GAME_DATE < d], tb[tb.game_date < d], [d])
        assert full[d] == trunc[d], f"player values for {d} depend on later games"
        print(f"ok  {d}: player values unchanged when all later player games are removed")


if __name__ == "__main__":
    main()
