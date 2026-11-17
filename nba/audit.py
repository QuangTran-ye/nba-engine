"""Independent reproduction of reported baseline numbers + targeted discrepancy checks.

  python -m nba.audit            # writes outputs/audit/*.csv and prints a summary

Deliberately does NOT reuse evaluate.py's metric code: metrics are recomputed from the locked
predictions file, the results table and the raw odds table with separate, minimal code.
"""
import sqlite3

import numpy as np
import pandas as pd

from .config import DB_PATH, ROOT, split_of

OUT = ROOT / "outputs" / "audit"
RNG = np.random.default_rng(11)


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def blocked_ci(x, dates, n=2000, stat=np.mean):
    """Bootstrap resampling whole game dates (games on one date share news, refs, market moves)."""
    df = pd.DataFrame({"x": np.asarray(x, float), "d": np.asarray(dates)})
    groups = [g.x.to_numpy() for _, g in df.groupby("d")]
    k = len(groups)
    sums = np.array([g.sum() for g in groups]); cnt = np.array([len(g) for g in groups])
    idx = RNG.integers(0, k, (n, k))
    s = sums[idx].sum(1) / cnt[idx].sum(1)
    return tuple(np.percentile(s, [2.5, 97.5]))


def iid_ci(x, n=2000):
    x = np.asarray(x, float)
    s = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    res = pd.read_sql("select game_id, home_pts, away_pts from results", con)
    odds = pd.read_sql("select * from odds", con)
    sched = pd.read_sql("select game_id, season, game_date from schedule", con)
    con.close()
    pred = pd.read_csv(ROOT / "outputs" / "predictions.csv", dtype={"game_id": str})
    d = pred.merge(res, on="game_id")
    d["y"] = (d.home_pts > d.away_pts).astype(int)
    d["m"] = d.home_pts - d.away_pts
    d["t"] = d.home_pts + d.away_pts
    rep = []

    # 1. headline model metrics, recomputed
    for split in ("test", "holdout"):
        e = d[d.split == split]
        p = e.p_home.clip(1e-6, 1 - 1e-6)
        ll = -(e.y * np.log(p) + (1 - e.y) * np.log(1 - p))
        rep += [(split, "v1 accuracy", ((p >= .5) == e.y).mean(), len(e)),
                (split, "v1 brier", ((p - e.y) ** 2).mean(), len(e)),
                (split, "v1 log_loss", ll.mean(), len(e)),
                (split, "v1 margin_mae", (e.m - e.pred_margin).abs().mean(), len(e)),
                (split, "v1 total_mae", (e.t - e.pred_total).abs().mean(), len(e))]

    # 2. market benchmark rebuilt from the raw odds table (no reuse of backtest.market_lines)
    o = odds[odds.line_type != "open"].copy()
    o["pri"] = o.source.map({"espn": 0, "kaggle_sbr": 1})
    o = o.sort_values("pri")
    first = o.groupby("game_id").first()
    ml = o.dropna(subset=["ml_home", "ml_away"]).groupby("game_id").first()
    mk = d[["game_id", "split", "game_date", "y", "m", "t"]].merge(
        first[["source", "home_spread", "total"]].reset_index(), on="game_id", how="left").merge(
        ml[["ml_home", "ml_away"]].rename(columns=lambda c: "raw_" + c).reset_index(), on="game_id", how="left")
    dh, da = dec(mk.raw_ml_home), dec(mk.raw_ml_away)
    mk["p_mkt"] = (1 / dh) / (1 / dh + 1 / da)
    # same sanity filter the pipeline uses (favourite by ML must match favourite by >=2pt spread)
    bad = (mk.home_spread.abs() >= 2) & ((mk.home_spread < 0) != (mk.raw_ml_home < mk.raw_ml_away))
    mk.loc[bad, "p_mkt"] = np.nan
    for split in ("test", "holdout"):
        e = mk[mk.split == split]
        q = e.dropna(subset=["p_mkt"])
        pq = q.p_mkt.clip(1e-6, 1 - 1e-6)
        rep += [(split, "market accuracy", ((pq >= .5) == q.y).mean(), len(q)),
                (split, "market brier", ((pq - q.y) ** 2).mean(), len(q)),
                (split, "market log_loss", (-(q.y * np.log(pq) + (1 - q.y) * np.log(1 - pq))).mean(), len(q)),
                (split, "market margin_mae", (e.m + e.home_spread).abs().mean(), int(e.home_spread.notna().sum())),
                (split, "market total_mae", (e.t - e.total).abs().mean(), int(e.total.notna().sum()))]
    rp = pd.DataFrame(rep, columns=["split", "metric", "value", "n"])
    reported = pd.read_csv(ROOT / "outputs" / "model_comparison.csv")
    look = {("v1", "accuracy"): ("ensemble", "accuracy"), ("v1", "brier"): ("ensemble", "brier"),
            ("v1", "log_loss"): ("ensemble", "log_loss"), ("v1", "margin_mae"): ("ensemble", "margin_mae"),
            ("v1", "total_mae"): ("ensemble", "total_mae"), ("market", "accuracy"): ("market_moneyline", "accuracy"),
            ("market", "brier"): ("market_moneyline", "brier"), ("market", "log_loss"): ("market_moneyline", "log_loss"),
            ("market", "margin_mae"): ("market_close_lines", "margin_mae"), ("market", "total_mae"): ("market_close_lines", "total_mae")}
    def rep_val(r):
        who, met = r.metric.split(" ")
        m, c = look[(who, met)]
        v = reported[(reported.split == r.split) & (reported.model == m)][c]
        return v.iloc[0] if len(v) else np.nan
    rp["reported"] = rp.apply(rep_val, axis=1)
    rp["abs_diff"] = (rp.value - rp.reported).abs()
    rp.to_csv(OUT / "reproduction.csv", index=False)
    print("== 1-2. Reproduction of reported metrics (independent code)")
    print(rp.round(4).to_string(index=False))

    # 3. which odds sources make up the 'market' per season
    src = mk.merge(sched[["game_id", "season"]], on="game_id").assign(
        ml_source=lambda x: np.where(x.raw_ml_home.notna(), "has_ml", "no_ml"))
    comp = src.groupby(["season", "source"], dropna=False).size().unstack(fill_value=0)
    comp["ml_available"] = src.groupby("season").raw_ml_home.apply(lambda s: s.notna().sum())
    comp.to_csv(OUT / "market_source_mix.csv")
    print("\n== 3. Market benchmark source mix by season (closing line source; ML availability)")
    print(comp.to_string())

    # 4. iid vs date-blocked bootstrap on key quantities (test split)
    e = d[d.split == "test"]
    p = e.p_home.clip(1e-6, 1 - 1e-6)
    ll = -(e.y * np.log(p) + (1 - e.y) * np.log(1 - p))
    em = mk[mk.split == "test"].set_index("game_id").loc[e.game_id]
    q = em.p_mkt.clip(1e-6, 1 - 1e-6).to_numpy()
    llm = -(e.y.to_numpy() * np.log(q) + (1 - e.y.to_numpy()) * np.log(1 - q))
    diff = ll.to_numpy() - llm
    ok = ~np.isnan(diff)
    rows = []
    for name, x, dts in (("v1 log loss", ll.to_numpy(), e.game_date), ("v1 - market log loss", diff[ok], e.game_date[ok]),
                         ("v1 margin abs error", (e.m - e.pred_margin).abs().to_numpy(), e.game_date)):
        rows.append({"quantity": name, "mean": np.mean(x), "iid_ci": iid_ci(x), "date_blocked_ci": blocked_ci(x, dts)})
    sp = e[e.close_spread_bet.isin(["home", "away"])]
    cov = sp.m + sp.close_home_spread
    win = np.where(sp.close_spread_bet == "home", cov > 0, cov < 0)
    dch = np.where(sp.close_spread_bet == "home", sp.close_spread_dec_home, sp.close_spread_dec_away)
    prof = np.where(cov == 0, 0.0, np.where(win, dch - 1, -1.0))
    rows.append({"quantity": "spread ROI at close", "mean": prof.mean(), "iid_ci": iid_ci(prof),
                 "date_blocked_ci": blocked_ci(prof, sp.game_date)})
    bt = pd.DataFrame(rows)
    bt["iid_width"] = bt.iid_ci.map(lambda c: c[1] - c[0])
    bt["blocked_width"] = bt.date_blocked_ci.map(lambda c: c[1] - c[0])
    bt.to_csv(OUT / "bootstrap_iid_vs_blocked.csv", index=False)
    print("\n== 4. iid vs date-blocked bootstrap (test)")
    print(bt.round(4).to_string(index=False))

    # 5. CLV definition: mean vs share 'beating' the close (ties)
    rows = []
    for mkname, side, lo, lc, sgn in (("spread", "open_spread_bet", "open_home_spread", "close_home_spread", "home"),
                                      ("total", "open_total_bet", "open_total", "close_total", "over")):
        b = d[d[side].isin(["home", "away", "over", "under"]) & d[lc].notna()]
        if mkname == "spread":
            c = np.where(b[side] == "home", b[lo] - b[lc], b[lc] - b[lo])
        else:
            c = np.where(b[side] == "over", b[lc] - b[lo], b[lo] - b[lc])
        rows.append({"market": mkname, "bets": len(c), "mean_clv_pts": c.mean(), "beat": (c > 0).mean(),
                     "tie": (c == 0).mean(), "worse": (c < 0).mean(), "beat_given_moved": (c > 0).sum() / max((c != 0).sum(), 1)})
    cl = pd.DataFrame(rows)
    cl.to_csv(OUT / "clv_ties.csv", index=False)
    print("\n== 5. CLV: share beat / tie / worse (open-line bets, all seasons with openers)")
    print(cl.round(3).to_string(index=False))

    # 6. sample counts
    counts = {"schedule games 2026-27 (DB now)": int((sched.season == 2027).sum()),
              "holdout predictions locked": int((pred.split == "holdout").sum()),
              "test predictions": int((pred.split == "test").sum()),
              "test games with ML (pipeline)": int(pred[(pred.split == "test")].close_mkt_p_home.notna().sum()),
              "test games with ML (raw rebuild)": int(mk[(mk.split == "test")].p_mkt.notna().sum())}
    pd.Series(counts).to_csv(OUT / "sample_counts.csv")
    print("\n== 6. Sample counts"); print(pd.Series(counts).to_string())

    # 7. player-name collisions in injury value lookup (key is name only)
    try:
        from .injuries import player_logs
        pl = player_logs()
        coll = pl.groupby(["player_key"]).PLAYER_ID.nunique()
        coll = coll[coll > 1]
        names = pl[pl.player_key.isin(coll.index)].groupby("player_key").agg(
            ids=("PLAYER_ID", lambda s: sorted(set(s))), names=("PLAYER_NAME", lambda s: sorted(set(s))),
            seasons=("SEASON_ID", lambda s: f"{s.min()}-{s.max()}"))
        names.to_csv(OUT / "player_name_collisions.csv")
        print(f"\n== 7. Player-name collisions (same normalized name, different player id): {len(names)}")
        print(names.to_string())
    except Exception as ex:   # player logs not cached
        print("player collision check skipped:", ex)


if __name__ == "__main__":
    main()
