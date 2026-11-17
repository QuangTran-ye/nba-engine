"""Outcome evaluation. Runs only after predictions are locked; verifies the lock hash first.

  python -m nba.evaluate
"""
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd

from .config import DB_PATH, OUT, split_of

RNG = np.random.default_rng(0)
N_BOOT = 1000


def boot_ci(x, stat=np.mean, n=N_BOOT):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), (n, len(x)))
    s = np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]))


def _y(df):
    return (df.home_pts > df.away_pts).astype(int).to_numpy()


def prob_metrics(df, col="p_home"):
    y, p = _y(df), np.clip(df[col].to_numpy(float), 1e-6, 1 - 1e-6)
    brier = (p - y) ** 2
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    acc = ((p >= 0.5) == y).astype(float)
    return {"n": len(y), "accuracy": acc.mean(), "accuracy_ci": boot_ci(acc), "brier": brier.mean(),
            "brier_ci": boot_ci(brier), "log_loss": ll.mean(), "log_loss_ci": boot_ci(ll)}


def point_metrics(df, mcol="pred_margin", tcol="pred_total"):
    em = np.abs((df.home_pts - df.away_pts) - df[mcol]).to_numpy()
    et = np.abs((df.home_pts + df.away_pts) - df[tcol]).to_numpy()
    return {"margin_mae": em.mean(), "margin_mae_ci": boot_ci(em), "total_mae": et.mean(),
            "total_mae_ci": boot_ci(et)}


def max_drawdown(units):
    c = np.cumsum(units)
    peak = np.maximum.accumulate(np.concatenate([[0], c]))[1:]
    return float((peak - c).max()) if len(c) else 0.0


def settle(df, prefix):
    """Per-bet profit in units (1 unit staked) for each market at the given line set."""
    m = df.home_pts - df.away_pts
    t = df.home_pts + df.away_pts
    rows = []
    for mk, col in (("ml", "ml_bet"), ("spread", "spread_bet"), ("total", "total_bet")):
        b = df[df[f"{prefix}{col}"].isin(["home", "away", "over", "under"])]
        side = b[f"{prefix}{col}"]
        if mk == "ml":
            win = np.where(side == "home", m.loc[b.index] > 0, m.loc[b.index] < 0)
            dec = np.where(side == "home", b[f"{prefix}ml_dec_home"], b[f"{prefix}ml_dec_away"])
            push = np.zeros(len(b), bool)
        elif mk == "spread":
            cov = m.loc[b.index] + b[f"{prefix}home_spread"]
            win = np.where(side == "home", cov > 0, cov < 0)
            push = (cov == 0).to_numpy()
            dec = np.where(side == "home", b[f"{prefix}spread_dec_home"], b[f"{prefix}spread_dec_away"])
        else:
            d = t.loc[b.index] - b[f"{prefix}total"]
            win = np.where(side == "over", d > 0, d < 0)
            push = (d == 0).to_numpy()
            dec = np.where(side == "over", b[f"{prefix}over_dec"], b[f"{prefix}under_dec"])
        profit = np.where(push, 0.0, np.where(win, dec - 1, -1.0))
        rows.append(pd.DataFrame({"game_id": b.game_id, "game_date": b.game_date, "market": mk,
                                  "side": side, "profit": profit, "win": win & ~push, "push": push}))
    return pd.concat(rows, ignore_index=True).sort_values(["game_date", "game_id"])


def betting_table(df, prefix):
    s = settle(df, prefix)
    out = []
    for mk in ("ml", "spread", "total"):
        b = s[s.market == mk]
        n = len(b)
        dec = b[~b.push]
        out.append({"market": mk, "bets": n, "win_rate": dec.win.mean() if len(dec) else np.nan,
                    "units": b.profit.sum(), "roi": b.profit.mean() if n else np.nan,
                    "roi_ci": boot_ci(b.profit) if n > 1 else (np.nan, np.nan),
                    "max_drawdown": max_drawdown(b.profit.to_numpy())})
    return pd.DataFrame(out)


def clv(df):
    """Closing-line value for bets placed at the ESPN opener (2023-24+)."""
    o = df[df.open_spread_bet.isin(["home", "away"]) & df.close_home_spread.notna()]
    sp = np.where(o.open_spread_bet == "home", o.open_home_spread - o.close_home_spread,
                  o.close_home_spread - o.open_home_spread)
    t = df[df.open_total_bet.isin(["over", "under"]) & df.close_total.notna()]
    tp = np.where(t.open_total_bet == "over", t.close_total - t.open_total, t.open_total - t.close_total)
    m = df[df.open_ml_bet.isin(["home", "away"]) & df.close_mkt_p_home.notna()]
    pc = np.where(m.open_ml_bet == "home", m.close_mkt_p_home, 1 - m.close_mkt_p_home)
    dec = np.where(m.open_ml_bet == "home", m.open_ml_dec_home, m.open_ml_dec_away)
    mev = pc * dec - 1
    return pd.DataFrame([
        {"market": "spread", "bets": len(sp), "mean_clv": np.mean(sp) if len(sp) else np.nan, "unit": "points",
         "ci": boot_ci(sp), "share_beat_close": np.mean(sp > 0) if len(sp) else np.nan},
        {"market": "total", "bets": len(tp), "mean_clv": np.mean(tp) if len(tp) else np.nan, "unit": "points",
         "ci": boot_ci(tp), "share_beat_close": np.mean(tp > 0) if len(tp) else np.nan},
        {"market": "ml", "bets": len(mev), "mean_clv": np.mean(mev) if len(mev) else np.nan, "unit": "EV vs no-vig close",
         "ci": boot_ci(mev), "share_beat_close": np.mean(mev > 0) if len(mev) else np.nan}])


def verify_lock(path):
    want = path.with_suffix(".sha256").read_text().strip()
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == want, f"{path.name} changed after locking ({got} != {want})"
    reg = json.loads((OUT / "model_registry.json").read_text())
    assert reg[-1].get("predictions_sha256") == got, "registry hash mismatch"
    return got


def flat(d):
    r = {}
    for k, v in d.items():
        if isinstance(v, tuple):
            r[k + "_lo"], r[k + "_hi"] = v
        else:
            r[k] = v
    return r


def main():
    pred_path = OUT / "predictions.csv"
    digest = verify_lock(pred_path)
    pred = pd.read_csv(pred_path, dtype={"game_id": str})
    con = sqlite3.connect(DB_PATH)
    res = pd.read_sql("select game_id, home_pts, away_pts from results", con)
    dev_home = pd.read_sql("select r.home_pts > r.away_pts as hw, s.season from results r join schedule s using(game_id)", con)
    con.close()
    hw = dev_home[dev_home.season.map(split_of) == "dev"].hw.mean()     # home win rate learned on dev only
    # ---- separate evaluation dataset (actual outcomes live only here)
    ev = pred.merge(res, on="game_id", how="left")
    ev["actual_margin"] = ev.home_pts - ev.away_pts
    ev["actual_total"] = ev.home_pts + ev.away_pts
    ev["winner_correct"] = (ev.p_home >= 0.5) == (ev.actual_margin > 0)
    ev["margin_error"] = ev.pred_margin - ev.actual_margin
    ev["total_error"] = ev.pred_total - ev.actual_total
    ev[["game_id", "game_date", "season", "split", "matchup", "model_version", "home_pts", "away_pts",
        "actual_margin", "actual_total", "pred_margin", "pred_total", "p_home", "winner_correct",
        "margin_error", "total_error"]].to_csv(OUT / "evaluation.csv", index=False)

    long = pd.read_csv(OUT / "predictions_all_models.csv", dtype={"game_id": str}).merge(res, on="game_id")
    long = long.merge(pred[["game_id", "season", "split", "close_mkt_p_home", "close_home_spread", "close_total"]],
                      on="game_id")
    # ---- model + baseline comparison by split
    rows = []
    for split in ("dev", "valid", "test", "holdout"):
        e = ev[ev.split == split]
        if not len(e):
            continue
        cands = {"ensemble": e, **{m: long[(long.split == split) & (long.model == m)] for m in long.model.unique()}}
        for name, d in cands.items():
            rows.append({"split": split, "model": name, **flat(prob_metrics(d)), **flat(point_metrics(d))})
        rnd = e.assign(p_rand=RNG.random(len(e)))
        r = flat(prob_metrics(e.assign(p_home=0.5)))
        coin = ((rnd.p_rand >= .5) == (e.actual_margin > 0)).astype(float)
        r["accuracy"] = coin.mean()
        r["accuracy_ci_lo"], r["accuracy_ci_hi"] = boot_ci(coin)
        rows.append({"split": split, "model": "baseline_random", **r})
        rows.append({"split": split, "model": "baseline_home", **flat(prob_metrics(e.assign(p_home=hw)))})
        mk = e[e.close_mkt_p_home.notna()]
        rows.append({"split": split, "model": "market_moneyline", **flat(prob_metrics(mk, "close_mkt_p_home")),
                     "note": f"games with ML odds: {len(mk)}"})
        ms = e[e.close_home_spread.notna() & e.close_total.notna()]
        rows.append({"split": split, "model": "market_close_lines",
                     **flat(point_metrics(ms.assign(m=-ms.close_home_spread, t=ms.close_total), "m", "t")),
                     "n": len(ms)})
        # same-games comparison model vs market (fair: identical sample)
        rows.append({"split": split, "model": "ensemble_on_market_games", **flat(prob_metrics(mk)),
                     **flat(point_metrics(ms))})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "model_comparison.csv", index=False)

    # ---- betting
    bets = []
    for split in ("valid", "test", "holdout"):
        e = ev[ev.split == split]
        for prefix in ("close_", "open_"):
            if not e[f"{prefix}home_spread"].notna().any():
                continue
            t = betting_table(e, prefix)
            t.insert(0, "line", prefix.rstrip("_"))
            t.insert(0, "split", split)
            bets.append(t)
    bets = pd.concat(bets, ignore_index=True)
    bets[["roi_ci_lo", "roi_ci_hi"]] = pd.DataFrame(bets.pop("roi_ci").tolist())
    bets.to_csv(OUT / "betting_performance.csv", index=False)

    # against-the-spread pick on every game (no threshold), closing line
    ats = []
    for split in ("dev", "valid", "test", "holdout"):
        e = ev[(ev.split == split) & ev.close_home_spread.notna()]
        cov = e.actual_margin + e.close_home_spread
        pick_home = e.pred_margin + e.close_home_spread > 0
        dec = cov != 0
        hit = (pick_home == (cov > 0))[dec]
        ou = e.actual_total - e.close_total
        pick_over = e.pred_total > e.close_total
        hit_t = (pick_over == (ou > 0))[ou != 0]
        ats.append({"split": split, "games": int(dec.sum()), "ats_hit_rate": hit.mean(), "ats_ci": boot_ci(hit),
                    "ou_games": int((ou != 0).sum()), "ou_hit_rate": hit_t.mean(), "ou_ci": boot_ci(hit_t),
                    "breakeven_-110": 110 / 210})
    pd.DataFrame([flat(a) for a in ats]).to_csv(OUT / "ats_all_games.csv", index=False)

    c = []
    for split in ("test", "holdout"):
        e = ev[ev.split == split]
        if e.open_home_spread.notna().any():
            x = clv(e)
            x.insert(0, "split", split)
            c.append(x)
    if c:
        c = pd.concat(c, ignore_index=True)
        c[["ci_lo", "ci_hi"]] = pd.DataFrame(c.pop("ci").tolist())
        c.to_csv(OUT / "clv.csv", index=False)

    # ---- per-season and monthly consistency (ensemble)
    ev["month"] = ev.game_date.str[:7]
    per = []
    for keys, g in ev.groupby(["season", "split"]):
        g2 = g[g.close_home_spread.notna()]
        cov = g2.actual_margin + g2.close_home_spread
        hit = ((g2.pred_margin + g2.close_home_spread > 0) == (cov > 0))[cov != 0]
        per.append({"season": keys[0], "split": keys[1], **{k: v for k, v in prob_metrics(g).items() if "_ci" not in k},
                    **{k: v for k, v in point_metrics(g).items() if "_ci" not in k},
                    "mkt_margin_mae": np.abs(g2.actual_margin + g2.close_home_spread).mean(),
                    "mkt_total_mae": np.abs(g2.actual_total - g2.close_total).mean(),
                    "ats_hit_rate": hit.mean()})
    pd.DataFrame(per).to_csv(OUT / "per_season.csv", index=False)
    s = settle(ev[ev.split.isin(["test", "holdout"])], "close_")
    s["month"] = s.game_date.str[:7]
    s.groupby(["month", "market"]).agg(bets=("profit", "size"), units=("profit", "sum")).reset_index() \
        .to_csv(OUT / "monthly_betting.csv", index=False)
    s.to_csv(OUT / "bets_ledger.csv", index=False)
    print("evaluated locked predictions", digest)
    print(comp[comp.model.isin(["ensemble", "elo", "logreg", "rf", "lgbm", "baseline_home", "market_moneyline",
                                "ensemble_on_market_games", "market_close_lines"])]
          [["split", "model", "n", "accuracy", "brier", "log_loss", "margin_mae", "total_mae"]].to_string())
    print(bets.to_string())


def compare(dir_a, dir_b, label_a="v1", label_b="v2"):
    """Paired comparison of two locked prediction sets on the games both cover."""
    import sys
    from .config import ROOT
    con = sqlite3.connect(DB_PATH)
    res = pd.read_sql("select game_id, home_pts, away_pts from results", con)
    con.close()
    a = pd.read_csv(ROOT / dir_a / "predictions.csv", dtype={"game_id": str})
    b = pd.read_csv(ROOT / dir_b / "predictions.csv", dtype={"game_id": str})
    feats = pd.read_csv(ROOT / dir_b / "features.csv", dtype={"game_id": str}, usecols=["game_id", "inj_known", "h_inj_gs", "a_inj_gs"])
    j = a.merge(b, on="game_id", suffixes=("_a", "_b")).merge(res, on="game_id").merge(feats, on="game_id")
    j = j[j.inj_known == 1]
    y = (j.home_pts > j.away_pts).to_numpy()
    m, t = (j.home_pts - j.away_pts).to_numpy(), (j.home_pts + j.away_pts).to_numpy()
    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    rows = []
    for split in ("valid", "test", "holdout", "all"):
        k = (j.split_a == split).to_numpy() if split != "all" else np.ones(len(j), bool)
        for name, da, db_ in (("log_loss", ll(j.p_home_a), ll(j.p_home_b)),
                              ("margin_abs_err", np.abs(m - j.pred_margin_a), np.abs(m - j.pred_margin_b)),
                              ("total_abs_err", np.abs(t - j.pred_total_a), np.abs(t - j.pred_total_b))):
            diff = (db_ - da)[k]
            rows.append({"split": split, "metric": name, "games": int(k.sum()), label_a: da[k].mean(), label_b: db_[k].mean(),
                         "diff_b_minus_a": diff.mean(), "ci": boot_ci(diff)})
        # same games: market closing line, for reference
        mk = k & j.close_mkt_p_home_a.notna().to_numpy()
        rows.append({"split": split, "metric": "log_loss_market", "games": int(mk.sum()), label_a: ll(j.close_mkt_p_home_a)[mk].mean()})
        rows.append({"split": split, "metric": "margin_abs_err_market", "games": int(k.sum()),
                     label_a: np.abs(m + j.close_home_spread_a)[k].mean()})
    out = pd.DataFrame(rows)
    out[["ci_lo", "ci_hi"]] = pd.DataFrame([r if isinstance(r, tuple) else (np.nan, np.nan) for r in out.pop("ci")])
    out.to_csv(ROOT / dir_b / f"compare_{label_a}_{label_b}.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare(*sys.argv[2:])
    else:
        main()
