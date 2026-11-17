"""Live predictions for one date with the frozen model, priced against current DraftKings lines.

  python -m nba.today [YYYY-MM-DD]

Same feature engine, same registry config, same bet rule as the backtest. Results known by the
09:00 ET cutoff are the only training data. Lines are ESPN/DraftKings "current" at run time.
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from .backtest import available_results, entry_features, load_db, recommend, registry
from .config import CODE_TO_ID, LAST_SEASON, OUT, RAW
from .evaluate import settle
from .features import cutoff_for, feature_columns
from .ingest import ESPN_ODDS, ESPN_SB, ET, _espn_event_odds  # noqa: F401
from .models import Model


def fetch_today(date):
    stamp = datetime.now(ET).strftime("%Y%m%d-%H%M")
    j = requests.get(ESPN_SB.format(date.replace("-", "")), timeout=60).json()
    rows = []
    for e in j.get("events", []):
        c = e["competitions"][0]
        if c["status"]["type"]["state"] != "pre":
            continue                                   # started or finished: not a pregame decision
        t = {x["homeAway"]: x["team"]["abbreviation"] for x in c["competitors"]}
        o = requests.get(ESPN_ODDS.format(e["id"]), timeout=60).json()
        snap = RAW / "espn" / "live" / f"{e['id']}-{stamp}.json"     # keep what we priced against
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(json.dumps(o))
        items = [i for i in o.get("items", []) if "Live" not in i["provider"]["name"]]
        it = next((i for i in items if i["provider"]["name"].replace(" ", "").lower() == "draftkings"), None)
        h, a = (it or {}).get("homeTeamOdds", {}), (it or {}).get("awayTeamOdds", {})
        rows.append({"game_id": f"live_{e['id']}", "season": LAST_SEASON, "game_date": date,
                     "home_id": CODE_TO_ID[t["home"].lower()], "away_id": CODE_TO_ID[t["away"].lower()],
                     "neutral": int(bool(c.get("neutralSite"))), "tip_utc": e["date"],
                     "matchup": f"{t['away']} @ {t['home']}", "lines_at": stamp,
                     "now_home_spread": (it or {}).get("spread"), "now_total": (it or {}).get("overUnder"),
                     "now_ml_home": h.get("moneyLine"), "now_ml_away": a.get("moneyLine"),
                     "now_spread_price_home": h.get("spreadOdds"), "now_spread_price_away": a.get("spreadOdds"),
                     "now_over_price": (it or {}).get("overOdds"), "now_under_price": (it or {}).get("underOdds")})
    return pd.DataFrame(rows)


def history_by_edge(entry):
    """Backtest ROI by EV bucket on test + holdout seasons (locked predictions, closing lines)."""
    pred = pd.read_csv(OUT / "predictions.csv", dtype={"game_id": str})
    res = pd.read_csv(OUT / "evaluation.csv", dtype={"game_id": str})[["game_id", "home_pts", "away_pts"]]
    ev = pred[pred.split.isin(["test", "holdout"])].merge(res, on="game_id")
    s = settle(ev, "close_").merge(ev[["game_id", "close_ml_ev", "close_spread_ev", "close_total_ev"]], on="game_id")
    s["ev"] = np.select([s.market == "ml", s.market == "spread"], [s.close_ml_ev, s.close_spread_ev], s.close_total_ev)
    s["edge_bucket"] = pd.cut(s.ev, [-1, 0.03, 0.06, 0.10, 0.15, 10], labels=["<3%", "3-6%", "6-10%", "10-15%", ">15%"])
    return s.groupby(["market", "edge_bucket"], observed=True).agg(
        bets=("profit", "size"), win_rate=("win", "mean"), roi=("profit", "mean")).reset_index()


def main(date=None):
    date = date or datetime.now(ET).strftime("%Y-%m-%d")
    entry = [r for r in registry() if r["stage"] == "final"][-1]
    db = load_db()
    live = fetch_today(date)
    if live.empty:
        print(f"No games before tip on {date}.")
        return
    sched = pd.concat([db["schedule"][db["schedule"].game_date < date],
                       live[["game_id", "season", "game_date", "home_id", "away_id", "neutral", "tip_utc"]]],
                      ignore_index=True)
    inj = None
    if entry.get("injuries"):
        from .injuries import injury_features, live_entries
        g = live.assign(tip_et=pd.to_datetime(live.tip_utc, utc=True).dt.tz_convert(ET).dt.tz_localize(None))
        ent, ch = live_entries(g[["game_id", "home_id", "away_id", "tip_et"]], date)
        inj = pd.concat([injury_features(), injury_features(ent, ch, live[["game_id", "home_id", "away_id"]])],
                        ignore_index=True)
        live = live.merge(inj[["game_id", "inj_published", "h_inj_gs", "a_inj_gs"]], on="game_id", how="left")
        ent[ent.status.isin(["Out", "Doubtful", "Questionable"]) & (ent.val_gs > 1)] \
            .sort_values("val_gs", ascending=False)[["game_id", "team_id", "player_raw", "status", "val_gs", "val_min"]] \
            .to_csv(OUT / f"today_{date}_injuries.csv", index=False)
    feats = entry_features(db, entry, schedule=sched, injuries=inj)
    cols = feature_columns(feats)
    cutoff = cutoff_for(date)
    lab = available_results(db["results"], cutoff)
    tr = feats.merge(lab, on="game_id")
    tr = tr[~((tr.season == feats.season.min()) & (tr[["h_gp", "a_gp"]].min(axis=1) < 5))]
    assert tr.game_date.max() < date
    te = feats[feats.game_date == date].set_index("game_id").loc[live.game_id]
    preds = []
    for name in entry["ensemble_members"]:
        m = Model(name, entry["hyperparams"]).fit(tr[cols], (tr.home_pts > tr.away_pts).astype(int),
                                                  tr.home_pts - tr.away_pts, tr.home_pts + tr.away_pts)
        preds.append(np.column_stack(m.predict(te[cols])))
    p = np.mean(preds, axis=0)
    out = live.set_index("game_id")
    out["p_home"], out["pred_margin"], out["pred_total"] = p[:, 0], p[:, 1], p[:, 2]
    out["pred_home_pts"] = (out.pred_total + out.pred_margin) / 2
    out["pred_away_pts"] = (out.pred_total - out.pred_margin) / 2
    out["model_version"], out["cutoff"], out["trained_through"] = entry["version"], cutoff, tr.game_date.max()
    rec = recommend(out, out, "now_", entry["bet_config"])
    out = pd.concat([out, rec], axis=1)
    path = OUT / f"today_{date}.csv"
    out.reset_index().to_csv(path, index=False)

    cfg = entry["bet_config"]
    print(f"{date}  model {entry['version']}  trained through {tr.game_date.max()}  lines at {out.lines_at.iloc[0]} ET")
    print(f"bet thresholds (EV): ML {cfg['thr_ml']:.0%}  spread {cfg['thr_spread']:.0%}  total {cfg['thr_total']:.0%}\n")
    for r in out.itertuples():
        print(f"{r.matchup:12s} model {r.pred_away_pts:5.1f}-{r.pred_home_pts:5.1f}  P(home) {r.p_home:.1%} "
              f"(mkt {r.now_mkt_p_home:.1%})")
        if "inj_published" in out:
            print(f"   injuries (report {r.inj_published}): missing value away {r.a_inj_gs:.1f}, home {r.h_inj_gs:.1f} Game Score")
        print(f"   spread {r.now_home_spread:+5.1f}  model {-r.pred_margin:+5.1f}  cover(home) {r.now_cover_p_home:.1%}  "
              f"-> {r.now_spread_bet:5s} EV {r.now_spread_ev:+.1%}")
        print(f"   total  {r.now_total:5.1f}  model {r.pred_total:5.1f}  over {r.now_over_p:.1%}  "
              f"-> {r.now_total_bet:5s} EV {r.now_total_ev:+.1%}")
        print(f"   ML {r.now_ml_away:+.0f}/{r.now_ml_home:+.0f}  -> {r.now_ml_bet:5s} EV {r.now_ml_ev:+.1%}")
    from .track import LOG, log_picks
    n, skipped = log_picks(out.reset_index(), entry["version"])
    print(f"\nforward log: {n} new picks recorded in {LOG.relative_to(LOG.parent.parent)}"
          + (f"; skipped {', '.join(skipped)}" if skipped else ""))
    h = history_by_edge(entry)
    h.to_csv(OUT / "roi_by_edge.csv", index=False)
    print("\nBacktest (test+holdout, closing lines): ROI by model-estimated edge")
    print(h.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nsaved", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
