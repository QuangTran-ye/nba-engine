"""Data ingestion -> normalized SQLite.

Sources
  * NBA.com stats (nba_api LeagueGameLog): schedule + team box scores, 2007-08 .. today.
  * Kaggle "NBA betting data Oct 2007 - Jun 2026" (SBR-derived): one spread/total/ML per game.
  * ESPN core API (DraftKings book): open + close spread/total/ML, 2022-23 .. today.

Every table separates *when it happened* from *when it became knowable*:
  schedule.known_at, team_box.available_at, results.available_at, odds.observed_at/line_type.
All timestamps are naive US/Eastern "YYYY-MM-DD HH:MM".
"""
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import CODE_TO_ID, DB_PATH, FIRST_SEASON, LAST_SEASON, RAW, season_str

ET = ZoneInfo("America/New_York")
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024"
ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}"
ESPN_ODDS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/odds"
ESPN_FIRST_SEASON = 2023
RESULT_LAG = timedelta(days=1, hours=6)   # box score treated as known 06:00 ET the next day

SCHEMA = """
CREATE TABLE IF NOT EXISTS schedule(
  game_id TEXT PRIMARY KEY, season INT, game_date TEXT, home_id INT, away_id INT,
  tip_utc TEXT, known_at TEXT, neutral INT);
CREATE TABLE IF NOT EXISTS team_box(
  game_id TEXT, team_id INT, opp_id INT, is_home INT, game_date TEXT,
  pts INT, fgm INT, fga INT, fg3m INT, fg3a INT, ftm INT, fta INT, oreb INT, dreb INT,
  ast INT, stl INT, blk INT, tov INT, pf INT, minutes INT, available_at TEXT,
  PRIMARY KEY(game_id, team_id));
CREATE TABLE IF NOT EXISTS results(
  game_id TEXT PRIMARY KEY, home_pts INT, away_pts INT, available_at TEXT);
CREATE TABLE IF NOT EXISTS odds(
  game_id TEXT, source TEXT, book TEXT, line_type TEXT, observed_at TEXT,
  home_spread REAL, total REAL, ml_home REAL, ml_away REAL,
  spread_price_home REAL, spread_price_away REAL, over_price REAL, under_price REAL,
  PRIMARY KEY(game_id, source, line_type));
"""


def get(url, path, timeout=60):
    """GET with on-disk cache so every run is reproducible from data/raw."""
    if path.exists():
        return path.read_bytes()
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return r.content
        except requests.RequestException as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch_gamelogs():
    from nba_api.stats.endpoints import leaguegamelog
    frames = []
    for s in range(FIRST_SEASON, LAST_SEASON + 1):
        p = RAW / "nba_gamelog" / f"{s}.csv"
        if not p.exists() or s == LAST_SEASON:          # current season: always refresh
            p.parent.mkdir(parents=True, exist_ok=True)
            df = leaguegamelog.LeagueGameLog(season=season_str(s), season_type_all_star="Regular Season",
                                             player_or_team_abbreviation="T", timeout=60).get_data_frames()[0]
            df.to_csv(p, index=False)
            time.sleep(1)
        df = pd.read_csv(p, dtype={"GAME_ID": str})
        df["season"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_games(gl):
    # cancelled games appear with 0 minutes / 0 points (e.g. BOS-IND 2013-04-16, never played)
    cancelled = gl[gl.MIN == 0].GAME_ID.unique()
    if len(cancelled):
        print(f"dropping cancelled games {list(cancelled)}", file=sys.stderr)
    gl = gl[~gl.GAME_ID.isin(cancelled)].copy()
    gl["is_home"] = gl.MATCHUP.str.contains(" vs. ").astype(int)
    # neutral-site games (international, NBA Cup knockouts) list both teams as away: provisional
    # home = team named after "@" in the lower TEAM_ID's row; re-oriented from ESPN in main().
    hs = gl.groupby("GAME_ID").is_home.transform("sum")
    gl["neutral"] = (hs == 0).astype(int)
    nz = gl[gl.neutral == 1].sort_values("TEAM_ID").drop_duplicates("GAME_ID")
    gl.loc[gl.neutral == 1, "is_home"] = 0
    for r in nz.itertuples():
        gl.loc[(gl.GAME_ID == r.GAME_ID) & (gl.TEAM_ID != r.TEAM_ID), "is_home"] = 1
    pairs = gl.groupby("GAME_ID").size()
    bad = pairs[pairs != 2]
    if len(bad):
        print(f"dropping {len(bad)} games without exactly 2 team rows", file=sys.stderr)
        gl = gl[~gl.GAME_ID.isin(bad.index)]
    # both teams play the same game length; fixes source rows with misrecorded team minutes
    gl["MIN"] = gl.groupby("GAME_ID").MIN.transform("max")
    opp = gl[["GAME_ID", "TEAM_ID"]].rename(columns={"TEAM_ID": "opp_id"})
    tb = gl.merge(opp, on="GAME_ID")
    tb = tb[tb.TEAM_ID != tb.opp_id]
    avail = (pd.to_datetime(tb.GAME_DATE) + RESULT_LAG).dt.strftime("%Y-%m-%d %H:%M")
    team_box = pd.DataFrame({
        "game_id": tb.GAME_ID, "team_id": tb.TEAM_ID, "opp_id": tb.opp_id, "is_home": tb.is_home,
        "game_date": tb.GAME_DATE, "pts": tb.PTS, "fgm": tb.FGM, "fga": tb.FGA, "fg3m": tb.FG3M,
        "fg3a": tb.FG3A, "ftm": tb.FTM, "fta": tb.FTA, "oreb": tb.OREB, "dreb": tb.DREB, "ast": tb.AST,
        "stl": tb.STL, "blk": tb.BLK, "tov": tb.TOV, "pf": tb.PF, "minutes": tb.MIN, "available_at": avail})
    h = team_box[team_box.is_home == 1].set_index("game_id")
    a = team_box[team_box.is_home == 0].set_index("game_id")
    seasons = gl.drop_duplicates("GAME_ID").set_index("GAME_ID").season
    schedule = pd.DataFrame({
        "game_id": h.index, "season": seasons.loc[h.index].values, "game_date": h.game_date.values,
        "home_id": h.team_id.values, "away_id": a.loc[h.index].team_id.values, "tip_utc": None,
        "neutral": gl.drop_duplicates("GAME_ID").set_index("GAME_ID").neutral.loc[h.index].values})
    # ponytail: schedule assumed public Aug 1 before the season; rescheduled dates are taken as final.
    schedule["known_at"] = (schedule.season - 1).astype(str) + "-08-01 00:00"
    results = pd.DataFrame({"game_id": h.index, "home_pts": h.pts.values,
                            "away_pts": a.loc[h.index].pts.values, "available_at": h.available_at.values})
    return schedule, team_box, results


def load_kaggle(schedule):
    d = RAW / "kaggle_odds"
    if not (d / "nba_2008-2026.csv").exists():
        import io, zipfile
        zipfile.ZipFile(io.BytesIO(get(KAGGLE_URL, d / "odds.zip"))).extractall(d)
    k = pd.read_csv(d / "nba_2008-2026.csv")
    k = k[k.regular == True].copy()
    k["home_id"] = k.home.str.lower().map(CODE_TO_ID)
    k["away_id"] = k.away.str.lower().map(CODE_TO_ID)
    m = k.merge(schedule[["game_id", "game_date", "home_id", "away_id"]],
                left_on=["date", "home_id", "away_id"], right_on=["game_date", "home_id", "away_id"])
    print(f"kaggle odds: {len(k)} regular-season rows, {len(m)} matched to schedule", file=sys.stderr)
    # dataset gives the favourite and an unsigned spread; convert to the home team's line
    home_spread = m.spread.where(m.whos_favored == "away", -m.spread)
    return pd.DataFrame({
        "game_id": m.game_id, "source": "kaggle_sbr", "book": "consensus/unknown",
        "line_type": "close_assumed", "observed_at": None,
        "home_spread": home_spread, "total": m.total, "ml_home": m.moneyline_home, "ml_away": m.moneyline_away,
        "spread_price_home": None, "spread_price_away": None, "over_price": None, "under_price": None})


def _num(x):
    if x is None: return None
    if isinstance(x, dict): x = x.get("american", x.get("value"))
    try: return float(str(x).replace("+", "")) if str(x).upper() not in ("EVEN", "OFF", "") else (100.0 if str(x).upper() == "EVEN" else None)
    except ValueError: return None


def _espn_event_odds(eid):
    j = json.loads(get(ESPN_ODDS.format(eid), RAW / "espn" / "odds" / f"{eid}.json"))
    items = [i for i in j.get("items", []) if "Live" not in i["provider"]["name"]]
    pref = [i for i in items if i["provider"]["name"].replace(" ", "").lower() == "draftkings"] or items
    return pref[0] if pref else None


def fetch_espn(schedule):
    sched = schedule[schedule.season >= ESPN_FIRST_SEASON]
    dates = sorted(sched.game_date.unique())
    today = datetime.now(ET).strftime("%Y-%m-%d")
    events = []
    for d in dates:
        p = RAW / "espn" / "scoreboard" / f"{d}.json"
        if d >= today and p.exists(): p.unlink()
        j = json.loads(get(ESPN_SB.format(d.replace("-", "")), p))
        for e in j.get("events", []):
            c = e["competitions"][0]
            if not c["status"]["type"].get("completed"):
                continue
            t = {x["homeAway"]: x["team"]["abbreviation"].lower() for x in c["competitors"]}
            events.append({"eid": e["id"], "tip_utc": e["date"], "home_id": CODE_TO_ID.get(t["home"]),
                           "away_id": CODE_TO_ID.get(t["away"])})
    ev = pd.DataFrame(events)
    ev["game_date"] = pd.to_datetime(ev.tip_utc).dt.tz_convert(ET).dt.strftime("%Y-%m-%d")
    pair = lambda d: d.home_id.astype(str).str.cat(d.away_id.astype(str), sep="-").where(
        d.home_id < d.away_id, d.away_id.astype(str).str.cat(d.home_id.astype(str), sep="-"))
    ev["pair"], sc = pair(ev), sched.assign(pair=pair(sched))
    ev = ev.merge(sc[["game_id", "game_date", "pair"]], on=["game_date", "pair"])
    print(f"espn: {len(ev)} completed events matched of {len(sched)} scheduled", file=sys.stderr)
    with ThreadPoolExecutor(6) as ex:
        items = list(ex.map(_espn_event_odds, ev.eid))
    rows = []
    for e, it in zip(ev.itertuples(), items):
        if it is None:
            continue
        h, a = it.get("homeTeamOdds", {}), it.get("awayTeamOdds", {})
        tip_et = pd.Timestamp(e.tip_utc).tz_convert(ET).strftime("%Y-%m-%d %H:%M")
        has_close = "close" in h
        rows.append({"game_id": e.game_id, "source": "espn", "book": it["provider"]["name"],
                     "line_type": "close" if has_close else "close_assumed", "observed_at": tip_et,
                     "home_spread": _num((h.get("close") or {}).get("pointSpread")) if has_close else it.get("spread"),
                     "total": _num((it.get("close") or {}).get("total")) if has_close else it.get("overUnder"),
                     "ml_home": _num((h.get("close") or {}).get("moneyLine")) if has_close else h.get("moneyLine"),
                     "ml_away": _num((a.get("close") or {}).get("moneyLine")) if has_close else a.get("moneyLine"),
                     "spread_price_home": _num((h.get("close") or {}).get("spread")) if has_close else h.get("spreadOdds"),
                     "spread_price_away": _num((a.get("close") or {}).get("spread")) if has_close else a.get("spreadOdds"),
                     "over_price": _num((it.get("close") or {}).get("over")) if has_close else it.get("overOdds"),
                     "under_price": _num((it.get("close") or {}).get("under")) if has_close else it.get("underOdds")})
        if "open" in h and "open" in it:
            # ponytail: exact time of the opener is not published; it precedes tip by an unknown amount.
            rows.append({"game_id": e.game_id, "source": "espn", "book": it["provider"]["name"],
                         "line_type": "open", "observed_at": None,
                         "home_spread": _num(h["open"].get("pointSpread")), "total": _num(it["open"].get("total")),
                         "ml_home": _num(h["open"].get("moneyLine")), "ml_away": _num(a["open"].get("moneyLine")),
                         "spread_price_home": _num(h["open"].get("spread")), "spread_price_away": _num(a["open"].get("spread")),
                         "over_price": _num(it["open"].get("over")), "under_price": _num(it["open"].get("under"))})
    return pd.DataFrame(rows), ev[["game_id", "tip_utc", "home_id", "away_id"]]


def orient_neutral(schedule, team_box, results, espn):
    """Use ESPN's designated home team for neutral-site games."""
    e = espn.set_index("game_id")
    flip = [g for g in schedule[schedule.neutral == 1].game_id if g in e.index
            and e.home_id[g] != schedule.set_index("game_id").home_id[g]]
    sc = schedule[schedule.neutral == 0].set_index("game_id")
    both = sc.index.intersection(e.index)
    assert (sc.home_id[both] == e.home_id[both]).all(), "ESPN home team disagrees on a non-neutral game"
    missing = set(schedule[schedule.neutral == 1].game_id) - set(e.index)
    assert not missing, f"neutral games without ESPN orientation: {missing}"
    s = schedule.game_id.isin(flip)
    schedule.loc[s, ["home_id", "away_id"]] = schedule.loc[s, ["away_id", "home_id"]].values
    results.loc[results.game_id.isin(flip), ["home_pts", "away_pts"]] = \
        results.loc[results.game_id.isin(flip), ["away_pts", "home_pts"]].values
    t = team_box.game_id.isin(flip)
    team_box.loc[t, "is_home"] = 1 - team_box.loc[t, "is_home"]
    print(f"neutral games: {int(schedule.neutral.sum())}, re-oriented {len(flip)}", file=sys.stderr)


def main():
    gl = fetch_gamelogs()
    schedule, team_box, results = build_games(gl)
    odds_e, espn = fetch_espn(schedule)
    orient_neutral(schedule, team_box, results, espn)
    # 2020 Orlando bubble: no home court (designated home team kept)
    schedule.loc[(schedule.season == 2020) & (schedule.game_date >= "2020-07-30"), "neutral"] = 1
    odds_k = load_kaggle(schedule)
    schedule = schedule.drop(columns="tip_utc").merge(espn[["game_id", "tip_utc"]], on="game_id", how="left")
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    for name, df in [("schedule", schedule), ("team_box", team_box), ("results", results),
                     ("odds", pd.concat([odds_k, odds_e], ignore_index=True))]:
        df.to_sql(name, con, if_exists="append", index=False)
    con.commit()
    for t in ("schedule", "team_box", "results", "odds"):
        print(t, con.execute(f"select count(*) from {t}").fetchone()[0])
    con.close()


if __name__ == "__main__":
    main()
