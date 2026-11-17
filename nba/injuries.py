"""Official NBA injury reports (2021-22 onward) -> point-in-time injury features.

  python -m nba.injuries          # download + parse reports, player logs; writes DB tables

Timing rule: a game uses the latest report whose printed publish time is <= tip - 60 min.
Player value = EWMA over the TEAM's games of the player's Game Score (0 when he did not play),
using only box scores available by 09:00 ET on game day. A player who has been out for weeks has
already decayed toward 0, so features measure *new* absences the team ratings have not absorbed.
"""
import re
import sqlite3
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import DB_PATH, LAST_SEASON, RAW, season_str
from .ingest import ESPN_SB, ET, RESULT_LAG, get

URL_OLD = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{h:02d}{ap}.pdf"
URL_NEW = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{h:02d}_{m:02d}{ap}.pdf"
FIRST_INJ_SEASON = 2022          # reports available from 2021-22
PLAYER_FIRST_SEASON = 2021       # one warm-up season of player values
LEAD = timedelta(minutes=60)
HALFLIFE = 10.0                  # team games
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")
TEAM_NAMES = {
    "atlantahawks": 1610612737, "bostonceltics": 1610612738, "brooklynnets": 1610612751,
    "charlottehornets": 1610612766, "chicagobulls": 1610612741, "clevelandcavaliers": 1610612739,
    "dallasmavericks": 1610612742, "denvernuggets": 1610612743, "detroitpistons": 1610612765,
    "goldenstatewarriors": 1610612744, "houstonrockets": 1610612745, "indianapacers": 1610612754,
    "laclippers": 1610612746, "losangelesclippers": 1610612746, "losangeleslakers": 1610612747,
    "memphisgrizzlies": 1610612763, "miamiheat": 1610612748, "milwaukeebucks": 1610612749,
    "minnesotatimberwolves": 1610612750, "neworleanspelicans": 1610612740, "newyorkknicks": 1610612752,
    "oklahomacitythunder": 1610612760, "orlandomagic": 1610612753, "philadelphia76ers": 1610612755,
    "phoenixsuns": 1610612756, "portlandtrailblazers": 1610612757, "sacramentokings": 1610612758,
    "sanantoniospurs": 1610612759, "torontoraptors": 1610612761, "utahjazz": 1610612762,
    "washingtonwizards": 1610612764}
# team names match with or without spaces ("Milwaukee Bucks" in 2021-23 reports, "MilwaukeeBucks" later)
TEAM_RE = re.compile("|".join("".join(re.escape(c) + r"\s*" for c in n)
                              for n in sorted(TEAM_NAMES, key=len, reverse=True)), re.I)
PREFIX_RE = re.compile(r"\d{2}/\d{2}/\d{4}|\d{2}:\d{2}\s*\(ET\)|\b[A-Z]{2,3}@[A-Z]{2,3}\b")
# player = everything (incl. spaces, suffix commas) up to the status word: "Brown Jr., Troy Questionable"
PLAYER_RE = re.compile(r"^\s*([^\W\d_][\w'.\- ]*?,[\w'.,\- ]*?)\s*(" + "|".join(STATUSES) + r")\b")  # \w: accents (López)
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")
HEAD_RE = re.compile(r"Injury Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2})\s*(AM|PM)")


SUFFIX = re.compile(r"(jr|sr|ii|iii|iv)$")


def norm(name):
    """Lowercase letters only, accents and generational suffixes removed."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def report_key(raw):
    """'BullockJr.,Reggie' / 'Udeh,Jr.,Ernest' -> 'reggiebullockjr' / 'ernestudehjr'."""
    parts = [SUFFIX.sub("", norm(x)) for x in raw.split(",")]   # 'BullockJr.' -> 'bullock'
    return parts[-1] + "".join(parts[:-1])


# ------------------------------------------------------------------ download
def slot_urls(date, hour):
    """(url, expected publish time) candidates for `hour` (24h), newest first.
    Old-format files 'HHPM' are stamped HH:30; new-format files carry their minute. Format changed ~Jan 2026."""
    ap, h12 = ("AM", hour) if hour < 12 else ("PM", hour - 12 if hour > 12 else 12)
    t = lambda m: pd.Timestamp(f"{date} {hour:02d}:{m:02d}")
    old = (URL_OLD.format(d=date, h=h12, ap=ap), t(30))
    new = [(URL_NEW.format(d=date, h=h12, m=m, ap=ap), t(m)) for m in (30, 0)]
    return [old, *new] if date < "2026-01-01" else [*new, old]


def fetch_pdf(url):
    p = RAW / "injury_reports" / url.rsplit("/", 1)[1]
    if p.exists():
        return p if p.stat().st_size > 1000 else None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            break
        except requests.RequestException:
            time.sleep(2 ** attempt)
    else:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(r.content if r.status_code == 200 else b"")   # empty marker = not published
    return p if r.status_code == 200 else None


def parse_pdf(path):
    import pdfplumber
    rows, published, gdate, team = [], None, None, None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").splitlines():
                h = HEAD_RE.search(line)
                if h and published is None:
                    published = datetime.strptime(f"{h[1]} {h[2]} {h[3]}", "%m/%d/%y %I:%M %p")
                d = DATE_RE.search(line)
                if d:
                    gdate = datetime.strptime(d[1], "%m/%d/%Y").strftime("%Y-%m-%d")
                cut = 0
                for m in PREFIX_RE.finditer(line):
                    cut = max(cut, m.end())
                t = TEAM_RE.search(line, cut)
                if t:
                    team = TEAM_NAMES[re.sub(r"\s", "", t[0]).lower()]
                    cut = t.end()
                m = PLAYER_RE.match(line[cut:])
                if m and gdate and team:
                    rows.append((gdate, team, m[1].strip(), report_key(m[1]), m[2]))
    df = pd.DataFrame(rows, columns=["game_date", "team_id", "player_raw", "player_key", "status"])
    df["published"] = published.strftime("%Y-%m-%d %H:%M") if published else None
    df["file"] = path.name
    return df


def tip_times(schedule):
    """Tip time (ET) per game: schedule.tip_utc, else ESPN scoreboard (cached), else 19:00 ET."""
    s = schedule[schedule.season >= FIRST_INJ_SEASON].copy()
    need = sorted(s[s.tip_utc.isna()].game_date.unique())
    tips = {}
    for d in need:
        if d >= datetime.now(ET).strftime("%Y-%m-%d"):
            continue
        import json
        j = json.loads(get(ESPN_SB.format(d.replace("-", "")), RAW / "espn" / "scoreboard" / f"{d}.json"))
        for e in j.get("events", []):
            c = e["competitions"][0]
            ab = sorted(x["team"]["abbreviation"].lower() for x in c["competitors"])
            tips[(d, tuple(ab))] = e["date"]
    from .config import TEAM_CODES
    code = {tid: cs[0] for tid, cs in TEAM_CODES.items()}
    alias = {"gsw": "gs", "nop": "no", "nyk": "ny", "sas": "sa", "uta": "utah", "was": "wsh"}
    def lookup(r):
        if isinstance(r.tip_utc, str):
            return r.tip_utc
        ab = tuple(sorted(alias.get(code[x], code[x]) for x in (r.home_id, r.away_id)))
        return tips.get((r.game_date, ab))
    s["tip_utc"] = s.apply(lookup, axis=1)
    tip = pd.to_datetime(s.tip_utc, utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    s["tip_et"] = tip.fillna(pd.to_datetime(s.game_date) + pd.Timedelta(hours=19))
    s["tip_known"] = tip.notna()
    return s[["game_id", "game_date", "home_id", "away_id", "tip_et", "tip_known"]]


def choose_reports(tips):
    """For each game, the newest published report <= tip - LEAD (walk back hour by hour)."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    tips = tips[tips.game_date <= today]
    deadlines = tips.assign(deadline=tips.tip_et - LEAD)
    jobs = {}
    for d, g in deadlines.groupby("game_date"):
        for dl in sorted(g.deadline.unique()):
            jobs[(d, pd.Timestamp(dl))] = None
    print(f"{len(jobs)} (date, deadline) slots", file=sys.stderr)

    def resolve(key):
        d, dl = key
        hour = dl.hour
        for back in range(0, 8):
            h = hour - back
            if h < 8:
                break
            for url, pub in slot_urls(d, h):
                if pub > dl:
                    continue
                p = fetch_pdf(url)
                if p is None:
                    continue
                return url, p
        return None, None
    with ThreadPoolExecutor(8) as ex:
        found = dict(zip(jobs, ex.map(resolve, jobs)))
    return deadlines, found


# ------------------------------------------------------------------ player values
def player_logs():
    from nba_api.stats.endpoints import leaguegamelog
    frames = []
    for s in range(PLAYER_FIRST_SEASON, LAST_SEASON + 1):
        p = RAW / "nba_playerlog" / f"{s}.csv"
        if not p.exists() or s == LAST_SEASON:
            p.parent.mkdir(parents=True, exist_ok=True)
            leaguegamelog.LeagueGameLog(season=season_str(s), season_type_all_star="Regular Season",
                                        player_or_team_abbreviation="P", timeout=60).get_data_frames()[0].to_csv(p, index=False)
            time.sleep(1)
        frames.append(pd.read_csv(p, dtype={"GAME_ID": str}))
    pl = pd.concat(frames, ignore_index=True)
    pl["gs"] = (pl.PTS + 0.4 * pl.FGM - 0.7 * pl.FGA - 0.4 * (pl.FTA - pl.FTM) + 0.7 * pl.OREB + 0.3 * pl.DREB
                + pl.STL + 0.7 * pl.AST + 0.7 * pl.BLK - 0.4 * pl.PF - pl.TOV)
    pl["player_key"] = pl.PLAYER_NAME.map(norm)
    return pl


def player_values(pl, team_games, dates):
    """Snapshot {date: {player_key: (val_gs, val_min)}} as of each date's 09:00 cutoff.

    team_games: (game_id, team_id, game_date) for every team-game. Box scores become available
    RESULT_LAG after the game date, i.e. before the next day's 09:00 cutoff, so a snapshot for D
    includes exactly the games dated < D.
    """
    d = 0.5 ** (1 / HALFLIFE)
    played = {(r.GAME_ID, r.TEAM_ID): [] for r in pl[["GAME_ID", "TEAM_ID"]].drop_duplicates().itertuples()}
    for r in pl[["GAME_ID", "TEAM_ID", "player_key", "gs", "MIN"]].itertuples(index=False):
        played[(r.GAME_ID, r.TEAM_ID)].append((r.player_key, r.gs, r.MIN))
    state = {}                      # player_key -> [team, gs_ewma, min_ewma]
    roster = {}                     # team -> set of player keys currently attached
    snaps, tg = {}, team_games.sort_values(["game_date", "game_id"])
    by_date = {k: v for k, v in tg.groupby("game_date")}
    want = set(dates)
    for day in sorted(set(by_date) | want):
        if day in want:
            snaps[day] = {k: (v[1], v[2]) for k, v in state.items()}
        if day not in by_date:
            continue
        for r in by_date[day].itertuples():
            for k in roster.get(r.team_id, ()):  # decay everyone attached to this team
                v = state[k]; v[1] *= d; v[2] *= d
            for key, gs, mn in played.get((r.game_id, r.team_id), []):
                v = state.setdefault(key, [r.team_id, 0.0, 0.0])
                if v[0] != r.team_id:           # traded: value carries, roster membership moves
                    roster.get(v[0], set()).discard(key)
                    v[0] = r.team_id
                roster.setdefault(r.team_id, set()).add(key)
                v[1] += (1 - d) * gs; v[2] += (1 - d) * mn
    return snaps


def main():
    con = sqlite3.connect(DB_PATH)
    sched = pd.read_sql("select * from schedule", con)
    tb = pd.read_sql("select game_id, team_id, game_date from team_box", con)
    tips = tip_times(sched)
    deadlines, found = choose_reports(tips)
    files = sorted({p for u, p in found.values() if p is not None})
    print(f"{len(files)} report PDFs to parse; {sum(v[0] is None for v in found.values())} slots without a report",
          file=sys.stderr)
    with ProcessPoolExecutor() as ex:
        parsed = pd.concat(list(ex.map(parse_pdf, files, chunksize=8)), ignore_index=True)
    # attach chosen report per game; enforce publish time <= deadline (else treat as missing)
    file_of = {k: (p.name if p is not None else None) for k, (u, p) in found.items()}
    deadlines["file"] = [file_of.get((r.game_date, pd.Timestamp(r.deadline))) for r in deadlines.itertuples()]
    pub = parsed.groupby("file").published.first()
    deadlines["published"] = deadlines.file.map(pub)
    late = pd.to_datetime(deadlines.published) > deadlines.deadline
    deadlines.loc[late, ["file", "published"]] = None
    print(f"games with a valid report: {deadlines.file.notna().sum()} / {len(deadlines)} "
          f"(late reports discarded: {int(late.sum())})", file=sys.stderr)

    # entries for each game's two teams from its chosen report
    gm = deadlines.melt(id_vars=["game_id", "game_date", "file", "published", "tip_et"],
                        value_vars=["home_id", "away_id"], value_name="team_id").drop(columns="variable")
    ent = gm.merge(parsed.drop(columns="published"), on=["file", "game_date", "team_id"], how="inner")

    pl = player_logs()
    snaps = player_values(pl, tb, sorted(deadlines.game_date.unique()))
    def val(r):
        return snaps.get(r.game_date, {}).get(r.player_key, (0.0, 0.0))
    v = ent.apply(val, axis=1, result_type="expand")
    ent["val_gs"], ent["val_min"] = v[0], v[1]
    ent["matched"] = ent.apply(lambda r: r.player_key in snaps.get(r.game_date, {}), axis=1)
    hist_keys = set(pl.player_key)
    known = ent[ent.status.isin(["Out", "Doubtful"]) & ent.player_key.isin(hist_keys)]
    print(f"injury rows: {len(ent)}; Out/Doubtful players with NBA history matched to team value: "
          f"{known.matched.mean():.1%}", file=sys.stderr)
    unmatched = ent[ent.status.isin(["Out", "Doubtful"]) & ~ent.player_key.isin(hist_keys)].player_raw.value_counts()
    print("most frequent Out/Doubtful names with no NBA game log (two-way/G-League or name mismatch):",
          unmatched.head(15).to_dict(), file=sys.stderr)

    ent["available_at"] = ent.published
    ent.drop(columns=["tip_et"]).to_sql("injury_entries", con, if_exists="replace", index=False)
    deadlines.assign(tip_et=deadlines.tip_et.astype(str), deadline=deadlines.deadline.astype(str)) \
        [["game_id", "game_date", "tip_et", "tip_known", "deadline", "file", "published"]] \
        .to_sql("injury_report_choice", con, if_exists="replace", index=False)
    con.close()


def live_entries(games, date):
    """Latest report published <= tip - LEAD for today's games, valued with today's snapshot.
    games: game_id, home_id, away_id, tip_et (naive ET)."""
    now = datetime.now(ET).replace(tzinfo=None)
    ch, parsed = [], {}
    for g in games.itertuples():
        dl = min(g.tip_et - LEAD, now)
        found = None
        for back in range(0, 8):
            h = dl.hour - back
            for url, pub_exp in slot_urls(date, h) if h >= 8 else []:
                if pub_exp > dl:
                    continue
                name = url.rsplit("/", 1)[1]
                f = RAW / "injury_reports" / name
                if f.exists() and f.stat().st_size == 0:
                    f.unlink()                      # 'not published' marker may be stale today
                path = fetch_pdf(url)
                if path is None:
                    continue
                parsed.setdefault(name, parse_pdf(path))
                pub = parsed[name].published.iloc[0] if len(parsed[name]) else None
                if pub and pd.Timestamp(pub) <= dl:
                    found = (name, pub)
                    break
            if found:
                break
        ch.append({"game_id": g.game_id, "file": found and found[0], "published": found and found[1],
                   "tip_et": str(g.tip_et)})
    ch = pd.DataFrame(ch)
    rows = []
    for c in ch[ch.file.notna()].itertuples():
        g = games.set_index("game_id").loc[c.game_id]
        e = parsed[c.file]
        e = e[(e.game_date == date) & e.team_id.isin([g.home_id, g.away_id])]
        rows.append(e.assign(game_id=c.game_id, published=c.published))
    ent = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["game_id", "team_id", "player_key", "status"])
    con = sqlite3.connect(DB_PATH)
    tb = pd.read_sql("select game_id, team_id, game_date from team_box", con)
    con.close()
    snap = player_values(player_logs(), tb, [date])[date]
    v = ent.player_key.map(lambda k: snap.get(k, (0.0, 0.0)))
    ent["val_gs"], ent["val_min"] = [x[0] for x in v], [x[1] for x in v]
    return ent, ch


def injury_features(ent=None, ch=None, sched=None):
    """Per game: home/away value of players listed Out / Doubtful / Questionable."""
    if ent is None:
        con = sqlite3.connect(DB_PATH)
        ent = pd.read_sql("select * from injury_entries", con)
        ch = pd.read_sql("select game_id, published, tip_et from injury_report_choice", con)
        sched = pd.read_sql("select game_id, home_id, away_id from schedule", con)
        con.close()
    ch = ch[["game_id", "published", "tip_et"]]
    ent = ent.drop(columns=[c for c in ("home_id", "away_id") if c in ent]).merge(sched, on="game_id")
    ent["side"] = (ent.team_id == ent.home_id).map({True: "h", False: "a"})
    rows = []
    for st, tag in (("Out", "out"), ("Doubtful", "doubt"), ("Questionable", "quest")):
        e = ent[ent.status == st]
        g = e.groupby(["game_id", "side"]).agg(gs=("val_gs", "sum"), mn=("val_min", "sum")).unstack("side")
        g.columns = [f"{s}_{tag}_{k}" for k, s in g.columns]
        rows.append(g)
    f = pd.concat(rows, axis=1)
    # a status with no entries on one side (e.g. no home Doubtful tonight) must still exist as zeros
    cols = [f"{s}_{tag}_{k}" for tag in ("out", "doubt", "quest") for s in ("h", "a") for k in ("gs", "mn")]
    f = f.reindex(columns=cols, fill_value=0.0)
    f = ch.set_index("game_id").join(f).fillna({c: 0.0 for c in cols})
    f["inj_known"] = f.published.notna().astype(int)
    f = f.rename(columns={"published": "inj_published"})
    for side in ("h", "a"):
        # ponytail: fixed status weights (Doubtful 0.75, Questionable 0.4); components also kept as features
        f[f"{side}_inj_gs"] = f[f"{side}_out_gs"] + 0.75 * f[f"{side}_doubt_gs"] + 0.4 * f[f"{side}_quest_gs"]
    f["d_inj_gs"] = f.h_inj_gs - f.a_inj_gs
    return f.reset_index()


if __name__ == "__main__":
    main()
