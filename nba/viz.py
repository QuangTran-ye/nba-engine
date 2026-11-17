"""Charts (matplotlib PNG) from evaluation outputs.

  python -m nba.viz
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUT

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#6250d6"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2})
CH = OUT / "charts"


def save(fig, name):
    CH.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CH / name, dpi=150)
    plt.close(fig)


def calibration(ev, long):
    e = ev[ev.split.isin(["test", "holdout"])]
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", label="Perfect calibration")
    y = e.actual_margin > 0
    series = [("Model v1 (logistic)", e.p_home, y, C["blue"]),
              ("Market (no-vig close)", e.close_mkt_p_home, y, C["orange"])]
    el = long[(long.model == "elo") & long.game_id.isin(e.game_id)].merge(ev[["game_id", "actual_margin"]], on="game_id")
    series.append(("Elo", el.p_home, el.actual_margin > 0, C["aqua"]))
    rows = []
    for name, p, yy, col in series:
        ok = p.notna()
        b = pd.cut(p[ok], bins, include_lowest=True)
        g = pd.DataFrame({"p": p[ok], "y": yy[ok], "b": b}).groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        g = g[g.n >= 20]
        ax.plot(g.p, g.y, marker="o", ms=5, color=col, label=name)
        rows.append(g.assign(model=name).reset_index())
    ax.set(xlabel="Predicted home win probability", ylabel="Observed home win rate", xlim=(0, 1), ylim=(0, 1),
           title="Calibration, test + holdout (2022-23 to 2026-27)")
    ax.legend(loc="upper left")
    save(fig, "calibration.png")
    pd.concat(rows).astype({"b": str}).to_csv(OUT / "calibration_table.csv", index=False)


def bankroll(ledger):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for mk, col, lab in (("ml", C["blue"], "Moneyline"), ("spread", C["orange"], "Spread"), ("total", C["aqua"], "Total")):
        b = ledger[ledger.market == mk].sort_values(["game_date", "game_id"])
        x = pd.to_datetime(b.game_date)
        ax.plot(x, b.profit.cumsum(), color=col, label=f"{lab} ({len(b)} bets, {b.profit.sum():+.1f}u)")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set(ylabel="Cumulative units (1u flat stakes)", title="Simulated bankroll at closing lines, test + holdout")
    ax.legend(loc="lower left")
    save(fig, "bankroll.png")


def monthly(mb):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    months = sorted(mb.month.unique())
    for ax, (mk, col, lab) in zip(axes, (("ml", C["blue"], "Moneyline"), ("spread", C["orange"], "Spread"),
                                         ("total", C["aqua"], "Total"))):
        m = mb[mb.market == mk].set_index("month").reindex(months).fillna(0)
        ax.bar(range(len(months)), m.units, color=col, width=0.8)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_ylabel("Units")
        ax.set_title(lab, loc="left", fontsize=10)
    axes[-1].set_xticks(range(len(months)), months, rotation=90, fontsize=7)
    fig.suptitle("Monthly betting result, closing lines (test + holdout)", fontweight="bold")
    save(fig, "monthly_units.png")


def per_season(ps):
    ps = ps.sort_values("season")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lab = ps.season.map(lambda s: f"{str(s-1)[-2:]}-{str(s)[-2:]}")
    axes[0].plot(lab, ps.margin_mae, marker="o", ms=4, color=C["blue"], label="Model")
    axes[0].plot(lab, ps.mkt_margin_mae, marker="o", ms=4, color=C["orange"], label="Closing spread")
    axes[0].set(title="Point-differential MAE by season", ylabel="MAE (points)")
    axes[1].plot(lab, ps.total_mae, marker="o", ms=4, color=C["blue"], label="Model")
    axes[1].plot(lab, ps.mkt_total_mae, marker="o", ms=4, color=C["orange"], label="Closing total")
    axes[1].set(title="Game-total MAE by season", ylabel="MAE (points)")
    for ax in axes:
        ax.legend()
        ax.tick_params(axis="x", rotation=60, labelsize=8)
    save(fig, "per_season_mae.png")


def accuracy_by_bucket(ev):
    e = ev[ev.split.isin(["test", "holdout"])].copy()
    conf = np.maximum(e.p_home, 1 - e.p_home)
    e["bucket"] = pd.cut(conf, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0], include_lowest=True)
    g = e.groupby("bucket", observed=True).agg(n=("winner_correct", "size"), acc=("winner_correct", "mean"),
                                               mean_conf=("p_home", lambda s: np.maximum(s, 1 - s).mean()))
    g.astype(float).reset_index().astype({"bucket": str}).to_csv(OUT / "accuracy_by_bucket.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(g))
    ax.bar(x, g.acc, color=C["blue"], width=0.7, label="Observed accuracy")
    ax.plot(x, g.mean_conf, color=C["orange"], marker="o", ms=5, label="Mean predicted confidence")
    for i, (a, n) in enumerate(zip(g.acc, g.n)):
        ax.text(i, a + 0.01, f"n={n}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [str(b) for b in g.index], rotation=30, fontsize=8)
    ax.set(ylim=(0.4, 1.0), ylabel="Winner accuracy", xlabel="Model confidence bucket",
           title="Accuracy by confidence, test + holdout")
    ax.legend(loc="upper left")
    save(fig, "accuracy_by_bucket.png")


def importance(imp):
    top = imp.sort_values("lgbm_gain_share", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.feature, top.lgbm_gain_share, color=C["blue"], height=0.7)
    ax.set(xlabel="Share of LightGBM win-model gain", title="Top 20 features (fit through 2025-26)")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "feature_importance.png")


def main():
    ev = pd.read_csv(OUT / "evaluation.csv", dtype={"game_id": str})
    pred = pd.read_csv(OUT / "predictions.csv", dtype={"game_id": str})
    ev = ev.merge(pred[["game_id", "close_mkt_p_home"]], on="game_id")
    long = pd.read_csv(OUT / "predictions_all_models.csv", dtype={"game_id": str})
    calibration(ev, long)
    bankroll(pd.read_csv(OUT / "bets_ledger.csv"))
    monthly(pd.read_csv(OUT / "monthly_betting.csv"))
    per_season(pd.read_csv(OUT / "per_season.csv"))
    accuracy_by_bucket(ev)
    if (OUT / "feature_importance.csv").exists():
        importance(pd.read_csv(OUT / "feature_importance.csv"))
    dashboard()
    print("charts ->", CH, "| dashboard ->", OUT / "dashboard.html")


def dashboard():
    """Self-contained offline HTML dashboard (charts embedded as data URIs)."""
    import base64, html
    comp = pd.read_csv(OUT / "model_comparison.csv")
    bets = pd.read_csv(OUT / "betting_performance.csv")
    keep = ["ensemble", "rf", "lgbm", "elo", "baseline_home", "baseline_random", "market_moneyline", "market_close_lines"]
    names = {"ensemble": "Model v1 (logistic)", "rf": "Random forest", "lgbm": "LightGBM", "elo": "Elo",
             "baseline_home": "Always home", "baseline_random": "Random", "market_moneyline": "Market: closing ML",
             "market_close_lines": "Market: closing spread / total"}
    f = lambda v, d=3: "" if pd.isna(v) else f"{v:.{d}f}"
    pct = lambda v: "" if pd.isna(v) else f"{100*v:.1f}%"

    def table(split):
        c = comp[(comp.split == split) & comp.model.isin(keep)].set_index("model").reindex(keep)
        rows = "".join(f"<tr><td>{names[m]}</td><td>{pct(r.accuracy)}</td><td>{f(r.brier)}</td><td>{f(r.log_loss)}</td>"
                       f"<td>{f(r.margin_mae, 2)}</td><td>{f(r.total_mae, 2)}</td></tr>" for m, r in c.iterrows())
        return ("<table><thead><tr><th>Model</th><th>Winner acc.</th><th>Brier</th><th>Log loss</th>"
                f"<th>Margin MAE</th><th>Total MAE</th></tr></thead><tbody>{rows}</tbody></table>")
    b = bets[bets.split.isin(["test", "holdout"])]
    brow = "".join(f"<tr><td>{r.split}</td><td>{r.line}</td><td>{r.market}</td><td>{r.bets}</td><td>{pct(r.win_rate)}</td>"
                   f"<td class='{'neg' if r.units < 0 else 'pos'}'>{r.units:+.1f}</td><td>{pct(r.roi)} [{pct(r.roi_ci_lo)}, {pct(r.roi_ci_hi)}]</td></tr>"
                   for r in b.itertuples())
    img = lambda n, alt: f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode((CH / n).read_bytes()).decode()}">'
    t = comp[(comp.split == "test")].set_index("model")
    kp = [("Test winner accuracy", pct(t.accuracy["ensemble"]), f"market {pct(t.accuracy['market_moneyline'])}"),
          ("Test log loss", f(t.log_loss["ensemble"]), f"market {f(t.log_loss['market_moneyline'])}"),
          ("Test margin MAE", f(t.margin_mae["ensemble"], 2), f"closing spread {f(t.margin_mae['market_close_lines'], 2)}"),
          ("Test spread ROI (close)", pct(b[(b.split == 'test') & (b.line == 'close') & (b.market == 'spread')].roi.iloc[0]), "after -110 vig")]
    tiles = "".join(f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v}</div><div class='s'>{s_}</div></div>" for k, v, s_ in kp)
    doc = f"""<title>NBA Model Backtest</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>
/* layout: single column report — verdict, KPI tiles, tables, then charts */
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#898781;--line:#e1e0d9;--neg:#d03b3b;--pos:#006300;--card:#f2f2ee;--img:#fcfcfb}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#141413;--fg:#f0efec;--fg2:#c3c2b7;--muted:#9a9890;--line:#2c2c2a;--neg:#e66767;--pos:#0ca30c;--card:#1f1f1d;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#141413;--fg:#f0efec;--fg2:#c3c2b7;--muted:#9a9890;--line:#2c2c2a;--neg:#e66767;--pos:#0ca30c;--card:#1f1f1d;color-scheme:dark}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,system-ui,sans-serif;padding-block:24px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
main{{max-width:1100px;margin:auto;display:grid;gap:28px}} h1{{margin:0;font-size:26px;text-wrap:balance}} main>*{{min-width:0}} h2{{font-size:17px;margin:0 0 10px}}
p{{color:var(--fg2);max-width:70ch;margin:6px 0}} .verdict{{border-left:4px solid var(--neg);padding:10px 14px;background:var(--card)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.tile{{background:var(--card);padding:12px 14px;border-radius:6px}} .k{{color:var(--fg2);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
.v{{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}} .s{{color:var(--muted);font-size:12px}}
.scroll{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
th{{color:var(--fg2);font-weight:600;font-size:12px}} .neg{{color:var(--neg)}} .pos{{color:var(--pos)}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(380px,100%),1fr));gap:20px}} img{{max-width:100%;border:1px solid var(--line);border-radius:6px;background:var(--img)}}
@media (max-width:440px){{.grid2{{grid-template-columns:1fr}}}}
</style><main>
<header><h1>NBA Model Backtest</h1><p>Walk-forward, leakage-tested predictions for every regular-season game, 2008-09 to 2026-27 (through 2026-11-16).
Model v1 was tuned on 2019-20 to 2021-22 only. Test = 2022-23 to 2025-26 (4,920 games); holdout = 2026-27 to date (205 games).</p></header>
<div class="verdict"><strong>Verdict: no edge.</strong> The model beats Elo and naive baselines and is well calibrated, but it trails the closing market on every metric in every season and loses money after vig in every market at closing lines.</div>
<section class="tiles">{tiles}</section>
<section><h2>Test seasons, 2022-23 to 2025-26</h2><div class="scroll">{table("test")}</div></section>
<section><h2>Holdout, 2026-27 to date</h2><div class="scroll">{table("holdout")}</div></section>
<section><h2>Betting after vig (1u flat stakes, 95% bootstrap CI)</h2><div class="scroll"><table><thead><tr><th>Split</th><th>Line</th><th>Market</th><th>Bets</th><th>Win %</th><th>Units</th><th>ROI</th></tr></thead><tbody>{brow}</tbody></table></div></section>
<section class="grid2">{img("calibration.png","Calibration curves")}{img("accuracy_by_bucket.png","Accuracy by confidence")}</section>
<section>{img("bankroll.png","Cumulative units")}</section>
<section class="grid2">{img("per_season_mae.png","MAE by season")}{img("feature_importance.png","Feature importance")}</section>
<section>{img("monthly_units.png","Monthly units")}</section>
</main>"""
    (OUT / "dashboard.html").write_text(doc)


if __name__ == "__main__":
    main()
