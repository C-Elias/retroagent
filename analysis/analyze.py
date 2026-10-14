"""Statistics and figures for the RetroAgent paper.

Usage: uv run analysis/analyze.py   (reads data/sessions.csv, writes figures/ and data/stats.json)
"""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RNG = np.random.default_rng(0)
B = 10_000  # bootstrap resamples

d = pd.read_csv("data/sessions.csv")
d["put_share"] = d.put_opens / (d.put_opens + d.call_opens)
stats = {}


def boot_sum_ci(x):
    """95% bootstrap CI for a month's total P/L, resampling sessions."""
    x = np.asarray(x)
    sums = RNG.choice(x, size=(B, len(x)), replace=True).sum(axis=1)
    return [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


# --- 1. Monthly totals with session-bootstrap CIs (fresh-KB chain) ----------
fresh = d[d.lane == "fresh"]
months = ["jan2025_fresh", "feb2025_fresh", "mar2025_fresh", "apr2025_fresh", "may2025_fresh"]
monthly = []
for run in months:
    x = d[d.run == run].pnl
    lo, hi = boot_sum_ci(x)
    monthly.append({"run": run, "sessions": int(len(x)), "total": float(x.sum()),
                    "ci95": [lo, hi], "mean_vix": float(d[d.run == run].vix.mean())})
stats["fresh_monthly"] = monthly
stats["fresh_total"] = float(fresh.pnl.sum())

# Accumulated-KB totals as documented in the run notes (logs not in this archive).
ACCUM = {"jan2025_fresh": -1038, "feb2025_fresh": -5293, "mar2025_fresh": -5108,
         "apr2025_fresh": -3235, "may2025_fresh": 10747}
stats["accumulated_documented"] = ACCUM

fig, ax = plt.subplots(figsize=(8, 4.2))
xs = np.arange(len(months))
tot = [m["total"] for m in monthly]
err = np.array([[m["total"] - m["ci95"][0], m["ci95"][1] - m["total"]] for m in monthly]).T
ax.bar(xs - 0.2, tot, 0.4, yerr=err, capsize=4, label="Fresh memory (logs, 95% bootstrap CI)", color="#3b6fb6")
ax.bar(xs + 0.2, [ACCUM[r] for r in months], 0.4, label="Accumulated memory (documented)", color="#c9a227")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xs, ["Jan", "Feb", "Mar", "Apr", "May"])
ax.set_ylabel("Monthly P/L (USD, $100K account)")
ax.set_title("Fresh vs accumulated agent memory, Jan-May 2025")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("figures/fig1_memory_monthly.png", dpi=160)

# --- 2. Distribution shift: put share vs VIX --------------------------------
f = fresh.dropna(subset=["put_share", "vix"])
slope, intercept = np.polyfit(f.vix, f.put_share, 1)
r = float(np.corrcoef(f.vix, f.put_share)[0, 1])
# Bootstrap CI on the correlation.
idx = RNG.integers(0, len(f), size=(B, len(f)))
rs = [np.corrcoef(f.vix.values[i], f.put_share.values[i])[0, 1] for i in idx[:2000]]
stats["put_share_vs_vix"] = {"n_sessions": int(len(f)), "pearson_r": r,
                             "r_ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))],
                             "slope_per_vix_point": float(slope)}
gd = fresh[fresh.gap == "gap_down"].groupby("run").apply(
    lambda g: g.put_opens.sum() / (g.put_opens.sum() + g.call_opens.sum()), include_groups=False)
stats["gap_down_put_share_by_month"] = {k: float(v) for k, v in gd.reindex(months).dropna().items()}

fig, ax = plt.subplots(figsize=(7, 4.2))
colors = {"jan2025_fresh": "#9bb7e0", "feb2025_fresh": "#6f97d1", "mar2025_fresh": "#3b6fb6",
          "apr2025_fresh": "#c0392b", "may2025_fresh": "#1f3f73"}
for run in months:
    g = f[f.run == run]
    ax.scatter(g.vix, g.put_share, s=22, alpha=0.8, color=colors[run], label=run.split("2025")[0].title())
vx = np.linspace(f.vix.min(), f.vix.max(), 50)
ax.plot(vx, intercept + slope * vx, color="black", lw=1, ls="--", label=f"OLS fit (r = {r:.2f})")
ax.set_xlabel("VIX at session open")
ax.set_ylabel("Share of new positions that are put spreads")
ax.set_title("Agent behaviour shifts with volatility regime")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig2_put_share_vs_vix.png", dpi=160)

# --- 3. Run-to-run variance on identical configuration ----------------------
def paired(a, b):
    x = d[d.run == a].set_index("session").pnl
    y = d[d.run == b].set_index("session").pnl
    j = x.index.intersection(y.index)
    x, y = x.loc[j], y.loc[j]
    return {"runs": [a, b], "matched_sessions": int(len(j)),
            "total_a": float(x.sum()), "total_b": float(y.sum()),
            "daily_pnl_corr": float(np.corrcoef(x, y)[0, 1]) if len(j) > 2 else None,
            "mean_abs_daily_diff": float((x - y).abs().mean())}

stats["replicates"] = [paired("apr2025_fresh", "apr2025_replay"),
                       paired("apr2025_fresh", "apr2025_rep1"),
                       paired("apr2025_replay", "apr2025_rep1"),
                       paired("jan2025_fresh", "jan2025_rep1")]
stats["apr_full_runs_totals"] = {"apr2025_fresh": float(d[d.run == "apr2025_fresh"].pnl.sum()),
                                 "apr2025_replay": float(d[d.run == "apr2025_replay"].pnl.sum())}

fig, ax = plt.subplots(figsize=(8, 4.2))
for run, c, lab in [("apr2025_fresh", "#c0392b", "Run A"), ("apr2025_replay", "#3b6fb6", "Run B (identical config)"),
                    ("apr2025_rep1", "#7f8c8d", "Run C (identical config, partial)")]:
    g = d[d.run == run].sort_values("session")
    ax.plot(g.session, g.pnl.cumsum(), marker="o", ms=3, color=c, label=lab)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Trading session (same days across runs)")
ax.set_ylabel("Cumulative P/L (USD)")
ax.set_title("April 2025: same model, prompt, memory and data; three outcomes")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig3_april_replicates.png", dpi=160)

# --- 4. Memory growth and agent effort --------------------------------------
stats["kb_growth_fresh"] = {run: int(d[d.run == run].kb_end.max()) for run in months}
stats["turns_median_by_run"] = {run: float(d[d.run == run].turns.median()) for run in d.run.unique()}
stats["rejected_open_rate"] = float(d.rejected_opens.sum() / (d.rejected_opens + d.put_opens + d.call_opens).sum())
stats["incomplete_sessions"] = d[~d.completed][["run", "session"]].to_dict("records")

# --- 5. Extra figures ---------------------------------------------------------
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May"]

# fig4: daily P/L, April Run A vs Run B, coloured by VIX.
a = d[d.run == "apr2025_fresh"].set_index("session")
b = d[d.run == "apr2025_replay"].set_index("session")
j = a.index.intersection(b.index)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
sc = ax.scatter(a.loc[j, "pnl"], b.loc[j, "pnl"], c=a.loc[j, "vix"], cmap="viridis", s=45, edgecolor="black", lw=0.4)
lim = [min(a.pnl.min(), b.pnl.min()) - 500, max(a.pnl.max(), b.pnl.max()) + 500]
ax.plot(lim, lim, color="grey", ls="--", lw=1, label="Identical outcome")
gap = (b.loc[j, "pnl"] - a.loc[j, "pnl"]).abs().sort_values(ascending=False)
for sid in gap.head(3).index:
    ax.annotate(f"day {sid}, VIX {a.loc[sid, 'vix']:.0f}", (a.loc[sid, "pnl"], b.loc[sid, "pnl"]),
                textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Run A daily P/L (USD)"); ax.set_ylabel("Run B daily P/L (USD)")
ax.set_title(f"Same day, same configuration (r = {np.corrcoef(a.loc[j, 'pnl'], b.loc[j, 'pnl'])[0, 1]:.2f})")
fig.colorbar(sc, ax=ax, label="VIX at session open")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig4_daily_pairs.png", dpi=160)

# fig5: where the divergence comes from: |A - B| per day with VIX.
div = (b.loc[j, "pnl"] - a.loc[j, "pnl"]).abs()
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(j, div, color=["#c0392b" if a.loc[s_, "gap"] == "gap_down" else "#95a5a6" for s_ in j])
ax.set_xlabel("Trading session (April 2025)"); ax.set_ylabel("|Run A - Run B| daily P/L (USD)")
ax2 = ax.twinx(); ax2.plot(j, a.loc[j, "vix"], color="black", marker="o", ms=3, lw=1); ax2.set_ylabel("VIX")
ax.set_title("Divergence between identical runs concentrates on stressed gap-down days")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#c0392b", label="gap-down day"), Patch(color="#95a5a6", label="other day")], fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig("figures/fig5_divergence_by_day.png", dpi=160)

# fig6: session P/L spread by month (variance by regime).
data6 = [d[d.run == r].pnl.values for r in months]
fig, ax = plt.subplots(figsize=(8, 4.2))
bp = ax.boxplot(data6, tick_labels=MONTH_LABELS, patch_artist=True, showfliers=True)
for patch, r in zip(bp["boxes"], months):
    patch.set_facecolor(colors[r]); patch.set_alpha(0.6)
for i, r in enumerate(months, start=1):
    ax.text(i, ax.get_ylim()[1] * 0.92, f"VIX {d[d.run == r].vix.mean():.0f}", ha="center", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Session P/L (USD)"); ax.set_title("Per-session outcomes widen in the high-volatility month")
fig.tight_layout(); fig.savefig("figures/fig6_session_spread.png", dpi=160)
stats["session_pnl_std_by_month"] = {r: float(d[d.run == r].pnl.std()) for r in months}

# fig7: gap-down put share by month.
gdv = [stats["gap_down_put_share_by_month"].get(r, np.nan) for r in months]
fig, ax = plt.subplots(figsize=(6.4, 4))
bars = ax.bar(MONTH_LABELS, gdv, color=[colors[r] for r in months])
for bar_, v in zip(bars, gdv):
    ax.text(bar_.get_x() + bar_.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
ax.set_ylim(0, 0.8); ax.set_ylabel("Put share of new positions on gap-down days")
ax.set_title("On gap-down days the agent leaned further into puts each month")
fig.tight_layout(); fig.savefig("figures/fig7_gapdown_put_share.png", dpi=160)

# fig8: memory growth and cumulative P/L across the chained fresh run.
chain = pd.concat([d[d.run == r].sort_values("session") for r in months], ignore_index=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True)
ax1.plot(chain.index + 1, chain.kb_end.ffill(), color="#1f3f73")  # incomplete sessions log no KB size
ax1.set_ylabel("Knowledge-base entries")
ax2.plot(chain.index + 1, chain.pnl.cumsum(), color="#3b6fb6")
ax2.axhline(0, color="black", lw=0.8); ax2.set_ylabel("Cumulative P/L (USD)")
ax2.set_xlabel("Session in the Jan-May 2025 chain")
start = 0
for r, lab in zip(months, MONTH_LABELS):
    n = int((d.run == r).sum())
    for ax_ in (ax1, ax2):
        ax_.axvline(start + 0.5, color="grey", lw=0.6, ls=":")
    ax1.text(start + n / 2, chain.kb_end.max() * 0.95, lab, ha="center", fontsize=8)
    start += n
ax1.set_title("Memory grows steadily; returns do not")
fig.tight_layout(); fig.savefig("figures/fig8_memory_growth.png", dpi=160)

# fig9: agent effort vs volatility.
e = d.dropna(subset=["turns", "vix"])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.scatter(e.vix, e.turns, s=16, alpha=0.6, color="#3b6fb6")
r_turns = float(np.corrcoef(e.vix, e.turns)[0, 1])
ax.set_xlabel("VIX at session open"); ax.set_ylabel("Agent turns in the session")
ax.set_title(f"The agent takes more turns in volatile sessions (r = {r_turns:.2f})")
fig.tight_layout(); fig.savefig("figures/fig9_turns_vs_vix.png", dpi=160)
stats["turns_vs_vix"] = {"n_sessions": int(len(e)), "pearson_r": r_turns}

with open("data/stats.json", "w") as fh:
    json.dump(stats, fh, indent=2)
print(json.dumps(stats, indent=2))
