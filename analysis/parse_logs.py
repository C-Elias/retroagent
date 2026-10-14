"""Parse RetroAgent training logs into one row per session.

Raw logs are not published (they contain strategy detail: strikes, entry
times). This script turns them into session-level aggregates, which are.

Usage: uv run analysis/parse_logs.py <raw_log_dir> [out_csv]
"""

import csv
import re
import sys
from pathlib import Path

# Run label -> (log file, lane). Lanes: "fresh" = empty knowledge base at the
# start of the chained Jan-May 2025 run; "replay" = the same month re-run with
# identical configuration; "continued" = later months chained after May.
RUNS = {
    "jan2025_fresh": ("training_jan2025_fresh.log", "fresh"),
    "feb2025_fresh": ("training_feb2025_fresh.log", "fresh"),
    "mar2025_fresh": ("training_mar2025_fresh.log", "fresh"),
    "apr2025_fresh": ("training_apr2025_fresh.log", "fresh"),
    "may2025_fresh": ("training_may2025_fresh.log", "fresh"),
    "apr2025_replay": ("training_apr2025_replay.log", "replay"),
    "apr2025_rep1": ("training_apr_rep1.log", "replay"),
    "jan2025_rep1": ("training_jan_rep1.log", "replay"),
    "jul2025": ("training_july_2025.log", "continued"),
    "aug2025": ("training_august_2025.log", "continued"),
    # training_september_2025.log is empty in the local mirror: excluded.
}

INIT = re.compile(r"init -> (\w+), VIX ([\d.]+) \((\w+)\), gap=(\w+)")
CUM = re.compile(r"cum_pnl=\$(-?[\d,]+)")
SUMMARY = re.compile(r"summary -> (\d+) trades, (\d+) won, P/L \$(-?[\d.]+)")
OPEN = re.compile(r"open -> (\S+) \S*?(PutCreditSpread|CallCreditSpread)")
DONE = re.compile(r"-> DONE in (\d+)s, (\d+) turns, KB=(\d+)")
AUDIT = re.compile(r"audit: (\w+), flags: (\d+)")
MODEL = re.compile(r"^Model: (\S+)")
PROMPT = re.compile(r"^prompt_version: (\S+)")

FIELDS = [
    "run", "lane", "model", "prompt_version", "session", "weekday", "vix",
    "vix_regime", "gap", "trades", "won", "pnl", "put_opens", "call_opens",
    "rejected_opens", "turns", "seconds", "kb_end", "audit", "audit_flags",
    "completed", "pnl_source",
]


def parse(run, path, lane):
    rows, cur, model, prompt, cums = [], None, None, None, []

    def close():
        if cur is not None:
            rows.append(cur)

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if m := MODEL.match(s):
            model = m.group(1)
        if m := PROMPT.match(s):
            prompt = m.group(1)
        if s.startswith("SESSION ") or s.startswith("TRAINING RUN COMPLETE"):
            close()
            cur = None
            if s.startswith("TRAINING RUN COMPLETE"):
                continue
            cur = {"run": run, "lane": lane, "model": model, "prompt_version": prompt,
                   "session": int(s.split()[1].split("/")[0]), "put_opens": 0,
                   "call_opens": 0, "rejected_opens": 0, "trades": 0, "won": 0,
                   "pnl": 0.0, "audit": "", "audit_flags": 0, "completed": False}
            continue
        if cur is None:
            continue
        if m := INIT.search(s):
            cur.update(weekday=m.group(1), vix=float(m.group(2)),
                       vix_regime=m.group(3), gap=m.group(4))
            c = CUM.search(s)
            # Running total BEFORE this session (absent on session 1 = $0).
            cums.append(float(c.group(1).replace(",", "")) if c else 0.0)
        elif m := SUMMARY.search(s):
            # The last summary of a session is its final tally.
            cur.update(trades=int(m.group(1)), won=int(m.group(2)), pnl=float(m.group(3)))
        elif m := OPEN.search(s):
            status, side = m.groups()
            if status == "None":
                cur["rejected_opens"] += 1
            elif side == "PutCreditSpread":
                cur["put_opens"] += 1
            else:
                cur["call_opens"] += 1
        elif m := AUDIT.search(s):
            cur.update(audit=m.group(1), audit_flags=int(m.group(2)))
        elif m := DONE.search(s):
            cur.update(seconds=int(m.group(1)), turns=int(m.group(2)),
                       kb_end=int(m.group(3)), completed=True)
    close()

    # The harness's running cum_pnl is authoritative: it includes P/L that a
    # session's last in-session summary can miss (late settlements, a session
    # cut short by the circuit breaker). Session i's P/L = cum[i+1] - cum[i];
    # the final session falls back to its own last summary.
    for i, r in enumerate(rows):
        if i + 1 < len(cums) and i < len(cums):
            r["pnl"] = round(cums[i + 1] - cums[i], 2)
            r["pnl_source"] = "cum_pnl_delta"
        else:
            r["pnl_source"] = "final_summary"
    return rows


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/sessions.csv")
    rows = []
    for run, (fname, lane) in RUNS.items():
        p = src / fname
        if p.exists():
            rows += parse(run, p, lane)
        else:
            print(f"missing: {fname}", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} sessions -> {out}")


if __name__ == "__main__":
    main()
