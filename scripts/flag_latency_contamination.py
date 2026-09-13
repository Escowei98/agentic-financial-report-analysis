"""
Flags gold-standard items whose measured `latency_seconds` is not a clean
measurement of the system under test.

Two causes are detected, both from the run log rather than the result file,
because neither leaves a trace in the JSON:

1. THROTTLED — the item's processing window contains Vertex 429 responses.
   `perf_counter` runs through the retry backoff, so the recorded latency
   includes waiting for quota, not work done by the system.
2. SUSPENDED — the wall clock jumped inside the item's window (machine
   asleep). The recorded latency then includes part of the sleep.

A flag marks a CANDIDATE, not a verdict: macOS does not advance
`perf_counter` consistently across a suspend, so an item can sit inside a
long gap and still carry a clean latency. `inflation` is therefore reported
alongside — the item's latency over the median of that system's UNFLAGGED
items — and `material` is set when it exceeds MATERIAL_INFLATION. Use
`material` to decide what to exclude; use the raw flags to describe what
happened.

Quality metrics are NOT affected by either: a retried query returns the same
answer, and a suspended process resumes where it left off. Only the
efficiency figures (`latency_seconds`, and any mean built from it) are.

Usage:
    uv run python scripts/flag_latency_contamination.py data/results/final_eval/run1
"""
import argparse
import csv
import glob
import json
import re
from datetime import datetime
from pathlib import Path

SYSTEMS = ["rag_monolith", "rag_agent", "long_context", "multi_agent"]

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
PROC = re.compile(r"\[(\w+)\] Processing \d+/\d+ \(id=(\d+)\)")
QUOTA = re.compile(r"Resource (?:has been )?exhausted|429")
FAILED = re.compile(r"\[(\w+)\] Query failed for id=(\d+) after \d+ attempts")

# A wall-clock gap larger than this inside one item means the process was
# suspended; no single legitimate step in any of the four systems takes
# this long.
SUSPEND_GAP_SECONDS = 300

# Below this the retry backoff is lost in normal run-to-run variation and
# excluding the item would cost more n than it buys in accuracy.
MATERIAL_INFLATION = 1.5


def parse_log(log_path: Path) -> dict:
    """Walk the log once, attributing every quota hit to the item in flight."""
    items: dict[tuple[str, int], dict] = {}
    cur: tuple[str, int] | None = None
    prev_dt: datetime | None = None

    for line in log_path.open(encoding="utf-8", errors="replace"):
        m = TS.match(line)
        dt = None
        if m:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

        p = PROC.search(line)
        if p:
            cur = (p.group(1), int(p.group(2)))
            items.setdefault(cur, {
                "system": p.group(1), "id": int(p.group(2)),
                "quota_hits": 0, "suspended_seconds": 0.0, "lost": False,
                "started": dt,
            })
            prev_dt = dt
            continue

        if dt is not None and prev_dt is not None and cur is not None:
            gap = (dt - prev_dt).total_seconds()
            if gap > SUSPEND_GAP_SECONDS:
                items[cur]["suspended_seconds"] += gap
        if dt is not None:
            prev_dt = dt

        if cur is not None and QUOTA.search(line):
            items[cur]["quota_hits"] += 1

        f = FAILED.search(line)
        if f:
            key = (f.group(1), int(f.group(2)))
            if key in items:
                items[key]["lost"] = True
    return items


def load_latencies(run_dir: Path) -> dict[tuple[str, int], float]:
    out = {}
    for system in SYSTEMS:
        merged = run_dir / f"eval_{system}_MERGED.json"
        cands = [merged] if merged.exists() else sorted(
            Path(p) for p in glob.glob(str(run_dir / f"eval_{system}_*.json")))
        if not cands:
            continue
        data = json.loads(cands[-1].read_text(encoding="utf-8"))
        for it in data.get("detailed_results", []):
            lat = (it.get("run_metrics") or {}).get("latency_seconds")
            if lat is not None:
                out[(system, it["query_id"])] = lat
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV target (default: <run_dir>/latency_contamination.csv)")
    args = ap.parse_args(argv)

    log_path = args.run_dir / "run.log"
    if not log_path.exists():
        raise SystemExit(f"No run.log in {args.run_dir}")

    items = parse_log(log_path)
    lats = load_latencies(args.run_dir)

    rows = []
    for key, rec in sorted(items.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        reasons = []
        if rec["quota_hits"]:
            reasons.append("throttled")
        if rec["suspended_seconds"] > 0:
            reasons.append("suspended")
        if rec["lost"]:
            reasons.append("lost")
        if not reasons:
            continue
        rows.append({
            "system": rec["system"], "query_id": rec["id"],
            "reason": "+".join(reasons),
            "quota_hits": rec["quota_hits"],
            "suspended_seconds": round(rec["suspended_seconds"], 1),
            "latency_seconds": lats.get(key, ""),
        })

    # Baseline per system: median latency over the items NOT flagged at all,
    # i.e. the system running under conditions it was meant to be measured in.
    flagged = {(r["system"], r["query_id"]) for r in rows}
    for r in rows:
        clean = sorted(v for k, v in lats.items()
                       if k[0] == r["system"] and k not in flagged)
        lat = r["latency_seconds"]
        if clean and isinstance(lat, (int, float)):
            med = clean[len(clean) // 2]
            r["inflation"] = round(lat / med, 2) if med else ""
            r["material"] = "yes" if med and lat / med >= MATERIAL_INFLATION else "no"
        else:
            r["inflation"] = ""
            r["material"] = "unknown" if not clean else "no"

    out = args.out or args.run_dir / "latency_contamination.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "system", "query_id", "reason", "quota_hits",
            "suspended_seconds", "latency_seconds", "inflation", "material"])
        w.writeheader()
        w.writerows(rows)

    def _mean(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else None

    def _median(vals):
        vals = sorted(vals)
        return vals[len(vals) // 2] if vals else None

    print(f"{'System':16s}{'geflaggt':>10s}{'material':>10s}{'verloren':>10s}"
          f"{'Mittel alle':>13s}{'Mittel clean':>14s}{'Median alle':>13s}")
    for system in SYSTEMS:
        sub = [r for r in rows if r["system"] == system]
        total = len([k for k in items if k[0] == system])
        if not total:
            continue
        lost = len([r for r in sub if "lost" in r["reason"]])
        mat = len([r for r in sub if r.get("material") == "yes"])
        unknown = any(r.get("material") == "unknown" for r in sub)

        all_lat = [v for k, v in lats.items() if k[0] == system]
        flag_ids = {r["query_id"] for r in sub}
        clean_lat = [v for k, v in lats.items()
                     if k[0] == system and k[1] not in flag_ids]

        mat_s = "n/a" if unknown else str(mat)
        ma, mc, md = _mean(all_lat), _mean(clean_lat), _median(all_lat)
        fmt = lambda x: f"{x:13.2f}" if x is not None else f"{'n/a':>13s}"
        print(f"{system:16s}{len(sub):>4d}/{total:<5d}{mat_s:>10s}{lost:>10d}"
              f"{fmt(ma)}{fmt(mc)[1:]:>14s}{fmt(md)}")
    print("\n  'Mittel clean' laesst alle geflaggten Items weg. Die Differenz zu"
          "\n  'Mittel alle' ist der Betrag, um den die Stoerungen die berichtete"
          "\n  Kennzahl verschieben. 'n/a': Ergebnisdatei noch nicht geschrieben.")
    print(f"\nCSV: {out}  ({len(rows)} Zeilen)")


if __name__ == "__main__":
    main()
