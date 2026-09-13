"""
Blind hand-rating sheet for the FA-Refusal stratum (Kap. 5.2.2, Kopfmetrik).

Writes one row per (refusal item x system) for a single run, system identity
hidden behind a random rating id, rows shuffled with a fixed seed. The judge
scores and the id -> system mapping go to a separate key file that stays
closed until the sheet is filled in. `--merge` joins a filled sheet back to
the key and prints refusal_accuracy / refusal_quality per system and subtype.

    uv run python scripts/build_refusal_rating_sheet.py --run run2
    uv run python scripts/build_refusal_rating_sheet.py --run run2 --merge
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.generate_report import SYSTEMS, latest_result_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "data" / "results" / "final_eval"
OUT_DIR = RUNS_ROOT / "refusal_hand_rating"
SEED = 20260911

SHEET_COLUMNS = [
    "rating_id", "subtype", "refusal_evidence", "question",
    "reference_gt_correction", "system_answer",
    "refusal_accuracy_human", "refusal_quality_human", "notes",
]


def build(run: str) -> None:
    rows, key = [], []
    for label, name in SYSTEMS:
        data = json.load(open(latest_result_file(RUNS_ROOT / run, name), encoding="utf-8"))
        for rec in data["detailed_results"]:
            if rec.get("expected_answerable"):
                continue
            cm = rec.get("custom_metrics") or {}
            rat = cm.get("rationales") or {}
            rows.append({
                "subtype": rec.get("subtype", ""),
                "refusal_evidence": rec.get("refusal_evidence", ""),
                "question": rec["question"],
                "reference_gt_correction": rec.get("gt_correction", ""),
                "system_answer": (rec.get("answer") or "").strip() or "<LEERANTWORT>",
                "_system": label, "_query_id": rec["query_id"],
                "_judge_refusal_accuracy": cm.get("refusal_accuracy"),
                "_judge_refusal_quality": cm.get("refusal_quality"),
                "_judge_rationale_accuracy": rat.get("refusal_accuracy", ""),
                "_judge_rationale_quality": rat.get("refusal_quality", ""),
            })
    rng = random.Random(SEED)
    rng.shuffle(rows)
    for i, r in enumerate(rows, start=1):
        r["rating_id"] = f"R{i:03d}"
        key.append({
            "rating_id": r["rating_id"], "run": run, "system": r["_system"],
            "query_id": r["_query_id"], "subtype": r["subtype"],
            "refusal_evidence": r["refusal_evidence"],
            "judge_refusal_accuracy": r["_judge_refusal_accuracy"],
            "judge_refusal_quality": r["_judge_refusal_quality"],
            "judge_rationale_accuracy": r["_judge_rationale_accuracy"],
            "judge_rationale_quality": r["_judge_rationale_quality"],
        })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet = OUT_DIR / f"refusal_rating_{run}.csv"
    # utf-8-sig + ';' so Excel/Numbers open it with umlauts and columns intact
    with open(sheet, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS, delimiter=";",
                           extrasaction="ignore", quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "refusal_accuracy_human": "", "refusal_quality_human": "", "notes": ""})
    with open(OUT_DIR / f"refusal_rating_{run}_KEY.json", "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=1, ensure_ascii=False)
    print(f"wrote {sheet} ({len(rows)} rows) and the key file — do not open the key before rating")


def _num(x: str) -> float | None:
    x = (x or "").strip().replace(",", ".")
    return float(x) if x else None


def merge(run: str) -> None:
    sheet = OUT_DIR / f"refusal_rating_{run}.csv"
    key = {k["rating_id"]: k for k in json.load(open(OUT_DIR / f"refusal_rating_{run}_KEY.json", encoding="utf-8"))}
    with open(sheet, encoding="utf-8-sig") as fh:
        rated = list(csv.DictReader(fh, delimiter=";"))
    merged, missing = [], []
    for r in rated:
        k = key[r["rating_id"]]
        acc, qual = _num(r["refusal_accuracy_human"]), _num(r["refusal_quality_human"])
        if acc is None or qual is None:
            missing.append(r["rating_id"])
        merged.append({**k, "human_refusal_accuracy": acc, "human_refusal_quality": qual, "notes": r.get("notes", "")})
    out = OUT_DIR / f"refusal_rating_{run}_MERGED.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)
    if missing:
        print(f"{len(missing)} rows still unrated: {', '.join(missing[:10])}{' …' if len(missing) > 10 else ''}")

    def table(groupkey: str) -> None:
        groups: dict[tuple, list] = defaultdict(list)
        for m in merged:
            if m["human_refusal_accuracy"] is not None:
                groups[(m[groupkey], m["system"])].append(m)
        names = sorted({g for g, _ in groups})
        print(f"\n| {groupkey} | n | " + " | ".join(f"{s} acc / qual" for s, _ in SYSTEMS) + " |")
        print("|---|---|" + "---|" * len(SYSTEMS))
        for g in names:
            cells = []
            for s, _ in SYSTEMS:
                ms = groups.get((g, s), [])
                if not ms:
                    cells.append("—")
                    continue
                acc = statistics.mean(m["human_refusal_accuracy"] for m in ms)
                qual = statistics.mean(m["human_refusal_quality"] for m in ms)
                cells.append(f"{acc:.3f} / {qual:.3f}")
            n = len(groups.get((g, SYSTEMS[0][0]), []))
            print(f"| {g} | {n} | " + " | ".join(cells) + " |")

    for m in merged:
        m["all"] = "alle"
    table("all")
    table("subtype")
    table("refusal_evidence")
    # agreement judge vs human, for the Nebenbefund sentence
    for metric in ("refusal_accuracy", "refusal_quality"):
        pairs = [(m[f"judge_{metric}"], m[f"human_{metric}"]) for m in merged
                 if m[f"human_{metric}"] is not None and m[f"judge_{metric}"] is not None]
        if pairs:
            agree = sum(1 for j, h in pairs if abs(float(j) - h) < 1e-9) / len(pairs)
            bias = statistics.mean(float(j) - h for j, h in pairs)
            print(f"\n{metric}: judge–human agreement {agree:.2f}, "
                  f"mean bias (judge − human) {bias:+.3f}, n = {len(pairs)}")
    print(f"\nwrote {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run2")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args(argv)
    merge(args.run) if args.merge else build(args.run)


if __name__ == "__main__":
    main()
