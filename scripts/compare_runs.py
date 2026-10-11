"""Paired comparison of evaluation runs on identical scenarios.

For each disturbance condition: route-success rate with a 95% Wilson interval
per method, and for each method against the first (reference) run, the paired
difference with an exact two-sided McNemar (binomial) test on discordant pairs.
Only scenarios present in every run are compared.
"""
import argparse
import json
from math import comb, sqrt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("runs", nargs="+", type=Path, help="first run is the reference")
parser.add_argument("--metric", default="route_success")
parser.add_argument("--markdown", type=Path)
args = parser.parse_args()


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),)*2
    p = k/n
    centre = (p + z*z/(2*n))/(1 + z*z/n)
    half = z*sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return centre-half, centre+half


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c)+1))/2**n
    return min(1.0, 2*tail)


runs = []
for path in args.runs:
    results = {}
    for line in (path/"episodes.jsonl").read_text().splitlines():
        r = json.loads(line)
        results[r["scenario"]] = r
    manifest = json.loads((path/"manifest.json").read_text())
    label = manifest["config"]["arguments"].get("controller", "?")
    if manifest["config"]["arguments"].get("tag"): label += f" ({manifest['config']['arguments']['tag']})"
    runs.append((path.name, label, results))

common = set.intersection(*(set(r[2]) for r in runs))
conditions = sorted({runs[0][2][s]["disturbance"] for s in common})
lines = [f"Paired comparison on {len(common)} common scenarios; metric `{args.metric}`.", "",
         "| Condition | Method | Run | Rate (95% CI) | Paired diff vs reference | McNemar p |",
         "|---|---|---|---|---|---|"]
for cond in conditions + ["all"]:
    ids = sorted(s for s in common if cond == "all" or runs[0][2][s]["disturbance"] == cond)
    ref = [bool(runs[0][2][s][args.metric]) for s in ids]
    for run_id, label, results in runs:
        vals = [bool(results[s][args.metric]) for s in ids]
        k, n = sum(vals), len(vals)
        lo, hi = wilson(k, n)
        if results is runs[0][2]:
            diff, p = "reference", ""
        else:
            b = sum(v and not r for v, r in zip(vals, ref))  # method succeeds, reference fails
            c = sum(r and not v for v, r in zip(vals, ref))
            diff, p = f"{100*(b-c)/n:+.1f} pts ({b} won, {c} lost)", f"{mcnemar(b, c):.3g}"
        lines.append(f"| {cond} | {label} | `{run_id}` | {100*k/n:.1f}% ({100*lo:.1f}–{100*hi:.1f}), n={n} | {diff} | {p} |")
text = "\n".join(lines)
print(text)
if args.markdown: args.markdown.write_text(text+"\n")
