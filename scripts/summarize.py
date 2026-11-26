"""Summarize `vllm bench serve` JSON results: median over runs for each (input, output, concurrency) cell.

Usage: python3 scripts/summarize.py results/<tag>
Writes results/<tag>/summary.md and summary.csv.
"""
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("median_ttft_ms", "TTFT p50 (ms)"),
    ("p99_ttft_ms", "TTFT p99 (ms)"),
    ("median_tpot_ms", "TPOT p50 (ms)"),
    ("p99_tpot_ms", "TPOT p99 (ms)"),
    ("output_throughput", "output tok/s"),
    ("request_throughput", "req/s"),
]

out_dir = Path(sys.argv[1])
cells = defaultdict(list)
for f in sorted((out_dir / "raw").glob("*.json")):
    m = re.match(r"in(\d+)_out(\d+)_c(\d+)_r(\d+)\.json", f.name)
    if m:
        cells[tuple(int(x) for x in m.groups()[:3])].append(json.loads(f.read_text()))

rows = []
for (inp, outp, conc), runs in sorted(cells.items()):
    row = {"input": inp, "output": outp, "concurrency": conc, "runs": len(runs),
           "failed": sum(r.get("failed", 0) for r in runs)}
    # Preempted requests per run (client_loop.sh writes "<before> <after>"; NA for servers without the metric).
    pre = []
    for p in sorted((out_dir / "raw").glob(f"in{inp}_out{outp}_c{conc}_r*.preemptions")):
        before, after = p.read_text().split()
        if before != "NA" and after != "NA":
            pre.append(int(after) - int(before))
    row["preemptions"] = statistics.median(pre) if pre else None
    for key, _ in METRICS:
        vals = [r[key] for r in runs if r.get(key) is not None]
        row[key] = statistics.median(vals) if vals else None
        # Run-to-run spread of the median, to see how noisy each cell is.
        row[key + "_spread_pct"] = (100 * (max(vals) - min(vals)) / statistics.median(vals)
                                    if len(vals) > 1 and statistics.median(vals) else None)
    rows.append(row)

with open(out_dir / "summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["empty"])
    w.writeheader()
    w.writerows(rows)

server_log = (out_dir / "server.log").read_text(errors="replace") if (out_dir / "server.log").exists() else ""
# vLLM: "Available KV cache memory" / "GPU KV cache size"; TRT-LLM: profiling peak, max_tokens quota, KV tiers.
kv_lines = [l.strip() for l in server_log.splitlines()
            if re.search(r"GPU KV cache size|Available KV cache memory|Peak memory during memory usage profiling"
                         r"|max_tokens \d+ is provided|KV cache manager v2 \w+ (cache )?quota", l)]
apps = (out_dir / "compute_apps.txt").read_text().split() if (out_dir / "compute_apps.txt").exists() else []
max_procs = max((int(a) for a in apps if a.isdigit()), default=0)

lines = [f"# {out_dir.name}", ""]
lines += (out_dir / "env.txt").read_text().strip().splitlines() if (out_dir / "env.txt").exists() else []
lines += ["", "Median over runs; spread = (max - min) / median across runs.", ""]
lines.append("| in/out | conc | " + " | ".join(label for _, label in METRICS) + " | preempted | failed |")
lines.append("|---|---|" + "---|" * len(METRICS) + "---|---|")
for r in rows:
    cells_txt = []
    for key, _ in METRICS:
        v, s = r[key], r[key + "_spread_pct"]
        cells_txt.append("—" if v is None else f"{v:.1f}" + (f" (±{s/2:.0f}%)" if s is not None and s >= 5 else ""))
    pre_txt = "—" if r["preemptions"] is None else f"{r['preemptions']:g}"
    lines.append(f"| {r['input']}/{r['output']} | {r['concurrency']} | " + " | ".join(cells_txt)
                 + f" | {pre_txt} | {r['failed']} |")
lines += ["", "## Server log: KV cache", ""] + [f"- `{l[:400]}`" for l in kv_lines[:10]]
lines += ["", f"Max processes on the GPU during the run: {max_procs} (the server itself shows 1-2; more means something else ran)"]
load = sorted(float(x) for x in (out_dir / "cpu_load.txt").read_text().split()) if (out_dir / "cpu_load.txt").exists() else []
if load:
    lines += [f"CPU load (1-minute average) during the run: median {load[len(load) // 2]:.1f}, max {load[-1]:.1f} "
              "(the benchmark itself is about 2; more means other jobs shared the CPU)"]
(out_dir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
