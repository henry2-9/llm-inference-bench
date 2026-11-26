"""Put the breakdowns of several runs side by side.

Usage: python3 scripts/profile_summary.py results/<prefix>
Reads <prefix>/<engine>_<precision>/breakdown.json (profile_breakdown.py); writes <prefix>/summary.md:
per capture, GPU kernel time per decode step and per prefill, one column per run.
"""
import json
import sys
from pathlib import Path

ENGINES = ["vllm", "trtllm"]
PRECISIONS = ["bf16", "fp8", "nvfp4"]
PARTS = ["linear", "attention", "quantize", "norm", "activation", "sampling", "other"]

root = Path(sys.argv[1])
runs = {}
for p in PRECISIONS:
    for e in ENGINES:
        f = root / f"{e}_{p}" / "breakdown.json"
        if f.exists():
            runs[f"{e}_{p}"] = json.loads(f.read_text())
names = list(runs)
captures = []
for r in runs.values():
    captures += [c for c in r if c not in captures]

lines = [f"# {root.name}: GPU kernel time per decode step and per prefill (ms)", "",
         "From Nsight Systems traces (profile.sh); medians, see each run's breakdown.md for the kernels.",
         "\"whole\" is the time from one step (or prefill) to the next, \"GPU idle\" the part of it with no kernel running."]
for cap in captures:
    spec = next(r[cap] for r in runs.values() if cap in r)
    for phase, unit in (("decode", "decode step"), ("prefill", "prefill")):
        cols = [n for n in names if phase in runs[n].get(cap, {})]
        if not cols:
            continue
        lines += ["", f"## {cap}, {unit}: {spec['input']} in / {spec['output']} out, {spec['concurrency']} at a time", "",
                  "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        get = lambda n: runs[n][cap][phase]
        lines.append("| whole | " + " | ".join(f"{get(n)['wall_ms']:.2f}" for n in cols) + " |")
        for part in PARTS:
            if any(get(n)["parts"].get(part, 0) >= 0.0005 for n in cols):
                lines.append(f"| {part} | " + " | ".join(f"{get(n)['parts'].get(part, 0):.3f}" for n in cols) + " |")
        lines.append("| GPU idle | " + " | ".join(f"{get(n)['wall_ms'] - get(n)['kernel_ms']:.3f}" for n in cols) + " |")
        key = "client_tpot_ms" if phase == "decode" else "client_ttft_ms"
        label = "client TPOT p50" if phase == "decode" else "client TTFT p50"
        lines.append(f"| {label} | " + " | ".join(f"{runs[n][cap][key]:.2f}" for n in cols) + " |")
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
