"""Break the captures (profile.sh) down into GPU kernel time per decode step and per prefill.

Usage: python3 scripts/profile_breakdown.py results/<tag>
Reads <tag>/captures.tsv and, per capture, <name>.sqlite (nsys export) and <name>.json (client result);
writes <tag>/breakdown.md and breakdown.json (the same numbers, for profile_summary.py).

A decode step is the stretch between two runs of the sampling kernel (once per step, whatever the batch).
The median step is reported, so steps that also hold a prefill or the pause between requests drop out.
The prefill of a request is everything before its first sampled token (only for one-at-a-time captures).
"""
import json
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PARTS = [  # first match wins
    ("attention", r"mha|fmha|flash_fwd|BatchDecode|BatchPrefill|UpdateKVCache|QKNormRope|[Rr]ope|"
                  r"paged_kv|PagedKV|kv_cache|KVCache|reshape_and_cache"),
    ("quantize", r"quant|Quant"),
    ("norm", r"[Nn]orm"),
    ("activation", r"silu|gelu|act_and_mul"),
    ("linear", r"gemv|gemm|scaled_mm|marlin|Marlin|cutlass|sgemm"),
    ("sampling", r"[Ss]ampling|topk|TopK|sort|Sort|multinomial|[Ss]oft[Mm]ax|cumsum|argmax"),
]
# Kernels that run once per decode step, in order of preference.
STEP_MARKERS = [r"fusedSamplingKernel", r"TopPSamplingFromProb", r"SamplingFromProb", r"cunn_SoftMaxForward"]


def part(name):
    for p, pat in PARTS:
        if re.search(pat, name):
            return p
    return "other"


def short(name):
    """Kernel name without return type, template arguments and parameters."""
    name = name.replace("void ", "").replace("<unnamed>::", "")
    name = re.sub(r"^std::enable_if<.*?>::type ", "", name)
    return re.sub(r"[(<].*", "", name)[:70]


def total(kernels):
    by_part, by_kernel = defaultdict(int), defaultdict(lambda: [0, 0])
    for s, e, n in kernels:
        by_part[part(n)] += e - s
        by_kernel[n][0] += 1
        by_kernel[n][1] += e - s
    return by_part, by_kernel


def tables(groups, wall, unit, record):
    """groups: list of (by_part, by_kernel); medians across them, as two markdown tables.
    The same numbers (in ms) go into the dict `record`."""
    med = statistics.median
    parts = {p for bp, _ in groups for p in bp}
    rows = sorted(((p, med(bp.get(p, 0) for bp, _ in groups)) for p in parts), key=lambda r: -r[1])
    busy = med(sum(bp.values()) for bp, _ in groups)
    record.update(wall_ms=wall / 1e6, kernel_ms=busy / 1e6, parts={p: v / 1e6 for p, v in rows})
    out = [f"| part | ms per {unit} | % of {unit} |", "|---|---|---|"]
    out += [f"| {p} | {v / 1e6:.3f} | {100 * v / wall:.1f}% |" for p, v in rows]
    out += [f"| GPU idle | {(wall - busy) / 1e6:.3f} | {100 * (wall - busy) / wall:.1f}% |", "",
            f"| kernel | part | runs per {unit} | ms per {unit} |", "|---|---|---|---|"]
    names = {n for _, bk in groups for n in bk}
    kern = {n: (med(bk[n][0] if n in bk else 0 for _, bk in groups),
                med(bk[n][1] if n in bk else 0 for _, bk in groups)) for n in names}
    record["kernels"] = []
    for n, (cnt, ns) in sorted(kern.items(), key=lambda kv: -kv[1][1])[:12]:
        if ns > 0:
            out.append(f"| `{short(n)}` | {part(n)} | {cnt:g} | {ns / 1e6:.3f} |")
            record["kernels"].append({"name": short(n), "part": part(n), "runs": cnt, "ms": ns / 1e6})
    return out


root = Path(sys.argv[1])
records = {}
lines = [f"# {root.name}: GPU kernel time (Nsight Systems, CUDA graphs traced per node)", "",
         "```", (root / "env.txt").read_text().strip(), "```"]
for row in (root / "captures.tsv").read_text().splitlines():
    name, n_in, n_out, prompts, conc = row.split("\t")
    n_out, prompts, conc = int(n_out), int(prompts), int(conc)
    client = json.loads((root / f"{name}.json").read_text())
    db = sqlite3.connect(f"file:{root / (name + '.sqlite')}?mode=ro", uri=True)
    kernels = db.execute("""select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k
                            join StringIds s on s.id = k.demangledName order by k.start""").fetchall()
    counts = Counter(n for _, _, n in kernels)
    rec = records[name] = {"input": int(n_in), "output": n_out, "prompts": prompts, "concurrency": conc,
                           "client_tpot_ms": client["median_tpot_ms"], "client_ttft_ms": client["median_ttft_ms"]}
    marker = next((n for pat in STEP_MARKERS for n in sorted(counts) if re.search(pat, n)), None)
    lines += ["", f"## {name}: {prompts} requests, {n_in} in / {n_out} out, {conc} at a time", "",
              f"Client: TPOT p50 {client['median_tpot_ms']:.2f} ms, TTFT p50 {client['median_ttft_ms']:.1f} ms, "
              f"{client['completed']} completed. {len(kernels)} kernels recorded."]
    if marker is None:
        lines += ["", "No sampling kernel found; cannot cut the capture into steps."]
        continue
    cuts = [(s, e) for s, e, n in kernels if n == marker]
    if conc == 1 and len(cuts) != prompts * n_out:
        lines += ["", f"WARNING: `{short(marker)}` ran {len(cuts)} times, expected {prompts * n_out}."]

    steps, i = [], 0
    for (a, _), (b, _) in zip(cuts, cuts[1:]):
        while i < len(kernels) and kernels[i][0] < a:
            i += 1
        j = i
        while j < len(kernels) and kernels[j][0] < b:
            j += 1
        steps.append((b - a, total(kernels[i:j])))
    wall = statistics.median(w for w, _ in steps)
    kept = [t for w, t in steps if 0.7 * wall < w < 1.3 * wall]
    busy = statistics.median(sum(bp.values()) for bp, _ in kept)
    lines += ["", f"**Decode step** (cut at `{short(marker)}`; median of {len(kept)} of {len(steps)} steps): "
              f"{wall / 1e6:.2f} ms, of which kernels {busy / 1e6:.2f} ms ({100 * busy / wall:.1f}%).", ""]
    rec["decode"] = {}
    lines += tables(kept, wall, "step", rec["decode"])

    if conc == 1 and len(cuts) == prompts * n_out:
        groups, walls = [], []
        for r in range(prompts):
            lo = cuts[r * n_out - 1][1] if r else 0
            ks = [k for k in kernels if lo < k[0] < cuts[r * n_out][0]]
            walls.append(cuts[r * n_out][0] - ks[0][0])
            groups.append(total(ks))
        wall = statistics.median(walls)
        busy = statistics.median(sum(bp.values()) for bp, _ in groups)
        lines += ["", f"**Prefill** (first kernel of a request to its first sampled token; median of {prompts} "
                  f"requests): {wall / 1e6:.1f} ms, of which kernels {busy / 1e6:.1f} ms ({100 * busy / wall:.1f}%).", ""]
        rec["prefill"] = {}
        lines += tables(groups, wall, "prefill", rec["prefill"])
(root / "breakdown.md").write_text("\n".join(lines) + "\n")
(root / "breakdown.json").write_text(json.dumps(records, indent=1) + "\n")
print("\n".join(lines))
