"""Summarize the GSM8K quality check (lm-eval output written by quality_gsm8k.sh).

Usage: python3 scripts/summarize_quality.py results/<prefix>
Reads <prefix>/<engine>_<precision>/lm_eval/**/{results_*.json,samples_gsm8k_*.jsonl.xz}; writes <prefix>/summary.md.
Accuracy alone can hide changes that cancel out, so runs are also compared question by question:
each precision against BF16 on the same engine, and the two engines at the same precision
(McNemar exact test on the questions only one of the pair got right).
<engine>_<precision>_rerun is that run a second time with nothing changed: greedy answers still depend on which
requests share a batch, so its row shows how many answers change by themselves. That noise floor differs by
precision (the quantized checkpoints change more answers between reruns than BF16), so a rerun row is only
the floor for comparisons that involve its own precision.
"""
import json
import lzma
import math
import sys
from pathlib import Path

FILTERS = ["strict-match", "flexible-extract"]
ENGINES = ["vllm", "trtllm"]
PRECISIONS = ["bf16", "fp8", "nvfp4"]

root = Path(sys.argv[1])
runs = {}
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    results = sorted(d.glob("lm_eval/**/results_*.json"))
    samples = sorted(d.glob("lm_eval/**/samples_gsm8k_*.jsonl.xz"))
    if not results or not samples:
        continue
    # File names end in a timestamp, so the last one is the newest run.
    r = json.loads(results[-1].read_text())
    g = r["results"]["gsm8k"]
    correct = {f: {} for f in FILTERS}  # filter -> doc_id -> right or wrong
    text = {}                            # doc_id -> generated text
    # Split on "\n" only: generated text can contain other line separators (e.g. U+2028), which
    # splitlines() would also split on, cutting a JSON record in two.
    for line in lzma.open(samples[-1], "rt", encoding="utf-8").read().split("\n"):
        if not line:
            continue
        s = json.loads(line)
        correct[s["filter"]][s["doc_id"]] = s["exact_match"] == 1
        text[s["doc_id"]] = s["resps"][0][0]
    runs[d.name] = {"acc": {f: (g[f"exact_match,{f}"], g[f"exact_match_stderr,{f}"]) for f in FILTERS},
                    "n": r["n-samples"]["gsm8k"]["effective"], "correct": correct, "text": text}


def mcnemar_p(b, c):
    """Two-sided exact McNemar p-value: b and c are the questions only one of the two runs got right."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def compare(a, b, f="strict-match"):
    ca, cb = runs[a]["correct"][f], runs[b]["correct"][f]
    docs = sorted(ca.keys() & cb.keys())
    only_a = sum(ca[i] and not cb[i] for i in docs)
    only_b = sum(cb[i] and not ca[i] for i in docs)
    both = sum(ca[i] and cb[i] for i in docs)
    same = sum(runs[a]["text"][i] == runs[b]["text"][i] for i in docs)
    return [f"{a} vs {b}", str(both), str(only_a), str(only_b), str(len(docs) - both - only_a - only_b),
            f"{mcnemar_p(only_a, only_b):.2f}", f"{100 * same / len(docs):.0f}%"]


order = [n for e in ENGINES for p in PRECISIONS for n in (f"{e}_{p}", f"{e}_{p}_rerun") if n in runs]
lines = [f"# {root.name}: GSM8K (lm-eval, 5-shot, greedy, /v1/completions)", "",
         "Accuracy ± lm-eval's standard error.", "",
         "| run | questions | strict-match | flexible-extract |", "|---|---|---|---|"]
for name in order:
    acc = runs[name]["acc"]
    lines.append(f"| {name} | {runs[name]['n']} | "
                 + " | ".join(f"{100 * v:.1f}% ± {100 * se:.1f}" for v, se in (acc[f] for f in FILTERS)) + " |")

pairs = [(f"{e}_{p}", f"{e}_{p}_rerun") for e in ENGINES for p in PRECISIONS]           # same setup run twice
pairs += [(f"{e}_bf16", f"{e}_{p}") for e in ENGINES for p in PRECISIONS[1:]]           # BF16 vs quantized
pairs += [(f"vllm_{p}", f"trtllm_{p}") for p in PRECISIONS]                             # engine vs engine
rows = [compare(a, b) for a, b in pairs if a in runs and b in runs]
if rows:
    lines += ["", "Question by question (strict-match): \"only A\" = right in the first run only, "
              "p = McNemar exact test (small p: the difference is unlikely to be run-to-run luck), "
              "same text = identical generated output. A *_rerun row is the same setup run twice.", "",
              "| pair (A vs B) | both right | only A | only B | both wrong | p | same text |",
              "|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
(root / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
