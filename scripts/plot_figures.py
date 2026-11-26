"""Plot the benchmark and profiling results.

Usage: python3 scripts/plot_figures.py     -> figures/*.png
Precision is the colour, engine is the line style / position; every figure has a table next to it in the README.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
SPEED = {("vllm", "bf16"): "speed/vllm_bf16", ("trtllm", "bf16"): "speed/trtllm_bf16",
         ("vllm", "fp8"): "speed/vllm_fp8", ("trtllm", "fp8"): "speed/trtllm_fp8",
         ("vllm", "nvfp4"): "speed/vllm_nvfp4", ("trtllm", "nvfp4"): "speed/trtllm_nvfp4"}
ENGINE = {"vllm": "vLLM", "trtllm": "TRT-LLM"}
PREC = {"bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4"}
COLOR = {"bf16": "#2a78d6", "fp8": "#eb6834", "nvfp4": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a65", "#e4e3de", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans"})


def speed(engine, prec):
    with open(RES / SPEED[engine, prec] / "summary.csv") as fh:
        return {(int(r["input"]), int(r["output"]), int(r["concurrency"])): r for r in csv.DictReader(fh)}


def profile(engine, prec):
    d = {}
    for pre in ("profile", "profile_sweep"):
        d.update(json.loads((RES / pre / f"{engine}_{prec}" / "breakdown.json").read_text()))
    return d


def style(ax, ylabel):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


# Figure 1: output throughput against concurrency, one panel per workload.
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, (n_in, n_out) in zip(axes, [(128, 512), (2048, 256)]):
    for prec in PREC:
        for engine, ls, mk in (("vllm", "-", "o"), ("trtllm", "--", "s")):
            s = speed(engine, prec)
            xs = [1, 4, 16, 64]
            ys = [float(s[n_in, n_out, c]["output_throughput"]) for c in xs]
            ax.plot(xs, ys, ls, color=COLOR[prec], marker=mk, markersize=6, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            if engine == "vllm":
                # FP8 and NVFP4 end close together: nudge the two labels apart.
                dy = {"nvfp4": 7, "fp8": -7}.get(prec, 0) * (1 if n_in == 128 else -1)
                ax.annotate(PREC[prec], (64, ys[-1]), xytext=(8, dy), textcoords="offset points", va="center", color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
    ax.set_xlim(0.8, 110)
    ax.set_xlabel("concurrent requests")
    ax.set_title(f"{n_in} tokens in / {n_out} out", loc="left", fontsize=10.5)
    style(ax, "output tokens / s (log)")
handles = [plt.Line2D([], [], color=MUTED, linestyle="-", marker="o", label="vLLM 0.31.0"),
           plt.Line2D([], [], color=MUTED, linestyle="--", marker="s", label="TensorRT-LLM 1.4.0rc2")]
axes[0].legend(handles=handles, frameon=False, loc="upper left")
fig.suptitle("Output throughput, Qwen3-8B on one RTX 4090", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "throughput.png", dpi=160)
plt.close(fig)

# Figure 2: where one batch-1 decode step goes (kernel time from the traces).
PARTS = [("linear", "linear layers", "#2a78d6"), ("attention", "attention", "#eb6834"),
         ("rest", "norm, activation, sampling, other", "#1baf7a"), ("idle", "GPU idle", "#c9c8c2")]
rows = [(e, p) for p in PREC for e in ("vllm", "trtllm")]
fig, ax = plt.subplots(figsize=(10.5, 3.9))
for i, (e, p) in enumerate(rows):
    dec = profile(e, p)["c1_in128"]["decode"]
    parts = dec["parts"]
    vals = {"linear": parts.get("linear", 0), "attention": parts.get("attention", 0),
            "idle": dec["wall_ms"] - dec["kernel_ms"]}
    vals["rest"] = dec["kernel_ms"] - vals["linear"] - vals["attention"]
    left = 0
    for key, _, color in PARTS:
        ax.barh(i, vals[key], left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2)
        left += vals[key]
    ax.text(left + 0.15, i, f"{dec['wall_ms']:.1f} ms   (attention {vals['attention']:.2f})", va="center", color=INK)
ax.set_yticks(range(len(rows)), [f"{PREC[p]}  {ENGINE[e]}" for e, p in rows])
ax.invert_yaxis()
ax.set_xlim(0, 22)
ax.set_xlabel("ms per decode step (batch 1, 128-token context)")
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c in PARTS], frameon=False,
          loc="lower right", ncol=1)
ax.set_title("Where one decode step goes", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "decode_step.png", dpi=160)
plt.close(fig)

# Figure 3: TRT-LLM FP8 step time by batch size, default and with the XQA kernel forced, against vLLM.
caps = [("c1_in128", "1"), ("c2_in128", "2"), ("c4_in128", "4")]
forced = json.loads((RES / "profile_force_xqa" / "trtllm_fp8" / "breakdown.json").read_text())
series = [("vLLM FP8", profile("vllm", "fp8"), "#2a78d6"), ("TRT-LLM FP8, default", profile("trtllm", "fp8"), "#eb6834"),
          ("TRT-LLM FP8, TRTLLM_FORCE_XQA=1", forced, "#1baf7a")]
fig, ax = plt.subplots(figsize=(7.2, 3.9))
w = 0.26
for k, (label, data, color) in enumerate(series):
    for j, (cap, _) in enumerate(caps):
        v = data[cap]["decode"]["wall_ms"]
        ax.bar(j + (k - 1) * w, v, width=w - 0.03, color=color, label=label if j == 0 else None)
        ax.text(j + (k - 1) * w, v + 0.15, f"{v:.1f}", ha="center", color=INK, fontsize=9)
ax.set_xticks(range(len(caps)), [f"batch {b}" for _, b in caps])
ax.set_ylim(0, 19)
style(ax, "ms per decode step")
ax.legend(frameon=False, loc="upper left")
ax.set_title("FP8 KV cache on SM89: attention kernel choice at small batch", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fp8_attention_fallback.png", dpi=160)
plt.close(fig)
print("wrote", *(p.name for p in sorted(FIG.glob("*.png"))))
