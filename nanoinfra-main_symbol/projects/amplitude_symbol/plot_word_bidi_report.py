"""Reproduce the word_bidi report figure for a training run.

Recreates the layout of word_bidi_report.png (2382x1530, 2x2 panels + gray
metrics box in the bottom-left corner):

  TL: training loss on log scale (blue #5182EF, pink vertical gridlines
      every 20K steps, "magnitude learning" / "sign learning" annotations)
  TR: MFU % per step (teal #4FB595)
  BL: validation accuracies over training (exact #2563EB, magnitude #059669,
      sign #7C3AED, legend lower-right)
  BR: orbit consistency (red #DC2626 horizontal line + value label)
  Gray box #F8FAFC: final metrics (512 val samples) + training stats

Usage:
    python -m projects.amplitude_symbol.plot_word_bidi_report \
        --history outputs/amplitude_symbol/word_bidi_150k_2026-08-17/training_history.jsonl \
        --eval outputs/amplitude_symbol/word_bidi_150k_2026-08-17/evaluation_val.json \
        --out projects/amplitude_symbol/reports/figures/word_bidi_150k_report.png \
        --title "Word-Bidirectional Attention Experiment - 150K Step Training Report" \
        --loss-title "Training Loss — 150K Steps" \
        --tick 50000 --vline-every 20000 --peak-gpu-mem "~1.8 GB"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

# Palette measured from word_bidi_report.png
C_LOSS = "#5182EF"      # blue-500
C_MFU = "#4FB595"       # teal-500
C_EXACT = "#2563EB"     # blue-600
C_MAG = "#059669"       # emerald-600
C_SIGN = "#7C3AED"      # violet-600
C_RED = "#DC2626"       # red-600
C_VGRID = "#F8D3D3"     # pink vertical gridlines
C_HGRID = "#F1F1F1"     # light-gray horizontal gridlines
C_BOX = "#F8FAFC"       # metrics box face (slate-50)
C_BOX_EDGE = "#94A3B8"
C_TEXT = "#2A2A2A"


def load_history(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    train = [r for r in rows if r.get("type") != "eval"]
    evals = [r for r in rows if r.get("type") == "eval"]
    if not train:
        raise ValueError(f"No training records found in {path}")
    if not evals:
        raise ValueError(f"No evaluation records found in {path}")

    train_fields = {"step", "loss", "mfu_pct", "tok_per_s", "total_time_min"}
    missing_train = sorted(train_fields - train[0].keys())
    if missing_train:
        raise ValueError(f"Training history is missing fields: {missing_train}")

    eval_fields = {
        "step",
        "val/exact_accuracy",
        "val/magnitude_accuracy",
        "val/sign_accuracy",
        "val/orbit_consistency",
    }
    missing_eval = sorted(eval_fields - evals[0].keys())
    if missing_eval:
        raise ValueError(f"Evaluation history is missing fields: {missing_eval}")

    return {
        "steps": np.array([r["step"] for r in train]),
        "loss": np.array([r["loss"] for r in train]),
        "mfu": np.array([r["mfu_pct"] for r in train]),
        "tok_per_s": np.array([r["tok_per_s"] for r in train]),
        "time_min": train[-1]["total_time_min"],
        "eval_steps": np.array([r["step"] for r in evals]),
        "eval": {k: np.array([r[k] for r in evals])
                 for k in ("val/exact_accuracy", "val/magnitude_accuracy",
                           "val/sign_accuracy", "val/orbit_consistency")},
    }


def load_final_eval(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    sm, om = d["sample_metrics"], d["orbit_metrics"]
    return {
        "n_samples": int(d.get("meta", {}).get("n_samples", sm["n_samples"])),
        "exact": 100 * sm["exact_accuracy"],
        "mag": 100 * sm["magnitude_accuracy"],
        "sign": 100 * sm["sign_accuracy"],
        "invalid": 100 * sm["invalid_output_rate"],
        "orbit": 100 * om["orbit_consistency"],
    }


def uniform_sample_indices(length: int, max_points: int = 1000) -> np.ndarray:
    """Return stable indices including both endpoints for a plotted series."""
    if length <= 0:
        raise ValueError("Cannot sample an empty series")
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=int))


def style_axes(ax, yticks):
    ax.set_facecolor("white")
    ax.grid(True, which="major", axis="y", color=C_HGRID, lw=1.0, zorder=0)
    ax.grid(False, axis="x")
    for s in ax.spines.values():
        s.set_color("#000000")
    ax.tick_params(colors=C_TEXT, labelsize=10.5)
    ax.set_yticks(yticks)
    ax.set_axisbelow(True)


def plot_report(h, final, *, out_path, title, loss_title, tick, vline_every,
                peak_gpu_mem, annotate_collapse=False):
    max_steps = int(h["steps"][-1])
    plt.rcParams["text.color"] = C_TEXT
    plt.rcParams["axes.labelcolor"] = C_TEXT
    plt.rcParams["axes.titlecolor"] = C_TEXT

    fig = plt.figure(figsize=(15.9, 10.2), dpi=150, facecolor="white")
    fig.suptitle(title, fontsize=18, y=0.98, fontweight="bold")

    gs = fig.add_gridspec(2, 2, left=0.0462, right=0.9933, top=0.909,
                          bottom=0.0588, wspace=0.1035, hspace=0.2088)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    x_ticks = [tick, 2 * tick, 3 * tick]

    # ---------------- TL: training loss (log) -------------------------
    loss = np.clip(h["loss"], 1.4e-3, None)   # floor: zero-loss tail sits on the bottom edge
    ax1.plot(h["steps"], loss, color=C_LOSS, lw=1.2, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylim(1.3e-3, 4.3e2)
    ax1.set_yticks([1e2, 1e1, 1, 1e-1, 1e-2],
                   labels=["100", "10", "1", "0.1", "0.01"])
    ax1.set_xlim(-0.05 * max_steps, 1.05 * max_steps)
    ax1.set_xticks(x_ticks)
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Training Loss (log scale)", fontsize=12)
    ax1.set_title(loss_title, fontsize=13.5)
    for v in range(vline_every, max_steps - 5_000, vline_every):
        ax1.axvline(v, color=C_VGRID, lw=1.0, zorder=0)
    ax1.grid(True, which="major", axis="y", color=C_HGRID, lw=1.0, zorder=0)
    ax1.grid(False, axis="x")
    ax1.set_axisbelow(True)
    # annotations (same wording/colors as the 500K figure)
    ax1.text(8_000, 30, "magnitude\nlearning", color=C_MAG, fontsize=10.5,
             ha="left", va="center")
    ax1.text(max_steps * 0.30, 0.15, "sign learning", color=C_SIGN, fontsize=10.5,
             ha="left", va="center")
    if annotate_collapse:
        zero_steps = h["steps"][h["loss"] == 0]
        if len(zero_steps):
            zero_step = int(zero_steps.min())
            ax1.annotate(f"loss=0 from {zero_step // 1000}K", xy=(zero_step, 1.4e-3),
                         xytext=(zero_step - 25_000, 5e-3), color=C_RED, fontsize=10.5,
                         arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0))

    # ---------------- TR: MFU -----------------------------------------
    # Step-like (piecewise constant), as in the original figure: MFU is
    # logged per step but drawn with steps-post.
    mfu_indices = uniform_sample_indices(len(h["steps"]))
    ax2.plot(h["steps"][mfu_indices], h["mfu"][mfu_indices], color=C_MFU, lw=1.2,
             drawstyle="steps-post", zorder=3)
    ax2.set_ylim(0, 30)
    style_axes(ax2, [10, 15, 20, 25, 30])
    ax2.set_xlim(-0.05 * max_steps, 1.05 * max_steps)
    ax2.set_xticks(x_ticks)
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("MFU (%)", fontsize=12)
    avg_mfu = float(h["mfu"].mean())
    peak_mfu = float(h["mfu"].max())
    ax2.set_title(f"MFU - avg={avg_mfu:.1f}%, peak={peak_mfu:.1f}%", fontsize=13.5)

    # ---------------- BL: validation accuracy --------------------------
    for key, color in (("val/exact_accuracy", C_EXACT),
                       ("val/magnitude_accuracy", C_MAG),
                       ("val/sign_accuracy", C_SIGN)):
        ax3.plot(h["eval_steps"], 100 * h["eval"][key], color=color, lw=1.8, zorder=3)
    ax3.set_ylim(0, 100)
    style_axes(ax3, [20, 40, 60, 80, 100])
    ax3.set_xlim(-0.05 * max_steps, 1.05 * max_steps)
    ax3.set_xticks(x_ticks)
    ax3.set_xlabel("Step", fontsize=12)
    ax3.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax3.set_title("Validation Accuracy over Training", fontsize=13.5)
    handles = [Line2D([0], [0], color=c, lw=2.4) for c in (C_EXACT, C_MAG, C_SIGN)]
    ax3.legend(handles, ["Exact", "Magnitude", "Sign"],
               loc="lower right", frameon=False, fontsize=10.5,
               handlelength=1.3, borderaxespad=0.6)

    # ---------------- BR: orbit consistency ----------------------------
    orbit = final["orbit"]
    ax4.axhline(orbit, color=C_RED, lw=2.6, zorder=3,
                xmin=0.04, xmax=0.97)
    ax4.text(max_steps * 0.80, orbit + 2.2, f"{orbit:.1f}%", color=C_RED,
             fontsize=10.5, ha="left", va="bottom")
    ax4.set_ylim(0, 100)
    style_axes(ax4, [20, 40, 60, 80, 100])
    ax4.set_xlim(-0.05 * max_steps, 1.05 * max_steps)
    ax4.set_xticks([0] + x_ticks)
    ax4.set_xlabel("Step", fontsize=12)
    ax4.set_ylabel("Orbit Consistency (%)", fontsize=12)
    ax4.set_title(f"Orbit Consistency - final={orbit:.1f}%", fontsize=13.5)

    # ---------------- gray metrics box (figure coords) ------------------
    box = Rectangle((0.0134, 0.0052), 0.191, 0.201, transform=fig.transFigure,
                    facecolor=C_BOX, edgecolor=C_BOX_EDGE, lw=1.0, zorder=6)
    fig.patches.append(box)

    def box_line(y, left, right, bold=False):
        fig.text(0.019, y, left, fontsize=10, fontweight="bold" if bold else "normal",
                 ha="left", va="center", zorder=7)
        if right:
            fig.text(0.196, y, right, fontsize=10, ha="right", va="center", zorder=7)

    avg_tps = float(h["tok_per_s"].mean())
    y = 0.196
    box_line(y, f"Final Metrics ({final['n_samples']:,} val samples):", None, bold=True)
    for label, val in (("Exact Accuracy:", f"{final['exact']:.1f}%"),
                       ("Magnitude Accuracy:", f"{final['mag']:.1f}%"),
                       ("Sign Accuracy:", f"{final['sign']:.1f}%"),
                       ("Invalid Rate:", f"{final['invalid']:.1f}%"),
                       ("Orbit Consistency:", f"{final['orbit']:.1f}%")):
        y -= 0.0137
        box_line(y, label, val)
    y -= 0.0225
    box_line(y, "Training Stats:", None, bold=True)
    for label, val in (("Total Steps:", f"{max_steps:,}"),
                       ("Total Time:", f"{h['time_min']:.1f} min ({h['time_min'] / 60:.1f} hrs)"),
                       ("Avg Throughput:", f"{avg_tps / 1000:.0f}K tok/s"),
                       ("Avg MFU:", f"{avg_mfu:.1f}%"),
                       ("Peak GPU Memory:", peak_gpu_mem)):
        y -= 0.0137
        box_line(y, label, val)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--loss-title", required=True)
    parser.add_argument("--tick", type=int, default=50_000)
    parser.add_argument("--vline-every", type=int, default=20_000)
    parser.add_argument("--peak-gpu-mem", default="~1.8 GB")
    parser.add_argument("--annotate-collapse", action="store_true")
    args = parser.parse_args()

    h = load_history(args.history)
    final = load_final_eval(args.eval)
    plot_report(h, final, out_path=args.out, title=args.title,
                loss_title=args.loss_title, tick=args.tick,
                vline_every=args.vline_every, peak_gpu_mem=args.peak_gpu_mem,
                annotate_collapse=args.annotate_collapse)


if __name__ == "__main__":
    main()
