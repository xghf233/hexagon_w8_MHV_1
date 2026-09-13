"""Generate a training report (markdown + loss curves) for a heptagon run directory.

Usage:
    python make_report.py /root/autodl-tmp/runs/heptagon/w6-random-seed42 \
        /root/autodl-tmp/runs/heptagon/reports/w6-random-seed42

Reads run.json / baseline.json / history.jsonl / summary.json and writes
report.md + training_curves.png into the report directory (which must not exist).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_events(history: Path):
    events = {"train": [], "val_subset": [], "val_full": [], "train_diagnostic": [], "checkpoint": []}
    with history.open() as stream:
        for line in stream:
            record = json.loads(line)
            events[record["event"]].append(record)
    return events


def ema(values, alpha=0.05):
    out = []
    running = None
    for value in values:
        running = value if running is None else alpha * value + (1 - alpha) * running
        out.append(running)
    return out


def main():
    run_dir = Path(sys.argv[1]).resolve()
    report_dir = Path(sys.argv[2]).resolve()
    if report_dir.exists():
        raise FileExistsError(f"Report directory exists: {report_dir}")
    report_dir.mkdir(parents=True)

    config = json.load(open(run_dir / "run.json"))
    baseline = json.load(open(run_dir / "baseline.json"))
    summary = json.load(open(run_dir / "summary.json"))
    events = load_events(run_dir / "history.jsonl")

    train = events["train"]
    steps = [r["completed_steps"] for r in train]
    losses = [r["loss"] for r in train]
    grad_norms = [r["grad_norm"] for r in train]
    lr0 = [r["lr_groups"][0] for r in train]
    lr1 = [r["lr_groups"][1] for r in train]
    lr2 = [r["lr_groups"][2] for r in train]
    vals = events["val_subset"]
    fulls = events["val_full"]
    diags = events["train_diagnostic"]

    # ---- figure: 2x2 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(steps, losses, color="#9db8d9", lw=0.6, alpha=0.6, label="per-step (logged every 100)")
    ax.plot(steps, ema(losses), color="#1f4e8c", lw=1.6, label="EMA (alpha=0.05)")
    ax.set_title("Training loss (bf16, batch 512)")
    ax.set_xlabel("completed steps")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, grad_norms, color="#7a4d1e", lw=0.7)
    ax.set_title("Gradient norm (after clip 1.0)")
    ax.set_xlabel("completed steps")
    ax.set_ylabel("||g||")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, lr0, lw=1.0, label="matrix lr (max 3e-4)")
    ax.plot(steps, lr1, lw=1.0, label="embedding lr (max 0.2)")
    ax.plot(steps, lr2, lw=1.0, label="unembedding lr (max 0.004)")
    ax.set_title("Learning rate groups (warmup 500 / warmdown 20%)")
    ax.set_xlabel("completed steps")
    ax.set_ylabel("lr")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    base_exact = baseline["val_full"]["exact_accuracy"]
    if vals:
        vs = [r["completed_steps"] for r in vals]
        ax.plot(vs, [r["exact_accuracy"] for r in vals], "o-", color="#1f4e8c", label="val subset exact (4096)")
        ax.plot(vs, [r["magnitude_accuracy"] for r in vals], "s-", color="#3a8c4f", label="val subset magnitude")
        ax.plot(vs, [r["sign_accuracy"] for r in vals], "^-", color="#b3453a", label="val subset sign")
    if fulls:
        fs = [r["completed_steps"] for r in fulls]
        ax.plot(fs, [r["exact_accuracy"] for r in fulls], "D", ms=8, color="#f2a900", label="FULL val exact (46,725)")
    ax.axhline(base_exact, color="gray", ls="--", lw=1,
               label=f"baseline majority exact {base_exact:.4f}")
    ax.set_title("Free-generation accuracy on val")
    ax.set_xlabel("completed steps")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.savefig(report_dir / "training_curves.png", dpi=130)
    plt.close(fig)

    # ---- markdown ----
    lines = []
    lines.append(f"# Heptagon w6 MHV 正式训练报告 — {run_dir.name}")
    lines.append("")
    lines.append(f"- 生成时间：2026-09-07（服务器 /root/autodl-tmp）")
    lines.append(f"- 运行目录：`{run_dir}`")
    lines.append(f"- 配置：`{config['config']['model']}`，device_batch_size={config['config']['device_batch_size']}，"
                 f"max_steps={config['config']['max_steps']}，seed={config['config']['seed']}，compile={config['config']['compile']}")
    lines.append(f"- GPU：{config['gpu']}，CUDA {config['cuda']}；参数量 {config['n_parameters']:,}")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 完成步数 | {summary['completed_steps']} / {config['config']['max_steps']} |")
    lines.append(f"| 样本呈现 | {summary['rows_consumed']:,}（{summary['train_passes']:.2f} 轮） |")
    lines.append(f"| 训练耗时 | {summary['elapsed_seconds'] / 60:.1f} 分钟 |")
    lines.append(f"| 峰值显存 | {summary['peak_cuda_bytes'] / 2**30:.2f} GiB |")
    lines.append(f"| acceptance | {summary['acceptance']} |")
    lines.append(f"| 最后 train loss | {losses[-1]:.4f}（第 {steps[-1]} 步） |")
    lines.append(f"| 最后 train 诊断 exact | {diags[-1]['exact_accuracy']:.4f}（{diags[-1]['n_samples']} 样本） |")
    lines.append("")
    if summary["best"] is not None:
        b = summary["best"]
        lines.append(f"## Best（仅全量 val exact 改善时更新）")
        lines.append("")
        lines.append(f"- 全量 val exact = **{b['exact_accuracy']:.4f}** @ 第 {b['completed_steps']} 步")
        lines.append(f"- checkpoint：`{b['checkpoint']}`")
        lines.append("")
    else:
        lines.append("best: null（无全量 val 改善）")
        lines.append("")
    lines.append(f"基线（train 多数类预测器 {baseline['predictor']}）：val_full exact = {base_exact:.4f}，"
                 f"magnitude = {baseline['val_full']['magnitude_accuracy']:.4f}，"
                 f"sign = {baseline['val_full']['sign_accuracy']:.4f}")
    lines.append("")
    lines.append("## 曲线")
    lines.append("")
    lines.append("![training_curves](training_curves.png)")
    lines.append("")
    if fulls:
        lines.append("## 全量 val（46,725 项，每 18,250 步）")
        lines.append("")
        lines.append("| 步数 | exact | magnitude | sign | invalid |")
        lines.append("|---|---|---|---|---|")
        for r in fulls:
            lines.append(f"| {r['completed_steps']} | {r['exact_accuracy']:.4f} | {r['magnitude_accuracy']:.4f} | "
                         f"{r['sign_accuracy']:.4f} | {r['invalid_rate']:.4f} |")
        lines.append("")
    lines.append("## 子集 val（4096 项，每 3,650 步）末次")
    lines.append("")
    if vals:
        r = vals[-1]
        lines.append(f"第 {r['completed_steps']} 步：exact={r['exact_accuracy']:.4f}，"
                     f"magnitude={r['magnitude_accuracy']:.4f}，sign={r['sign_accuracy']:.4f}，invalid={r['invalid_rate']:.4f}")
    lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_dir / 'report.md'} and {report_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
