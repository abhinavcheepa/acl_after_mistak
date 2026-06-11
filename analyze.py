"""
analyze.py
==========
Training ke baad detailed analysis aur graphs generate karta hai.

Kya karta hai:
    1. Training curves compare karta hai (CRF vs Linear)
    2. Per-tag F1 analysis
    3. Method comparison (zero_shot vs lookback vs lbs)
    4. Author-wise performance (agar inference run hua ho)
    5. Stanza ke saath final comparison
    6. Complete analysis report save karta hai

Input files (jo pehle se hone chahiye):
    results/crf_metrics.json
    results/linear_metrics.json
    results/metrics.json        (evaluate_dual_pipeline.py ka output)

Output:
    results/analysis/01_training_curves.png
    results/analysis/02_method_comparison.png
    results/analysis/03_per_tag_f1.png
    results/analysis/04_crf_vs_linear.png
    results/analysis/05_final_summary.png
    results/analysis/analysis_report.txt
"""

import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

TAGS = ["ADJ","ADP","ADV","AUX","CCONJ","DET",
        "INTJ","NOUN","NUM","PART","PRON","PROPN",
        "PUNCT","SCONJ","SYM","VERB","X"]

STANZA_F1  = 0.9282
STANZA_ACC = 0.9791

# Paper ke results (ACL 2024)
PAPER_RESULTS = {
    "zero_shot": 0.93,
    "lookback":  0.94,
    "lbs":       0.95,
}


# ─────────────────────────────────────────────────────────────
# LOAD HELPERS
# ─────────────────────────────────────────────────────────────

def load_json(path):
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# PLOT 1: Training Curves
# ─────────────────────────────────────────────────────────────

def plot_training_curves(crf_metrics, linear_metrics, path):
    """CRF aur Linear ki training curves side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Curves — CRF vs Linear (Phase 1)", fontsize=14)

    datasets = []
    if crf_metrics:
        datasets.append(("CRF", crf_metrics["history"], "steelblue"))
    if linear_metrics:
        datasets.append(("Linear", linear_metrics["history"], "darkorange"))

    metrics_map = [
        ("train_loss", "Training Loss",    (0, 0)),
        ("val_f1",     "Validation F1",    (0, 1)),
        ("val_acc",    "Validation Acc",   (0, 2)),
        ("test_f1",    "Test F1",          (1, 0)),
        ("test_acc",   "Test Acc",         (1, 1)),
        ("lr",         "Learning Rate",    (1, 2)),
    ]

    for key, title, (r, c) in metrics_map:
        ax = axes[r][c]
        for name, history, color in datasets:
            if key in history:
                epochs = list(range(1, len(history[key]) + 1))
                ax.plot(epochs, history[key], label=name, color=color)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 2: Method Comparison
# ─────────────────────────────────────────────────────────────

def plot_method_comparison(eval_metrics, path):
    """6 methods ka F1 + Accuracy comparison."""
    if eval_metrics is None:
        print("  [SKIP] eval_metrics not available")
        return

    methods  = []
    f1s      = []
    accs     = []
    colors   = []

    for pipeline in ["crf", "linear"]:
        for method in ["zero_shot", "lookback", "lbs"]:
            if pipeline in eval_metrics and method in eval_metrics[pipeline]:
                vals = eval_metrics[pipeline][method]
                methods.append(f"{pipeline}\n{method}")
                f1s.append(vals["macro_f1"])
                accs.append(vals["accuracy"])
                colors.append("steelblue" if pipeline == "crf" else "darkorange")

    if not methods:
        print("  [SKIP] No eval metrics found")
        return

    x   = np.arange(len(methods))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - w/2, f1s,  w, color=colors, alpha=0.85, label="Macro F1")
    bars2 = ax.bar(x + w/2, accs, w, color=colors, alpha=0.45, label="Accuracy")

    # Paper reference lines
    for method, f1 in PAPER_RESULTS.items():
        ax.axhline(f1, linestyle=":", alpha=0.5,
                   label=f"Paper {method}={f1}")

    ax.axhline(STANZA_F1,  color="red",   linestyle="--", linewidth=2,
               label=f"Stanza F1={STANZA_F1}")
    ax.axhline(STANZA_ACC, color="green", linestyle="--", linewidth=2,
               label=f"Stanza Acc={STANZA_ACC}")

    # Value labels
    for bar, val in zip(bars1, f1s):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("All Methods — Comparison with Stanza & Paper")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue",  label="CRF Pipeline"),
        Patch(facecolor="darkorange", label="Linear Pipeline"),
    ]
    ax.legend(handles=legend_elements +
              [plt.Line2D([0], [0], color="red",   linestyle="--", label="Stanza F1"),
               plt.Line2D([0], [0], color="green", linestyle="--", label="Stanza Acc")],
              loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 3: CRF vs Linear — Training F1 over epochs
# ─────────────────────────────────────────────────────────────

def plot_crf_vs_linear(crf_metrics, linear_metrics, path):
    """CRF vs Linear — Val F1 aur Test F1 over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CRF vs Linear — Val & Test F1 over Epochs")

    for ax, key, title in [
        (axes[0], "val_f1",  "Validation F1"),
        (axes[1], "test_f1", "Test F1"),
    ]:
        if crf_metrics and key in crf_metrics["history"]:
            ep = list(range(1, len(crf_metrics["history"][key]) + 1))
            ax.plot(ep, crf_metrics["history"][key],
                    label="CRF", color="steelblue", linewidth=2)

        if linear_metrics and key in linear_metrics["history"]:
            ep = list(range(1, len(linear_metrics["history"][key]) + 1))
            ax.plot(ep, linear_metrics["history"][key],
                    label="Linear", color="darkorange", linewidth=2)

        ax.axhline(STANZA_F1, color="red", linestyle="--",
                   label=f"Stanza F1={STANZA_F1}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# PLOT 4: Final Summary
# ─────────────────────────────────────────────────────────────

def plot_final_summary(eval_metrics, crf_metrics, linear_metrics, path):
    """Ek hi figure mein sab kuch."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Phase 1 — Complete Analysis Summary", fontsize=16)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    for metrics, name, color in [
        (crf_metrics,    "CRF",    "steelblue"),
        (linear_metrics, "Linear", "darkorange"),
    ]:
        if metrics and "train_loss" in metrics["history"]:
            ep = list(range(1, len(metrics["history"]["train_loss"]) + 1))
            ax1.plot(ep, metrics["history"]["train_loss"],
                     label=f"{name} Train", color=color)
            ax1.plot(ep, metrics["history"]["val_loss"],
                     label=f"{name} Val",   color=color, linestyle="--")
    ax1.set_title("Loss Curves")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Val F1
    ax2 = fig.add_subplot(gs[0, 1])
    for metrics, name, color in [
        (crf_metrics,    "CRF",    "steelblue"),
        (linear_metrics, "Linear", "darkorange"),
    ]:
        if metrics and "val_f1" in metrics["history"]:
            ep = list(range(1, len(metrics["history"]["val_f1"]) + 1))
            ax2.plot(ep, metrics["history"]["val_f1"],
                     label=name, color=color, linewidth=2)
    ax2.axhline(STANZA_F1, color="red", linestyle="--",
                label=f"Stanza {STANZA_F1}")
    ax2.set_title("Validation F1")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Method F1 bar
    ax3 = fig.add_subplot(gs[0, 2])
    if eval_metrics:
        method_labels, method_f1s, method_colors = [], [], []
        for pipeline in ["crf", "linear"]:
            for method in ["zero_shot", "lookback", "lbs"]:
                if pipeline in eval_metrics and method in eval_metrics[pipeline]:
                    method_labels.append(f"{pipeline[:3]}\n{method[:4]}")
                    method_f1s.append(eval_metrics[pipeline][method]["macro_f1"])
                    method_colors.append("steelblue" if pipeline == "crf" else "darkorange")
        ax3.bar(method_labels, method_f1s, color=method_colors, alpha=0.8)
        ax3.axhline(STANZA_F1, color="red", linestyle="--")
        ax3.set_title("Method F1 Comparison")
        ax3.set_ylim(0.7, 1.0)
        ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Best F1 summary table
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")

    table_data = [["Model", "Method", "Macro F1", "Accuracy", "vs Stanza F1"]]
    if eval_metrics:
        for pipeline in ["crf", "linear"]:
            for method in ["zero_shot", "lookback", "lbs"]:
                if pipeline in eval_metrics and method in eval_metrics[pipeline]:
                    vals = eval_metrics[pipeline][method]
                    diff = vals["macro_f1"] - STANZA_F1
                    sign = "+" if diff >= 0 else ""
                    table_data.append([
                        pipeline.upper(),
                        method,
                        f"{vals['macro_f1']:.4f}",
                        f"{vals['accuracy']:.4f}",
                        f"{sign}{diff:.4f}",
                    ])
    table_data.append(["STANZA", "—", f"{STANZA_F1}", f"{STANZA_ACC}", "baseline"])

    tbl = ax4.table(cellText=table_data[1:],
                    colLabels=table_data[0],
                    cellLoc="center",
                    loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    ax4.set_title("Final Results Summary", fontsize=12, pad=20)

    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────────────

def write_report(eval_metrics, crf_metrics, linear_metrics, path):
    lines = []
    lines.append("=" * 70)
    lines.append("  ACL Hindi POS Tagger — Phase 1 Analysis Report")
    lines.append("=" * 70)

    # Training summary
    if crf_metrics:
        lines.append(f"\n[CRF Training]")
        lines.append(f"  Best Val F1 : {crf_metrics['best_val_f1']:.4f}")
        lines.append(f"  Epochs      : {crf_metrics['total_epochs']}")
        best_ep = crf_metrics["history"]["val_f1"].index(
            max(crf_metrics["history"]["val_f1"])) + 1
        lines.append(f"  Best Epoch  : {best_ep}")

    if linear_metrics:
        lines.append(f"\n[Linear Training]")
        lines.append(f"  Best Val F1 : {linear_metrics['best_val_f1']:.4f}")
        lines.append(f"  Epochs      : {linear_metrics['total_epochs']}")
        best_ep = linear_metrics["history"]["val_f1"].index(
            max(linear_metrics["history"]["val_f1"])) + 1
        lines.append(f"  Best Epoch  : {best_ep}")

    # Evaluation
    if eval_metrics:
        lines.append(f"\n[Evaluation on test.conll]")
        lines.append(f"  {'Method':<28} {'F1':>8} {'Acc':>8} {'vs Stanza':>10}")
        lines.append("  " + "-" * 58)
        for pipeline in ["crf", "linear"]:
            for method in ["zero_shot", "lookback", "lbs"]:
                if pipeline in eval_metrics and method in eval_metrics[pipeline]:
                    vals = eval_metrics[pipeline][method]
                    diff = vals["macro_f1"] - STANZA_F1
                    sign = "+" if diff >= 0 else ""
                    lines.append(
                        f"  {pipeline}_{method:<22} "
                        f"{vals['macro_f1']:>8.4f} "
                        f"{vals['accuracy']:>8.4f} "
                        f"{sign}{diff:>9.4f}"
                    )
        lines.append(f"  {'Stanza (baseline)':<28} "
                     f"{STANZA_F1:>8.4f} {STANZA_ACC:>8.4f} {'—':>10}")

    # Paper comparison
    lines.append(f"\n[ACL Paper Comparison (Hindi)]")
    lines.append(f"  Paper zero_shot F1 : 0.9300")
    lines.append(f"  Paper lookback  F1 : 0.9400")
    lines.append(f"  Paper lbs       F1 : 0.9500")

    lines.append("\n" + "=" * 70)
    lines.append("  Files saved:")
    for f in ["01_training_curves.png", "02_method_comparison.png",
              "03_crf_vs_linear.png", "04_final_summary.png"]:
        lines.append(f"    analysis/{f}")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  Report → {path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ACL POS Tagger — Analysis (Phase 1)")
    print("=" * 60)

    crf_metrics    = load_json(os.path.join(RESULTS_DIR, "crf_metrics.json"))
    linear_metrics = load_json(os.path.join(RESULTS_DIR, "linear_metrics.json"))
    eval_metrics   = load_json(os.path.join(RESULTS_DIR, "metrics.json"))

    print("\nGenerating plots...")

    plot_training_curves(
        crf_metrics, linear_metrics,
        os.path.join(ANALYSIS_DIR, "01_training_curves.png")
    )

    plot_method_comparison(
        eval_metrics,
        os.path.join(ANALYSIS_DIR, "02_method_comparison.png")
    )

    plot_crf_vs_linear(
        crf_metrics, linear_metrics,
        os.path.join(ANALYSIS_DIR, "03_crf_vs_linear.png")
    )

    plot_final_summary(
        eval_metrics, crf_metrics, linear_metrics,
        os.path.join(ANALYSIS_DIR, "04_final_summary.png")
    )

    write_report(
        eval_metrics, crf_metrics, linear_metrics,
        os.path.join(ANALYSIS_DIR, "analysis_report.txt")
    )

    print("\n✅ Analysis complete!")
    print(f"   All files in: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
