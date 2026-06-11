"""
evaluate_dual_pipeline.py
Dono pipelines (CRF + Linear) ke teeno methods ka F1/Accuracy comparison.
Results → results/dual_pipeline_report.txt  +  results/metrics.json
         results/pipeline_f1_bars.png
         results/per_tag_f1.png
         results/dual_pipeline_comparison.png
"""

import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "writers who died before 1965")
OUT_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS   = ["zero_shot", "lookback", "lbs", "lbs_weighted"]
PIPELINES = ["crf", "linear"]

STANZA_F1  = 0.9282
STANZA_ACC = 0.9791

ALL_TAGS = ["ADJ","ADP","ADV","AUX","CCONJ","DET",
            "INTJ","NOUN","NUM","PART","PRON","PROPN",
            "PUNCT","SCONJ","SYM","VERB","X"]


# ─────────────────────────────────────────────
# LOAD GOLD LABELS
# ─────────────────────────────────────────────
def load_gold(folder):
    """Load gold POS tags from .conll files."""
    gold = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".conll"):
            continue
        with open(os.path.join(folder, fname), encoding="utf-8") as f:
            sentence = []
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if sentence:
                        gold.append(sentence)
                        sentence = []
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        sentence.append(parts[1])
            if sentence:
                gold.append(sentence)
    return gold


# ─────────────────────────────────────────────
# LOAD PREDICTED LABELS
# ─────────────────────────────────────────────
def load_pred(pipeline, method):
    """Load predicted tags from results/*.conll"""
    path = os.path.join(OUT_DIR, f"pipeline_{pipeline}_{method}.conll")
    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        return None
    preds = []
    with open(path, encoding="utf-8") as f:
        sentence = []
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if sentence:
                    preds.append(sentence)
                    sentence = []
            else:
                parts = line.split("\t")
                tag = parts[1] if len(parts) >= 2 else "X"
                sentence.append(tag)
        if sentence:
            preds.append(sentence)
    return preds


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(gold_sents, pred_sents):
    """Token-level accuracy + macro F1 per tag."""
    tp_map  = defaultdict(int)
    fp_map  = defaultdict(int)
    fn_map  = defaultdict(int)
    correct = 0
    total   = 0

    for g_sent, p_sent in zip(gold_sents, pred_sents):
        min_len = min(len(g_sent), len(p_sent))
        for g, p in zip(g_sent[:min_len], p_sent[:min_len]):
            total += 1
            if g == p:
                correct += 1
                tp_map[g] += 1
            else:
                fp_map[p] += 1
                fn_map[g] += 1

    accuracy = correct / total if total > 0 else 0.0

    all_tags = set(list(tp_map.keys()) + list(fp_map.keys()) + list(fn_map.keys()))
    f1_per_tag = {}
    for tag in sorted(all_tags):
        tp   = tp_map[tag]
        fp   = fp_map[tag]
        fn   = fn_map[tag]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_tag[tag] = round(f1, 4)

    macro_f1 = sum(f1_per_tag.values()) / len(f1_per_tag) if f1_per_tag else 0.0

    return {
        "accuracy"      : round(accuracy, 4),
        "macro_f1"      : round(macro_f1, 4),
        "f1_per_tag"    : f1_per_tag,
        "total_tokens"  : total,
        "correct_tokens": correct,
    }


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def save_f1_bar(all_metrics, path):
    """F1 + Accuracy bar chart — all methods."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    labels, f1s, accs, colors = [], [], [], []
    cmap = {"crf": "steelblue", "linear": "darkorange"}

    for pipeline in PIPELINES:
        for method in METHODS:
            key = f"{pipeline}_{method}"
            if key not in all_metrics:
                continue
            labels.append(f"{pipeline}\n{method}")
            f1s.append(all_metrics[key]["macro_f1"])
            accs.append(all_metrics[key]["accuracy"])
            colors.append(cmap[pipeline])

    if not labels:
        return

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, f1s,  w, label="Macro F1", color=colors, alpha=0.85)
    ax.bar(x + w/2, accs, w, label="Accuracy", color=colors, alpha=0.40)
    ax.axhline(STANZA_F1,  color="red",   linestyle="--", linewidth=1.2,
               label=f"Stanza F1={STANZA_F1}")
    ax.axhline(STANZA_ACC, color="green", linestyle="--", linewidth=1.2,
               label=f"Stanza Acc={STANZA_ACC}")

    for i, (f, a) in enumerate(zip(f1s, accs)):
        ax.text(x[i] - w/2, f + 0.002, f"{f:.4f}", ha="center", va="bottom", fontsize=7)
        ax.text(x[i] + w/2, a + 0.002, f"{a:.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Dual Pipeline — F1 & Accuracy Comparison  [{date_str}]")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    # Blue = CRF, Orange = Linear patch legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color="steelblue", label="CRF pipeline"),
                      Patch(color="darkorange", label="Linear pipeline")]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][2:],
              loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Graph → {path}")


def save_per_tag_f1(all_metrics, path):
    """Per-tag F1 — best CRF vs best Linear."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Best method per pipeline
    best_crf_key = best_lin_key = None
    best_crf_f1 = best_lin_f1 = -1
    for method in METHODS:
        ck = f"crf_{method}"
        lk = f"linear_{method}"
        if ck in all_metrics and all_metrics[ck]["macro_f1"] > best_crf_f1:
            best_crf_f1, best_crf_key = all_metrics[ck]["macro_f1"], ck
        if lk in all_metrics and all_metrics[lk]["macro_f1"] > best_lin_f1:
            best_lin_f1, best_lin_key = all_metrics[lk]["macro_f1"], lk

    if not best_crf_key and not best_lin_key:
        return

    tags = ALL_TAGS
    x    = np.arange(len(tags))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(16, 6))

    if best_crf_key:
        f1_crf = [all_metrics[best_crf_key]["f1_per_tag"].get(t, 0) for t in tags]
        ax.bar(x - w/2, f1_crf, w, label=f"CRF ({best_crf_key.split('_',1)[1]})",
               color="steelblue", alpha=0.85)

    if best_lin_key:
        f1_lin = [all_metrics[best_lin_key]["f1_per_tag"].get(t, 0) for t in tags]
        ax.bar(x + w/2, f1_lin, w, label=f"Linear ({best_lin_key.split('_',1)[1]})",
               color="darkorange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title(f"Per-tag F1 — Best CRF vs Best Linear  [{date_str}]")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Graph → {path}")


def save_comparison_heatmap(all_metrics, path):
    """Heatmap — pipeline × method grid (Macro F1)."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    available_methods = [m for m in METHODS
                         if any(f"{p}_{m}" in all_metrics for p in PIPELINES)]
    if not available_methods:
        return

    data = np.zeros((len(PIPELINES), len(available_methods)))
    for i, pipeline in enumerate(PIPELINES):
        for j, method in enumerate(available_methods):
            key = f"{pipeline}_{method}"
            data[i, j] = all_metrics.get(key, {}).get("macro_f1", 0)

    fig, ax = plt.subplots(figsize=(max(8, len(available_methods)*2), 4))
    im = ax.imshow(data, cmap="YlGn", vmin=0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Macro F1")

    ax.set_xticks(range(len(available_methods)))
    ax.set_xticklabels(available_methods, rotation=20, ha="right")
    ax.set_yticks(range(len(PIPELINES)))
    ax.set_yticklabels([p.upper() for p in PIPELINES])
    ax.set_title(f"Macro F1 Heatmap — Pipeline × Method  [{date_str}]")

    for i in range(len(PIPELINES)):
        for j in range(len(available_methods)):
            val = data[i, j]
            color = "white" if val > 0.85 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Graph → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Loading gold labels...")
    gold = load_gold(DATA_DIR)
    print(f"  Gold sentences: {len(gold)}")

    all_metrics = {}

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  DUAL PIPELINE EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"  Gold sentences : {len(gold)}")
    report_lines.append("")

    header = f"{'Pipeline':<10} {'Method':<16} {'Accuracy':>10} {'Macro F1':>10} {'Tokens':>10}"
    report_lines.append(header)
    report_lines.append("-" * 70)

    for pipeline in PIPELINES:
        for method in METHODS:
            preds = load_pred(pipeline, method)
            if preds is None:
                continue
            metrics = compute_metrics(gold, preds)
            key = f"{pipeline}_{method}"
            all_metrics[key] = metrics

            beat = " ← beats Stanza!" if metrics["macro_f1"] > STANZA_F1 else ""
            row = (
                f"{pipeline:<10} {method:<16} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['macro_f1']:>10.4f} "
                f"{metrics['total_tokens']:>10}{beat}"
            )
            report_lines.append(row)

    report_lines.append("-" * 70)

    # ── Stanza baseline ──────────────────────
    report_lines.append(
        f"{'stanza':<10} {'baseline':<16} "
        f"{STANZA_ACC:>10.4f} {STANZA_F1:>10.4f} {'':>10}"
    )
    report_lines.append("=" * 70)

    # ── Best method per pipeline ────────────
    report_lines.append("")
    report_lines.append("BEST METHOD PER PIPELINE (by Macro F1):")
    for pipeline in PIPELINES:
        pipeline_keys = [k for k in all_metrics if k.startswith(pipeline)]
        if not pipeline_keys:
            continue
        best_key = max(pipeline_keys, key=lambda k: all_metrics[k]["macro_f1"])
        best     = all_metrics[best_key]
        report_lines.append(
            f"  {pipeline.upper():8s} → {best_key:<25s}  "
            f"Acc={best['accuracy']:.4f}  F1={best['macro_f1']:.4f}"
        )

    # ── CRF vs Linear ───────────────────────
    report_lines.append("")
    report_lines.append("CRF vs LINEAR (same method, Macro F1):")
    for method in METHODS:
        ck = f"crf_{method}"
        lk = f"linear_{method}"
        if ck in all_metrics and lk in all_metrics:
            crf_f1    = all_metrics[ck]["macro_f1"]
            linear_f1 = all_metrics[lk]["macro_f1"]
            winner    = "CRF" if crf_f1 >= linear_f1 else "Linear"
            diff      = abs(crf_f1 - linear_f1)
            report_lines.append(
                f"  {method:<16}  CRF={crf_f1:.4f}  Linear={linear_f1:.4f}  "
                f"→ {winner} wins (+{diff:.4f})"
            )

    report_lines.append("")
    report_lines.append("=" * 70)

    # ── Print & save report ─────────────────
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = os.path.join(OUT_DIR, "dual_pipeline_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved → {report_path}")

    # ── Save JSON ───────────────────────────
    summary = {k: {m: v for m, v in vals.items() if m != "f1_per_tag"}
               for k, vals in all_metrics.items()}
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed": all_metrics}, f,
                  indent=2, ensure_ascii=False)
    print(f"Metrics saved  → {metrics_path}")

    # ── Save Graphs ─────────────────────────
    if all_metrics:
        print("\nSaving graphs...")
        save_f1_bar(all_metrics,
                    os.path.join(OUT_DIR, "pipeline_f1_bars.png"))
        save_per_tag_f1(all_metrics,
                        os.path.join(OUT_DIR, "per_tag_f1.png"))
        save_comparison_heatmap(all_metrics,
                                os.path.join(OUT_DIR, "dual_pipeline_comparison.png"))
    else:
        print("\n⚠ No results to plot — run inference pipelines first.")


if __name__ == "__main__":
    main()