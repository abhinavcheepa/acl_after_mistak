"""
main.py
=======
Master pipeline controller — Phase 1

Poora pipeline ek command se chalata hai:
    python main.py

Steps:
    1. Dependencies check karta hai
    2. Dataset verify karta hai
    3. CRF model train karta hai
    4. Linear model train karta hai
    5. Dual pipeline evaluate karta hai
    6. Analysis + graphs generate karta hai

Flags:
    --skip_crf_train     CRF training skip karo
    --skip_linear_train  Linear training skip karo
    --skip_eval          Evaluation skip karo
    --skip_analyze       Analysis skip karo
    --resume             Resume training from checkpoint
"""

import os, sys, time, argparse, subprocess, json

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def run_script(script, extra_args=""):
    """Ek Python script run karta hai aur output print karta hai."""
    cmd = f"{sys.executable} {os.path.join(BASE_DIR, script)} {extra_args}"
    start = time.time()
    print(f"\n▶ Running: {script}")
    print("-" * 40)
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n❌ {script} failed (returncode={result.returncode})")
        return False
    print(f"\n✅ {script} done in {elapsed/60:.1f}m")
    return True


# ─────────────────────────────────────────────────────────────
# CHECKS
# ─────────────────────────────────────────────────────────────

def check_dependencies():
    """Required packages check karta hai."""
    banner("Checking Dependencies")
    required = {
        "torch":        "PyTorch",
        "transformers": "HuggingFace Transformers",
        "torchcrf":     "pytorch-crf",
        "sklearn":      "scikit-learn",
        "matplotlib":   "Matplotlib",
        "tqdm":         "tqdm",
        "numpy":        "NumPy",
    }
    all_ok = True
    for pkg, name in required.items():
        try:
            __import__(pkg)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} — pip install {pkg}")
            all_ok = False

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠  GPU: Not available — CPU pe chalega (slow)")
    except Exception:
        pass

    return all_ok


def check_dataset():
    """Dataset files verify karta hai."""
    banner("Checking Dataset")
    required_files = [
        os.path.join(BASE_DIR, "dataset", "train.conll"),
        os.path.join(BASE_DIR, "dataset", "dev.conll"),
        os.path.join(BASE_DIR, "dataset", "test.conll"),
    ]
    all_ok = True
    for path in required_files:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            # Count sentences
            with open(path, encoding="utf-8") as f:
                sents = sum(1 for line in f if line.strip() == "")
            print(f"  ✅ {os.path.basename(path)} ({size:.0f} KB, ~{sents} sentences)")
        else:
            print(f"  ❌ Missing: {path}")
            all_ok = False

    if not all_ok:
        print("""
  Download Hindi UD dataset:
    https://github.com/UniversalDependencies/UD_Hindi-HDTB

  Place files in dataset/ folder:
    dataset/train.conll
    dataset/dev.conll
    dataset/test.conll
        """)
    return all_ok


def check_lookback():
    """lookback.py check karta hai."""
    path = os.path.join(BASE_DIR, "lookback.py")
    if os.path.exists(path):
        print(f"  ✅ lookback.py")
        return True
    print(f"  ❌ lookback.py not found!")
    return False


def check_checkpoints():
    """Existing checkpoints check karta hai."""
    crf_path = os.path.join(BASE_DIR, "checkpoints_crf",    "best_crf.pt")
    lin_path = os.path.join(BASE_DIR, "checkpoints_linear", "best_linear.pt")

    crf_exists = os.path.exists(crf_path)
    lin_exists = os.path.exists(lin_path)

    if crf_exists:
        size = os.path.getsize(crf_path) / 1e6
        print(f"  ✅ CRF checkpoint ({size:.0f} MB)")
    else:
        print(f"  ⚠  CRF checkpoint not found — will train")

    if lin_exists:
        size = os.path.getsize(lin_path) / 1e6
        print(f"  ✅ Linear checkpoint ({size:.0f} MB)")
    else:
        print(f"  ⚠  Linear checkpoint not found — will train")

    return crf_exists, lin_exists


# ─────────────────────────────────────────────────────────────
# PIPELINE SUMMARY
# ─────────────────────────────────────────────────────────────

def print_pipeline_summary():
    banner("Pipeline 1 — Phase 1 Architecture")
    print("""
  MuRIL (google/muril-base-cased)
           ↓
    Last Hidden State (768 dim)
           ↓
      Linear (768 → 17)
           ↓
      ┌─────────────┐
      │             │
    CRF          Linear
  (Pipeline 1) (Pipeline 2)
      │             │
      ↓             ↓
  Zero-shot     Zero-shot
  Lookback      Lookback
    LBS           LBS

  Training:
    - EMA (decay=0.9995)
    - Gradient Accumulation (batch=16×4=64)
    - Cosine LR + Warmup (10%)
    - Class-weighted Loss
    - Label Smoothing (0.1)
    - Mixed Precision (FP16)
    - Early Stopping (patience=5)
    """)


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

def print_final_summary():
    """Results summary print karta hai."""
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    STANZA_F1  = 0.9282
    STANZA_ACC = 0.9791

    banner("Final Results Summary")
    print(f"  {'Method':<28} {'F1':>8} {'Acc':>8} {'vs Paper':>10}")
    print("  " + "-" * 58)

    paper = {"zero_shot": 0.93, "lookback": 0.94, "lbs": 0.95}

    for pipeline in ["crf", "linear"]:
        for method in ["zero_shot", "lookback", "lbs"]:
            if pipeline in metrics and method in metrics[pipeline]:
                vals = metrics[pipeline][method]
                diff = vals["macro_f1"] - paper.get(method, 0.93)
                sign = "+" if diff >= 0 else ""
                beat = " ✅" if vals["macro_f1"] > STANZA_F1 else ""
                print(f"  {pipeline}_{method:<22} "
                      f"{vals['macro_f1']:>8.4f} "
                      f"{vals['accuracy']:>8.4f} "
                      f"{sign}{diff:>9.4f}{beat}")

    print(f"  {'Stanza (baseline)':<28} "
          f"{STANZA_F1:>8.4f} {STANZA_ACC:>8.4f} {'—':>10}")
    print("  " + "-" * 58)
    print(f"\n  Results saved in: {RESULTS_DIR}/")
    print(f"  Analysis in    : {RESULTS_DIR}/analysis/")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ACL Hindi POS Tagger — Phase 1 Pipeline"
    )
    parser.add_argument("--skip_crf_train",    action="store_true")
    parser.add_argument("--skip_linear_train", action="store_true")
    parser.add_argument("--skip_eval",         action="store_true")
    parser.add_argument("--skip_analyze",      action="store_true")
    parser.add_argument("--resume",            action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    total_start = time.time()

    # ── Print architecture ───────────────────────────────────
    print_pipeline_summary()

    # ── Dependency check ─────────────────────────────────────
    if not check_dependencies():
        print("\n❌ Missing dependencies. Install them and retry.")
        sys.exit(1)

    # ── Dataset check ────────────────────────────────────────
    banner("Checking Files")
    if not check_dataset():
        print("\n❌ Dataset missing. See instructions above.")
        sys.exit(1)
    check_lookback()
    crf_exists, lin_exists = check_checkpoints()

    resume_flag = "--resume" if args.resume else ""

    # ── Step 1: Train CRF ────────────────────────────────────
    if not args.skip_crf_train and not crf_exists:
        banner("Step 1/4 — Training CRF Model")
        ok = run_script("Train_model_crf.py", resume_flag)
        if not ok:
            print("CRF training failed. Check errors above.")
            sys.exit(1)
    elif args.skip_crf_train or crf_exists:
        banner("Step 1/4 — CRF Training [SKIPPED]")
        print("  Checkpoint already exists or --skip_crf_train set.")

    # ── Step 2: Train Linear ─────────────────────────────────
    if not args.skip_linear_train and not lin_exists:
        banner("Step 2/4 — Training Linear Model")
        ok = run_script("train_model_linear.py", resume_flag)
        if not ok:
            print("Linear training failed. Check errors above.")
            sys.exit(1)
    elif args.skip_linear_train or lin_exists:
        banner("Step 2/4 — Linear Training [SKIPPED]")
        print("  Checkpoint already exists or --skip_linear_train set.")

    # ── Step 3: Evaluate ─────────────────────────────────────
    if not args.skip_eval:
        banner("Step 3/4 — Dual Pipeline Evaluation")
        ok = run_script("evaluate_dual_pipeline.py")
        if not ok:
            print("Evaluation failed. Check errors above.")

    # ── Step 4: Analyze ──────────────────────────────────────
    if not args.skip_analyze:
        banner("Step 4/4 — Analysis + Graphs")
        ok = run_script("analyze.py")
        if not ok:
            print("Analysis failed — results may still be saved.")

    # ── Final summary ────────────────────────────────────────
    print_final_summary()

    total_time = time.time() - total_start
    banner(f"Pipeline Complete! Total time: {total_time/60:.1f}m")
    print(f"""
  Output files:
    checkpoints_crf/best_crf.pt
    checkpoints_linear/best_linear.pt
    results/dual_pipeline_report.txt
    results/dual_pipeline_comparison.png
    results/pipeline_f1_bars.png
    results/per_tag_f1.png
    results/confusion_matrix.png
    results/metrics.json
    results/analysis/01_training_curves.png
    results/analysis/02_method_comparison.png
    results/analysis/03_crf_vs_linear.png
    results/analysis/04_final_summary.png
    results/analysis/analysis_report.txt

  Next run options:
    python main.py --skip_crf_train --skip_linear_train  (eval only)
    python main.py --resume                              (resume training)
    python main.py --skip_eval --skip_analyze            (train only)
    """)


if __name__ == "__main__":
    main()
