# ACL Hindi POS Tagger — Phase 1

MuRIL + CRF + Linear | Hindi POS Tagging | ACL 2024 Reference

---

## Project Structure

```
acl_with_another_method/
│
├── lookback.py                   # Core ACL methods (zero_shot, lookback, lbs)
├── Train_model_crf.py            # Pipeline 1 — MuRIL + CRF training
├── train_model_linear.py         # Pipeline 2 — MuRIL + Linear training
├── evaluate_dual_pipeline.py     # Evaluate both pipelines on test.conll
├── inference_pipeline_crf.py     # CRF batch inference on author text
├── inference_pipeline_linear.py  # Linear batch inference on author text
├── analyze.py                    # Analysis graphs + report
├── main.py                       # Master pipeline controller
│
├── dataset/
│   ├── train.conll               # Hindi UD training data (~13K sentences)
│   ├── dev.conll                 # Validation data (~1.6K sentences)
│   └── test.conll                # Test data (~1.7K sentences)
│
├── writers who died before 1965/ # Raw Hindi literary text
│   └── <author>/<category>/*.txt
│
├── checkpoints_crf/              # CRF model checkpoints (auto-created)
│   └── best_crf.pt
│
├── checkpoints_linear/           # Linear model checkpoints (auto-created)
│   └── best_linear.pt
│
├── pos_model_crf/                # Saved tokenizer CRF (auto-created)
├── pos_model_linear/             # Saved tokenizer Linear (auto-created)
│
└── results/                      # All outputs (auto-created)
    ├── analysis/                 # Detailed analysis graphs
    ├── metrics.json
    ├── dual_pipeline_report.txt
    ├── pipeline_crf_*.conll
    └── pipeline_linear_*.conll
```

---

## Setup

```bash
conda activate pos_tagger
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers torchcrf scikit-learn matplotlib tqdm seqeval
```

---

## Dataset

Place Hindi UD dataset files in `dataset/` folder:
- `dataset/train.conll`
- `dataset/dev.conll`
- `dataset/test.conll`

Download from: https://github.com/UniversalDependencies/UD_Hindi-HDTB

Format: `word\tTAG` per line, blank lines between sentences.

---

## Run

### Full Pipeline (recommended)
```bash
python main.py
```

### Individual Steps
```bash
python Train_model_crf.py           # Train CRF model
python Train_model_crf.py --resume  # Resume CRF training
python train_model_linear.py        # Train Linear model
python evaluate_dual_pipeline.py    # Evaluate both pipelines
python inference_pipeline_crf.py    # Tag author text with CRF
python inference_pipeline_linear.py # Tag author text with Linear
python analyze.py                   # Generate analysis graphs
```

### Skip Flags
```bash
python main.py --skip_crf_train --skip_linear_train  # Eval only
python main.py --skip_eval --skip_analyze             # Train only
python main.py --resume                               # Resume training
```

---

## Architecture (Phase 1)

```
MuRIL (google/muril-base-cased)
         |
   Last Hidden State (768 dim)
         |
    Linear (768 → 17)
         |
    ┌────┴────┐
    │         │
  CRF      Linear
    │         │
    └────┬────┘
         |
  Zero-shot / Lookback / LBS
```

---

## ACL Methods

| Method | Description | Paper Hindi F1 |
|--------|-------------|----------------|
| Zero-shot | First subword argmax | 0.93 |
| Lookback | Copy first subword tag | 0.94 |
| LBS | Most confident subword tag | 0.95 |

---

## Baselines

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Stanza | 0.9791 | 0.9282 |
| Paper (MuRIL) | — | 0.95 |

---

## Reference

**Paper:** Part-of-Speech Tagging for Extremely Low-resource Indian Languages  
**Authors:** Sanjeev Kumar, Preethi Jyothi, Pushpak Bhattacharyya (IIT Bombay)  
**Venue:** ACL 2024  
**Code:** https://github.com/snjev310/acl-24-pos
