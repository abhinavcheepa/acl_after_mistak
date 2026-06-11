"""
inference_pipeline_linear.py
MuRIL + Linear (Softmax) pipeline — teeno methods: zero_shot, lookback, lookback_with_score
"""

import os, json, time, torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from lookback import zero_shot, lookback, lookback_with_score, lookback_with_score_weighted

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "writers who died before 1965")
CKPT       = os.path.join(BASE_DIR, "checkpoints_linear", "best_linear.pt")
OUT_DIR    = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME  = "google/muril-base-cased"
BATCH_SIZE  = 32          # FIX: was 1024 → OOM crash hota tha
MAX_LEN     = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class MuRIL_Linear(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.muril         = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        hidden             = self.muril.config.hidden_size   # 768
        self.layer_weights = nn.Parameter(torch.ones(4))
        self.mlp           = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.muril(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        w      = F.softmax(self.layer_weights, dim=0)
        hidden = sum(w[i] * out.hidden_states[-(i + 1)] for i in range(4))
        logits = self.mlp(hidden)             # (B, T, num_labels)
        return logits


# ─────────────────────────────────────────────
# LOAD DATA  (.conll  word\tTAG)
# ─────────────────────────────────────────────
def load_conll(folder):
    sentences, gold_labels = [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".conll"):
            continue
        words, tags = [], []
        with open(os.path.join(folder, fname), encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if words:
                        sentences.append(words)
                        gold_labels.append(tags)
                        words, tags = [], []
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        words.append(parts[0])
                        tags.append(parts[1])
        if words:
            sentences.append(words)
            gold_labels.append(tags)
    return sentences, gold_labels


# ─────────────────────────────────────────────
# TOKENIZE BATCH
# ─────────────────────────────────────────────
def tokenize_batch(tokenizer, batch_words):
    encodings = tokenizer(
        batch_words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    return encodings


# ─────────────────────────────────────────────
# INFERENCE — ONE BATCH
# ─────────────────────────────────────────────
def run_batch(model, tokenizer, batch_words, id2label):
    enc    = tokenize_batch(tokenizer, batch_words)
    ids    = enc["input_ids"].to(DEVICE)
    mask   = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(ids, mask)             # (B, T, C)

    results = {"zero_shot": [], "lookback": [], "lbs": [], "lbs_weighted": []}

    for b, words in enumerate(batch_words):
        wids   = enc.word_ids(batch_index=b)
        amask  = mask[b].cpu().tolist()
        logit_b = logits[b].cpu()             # (T, C)

        # ── Method 1: zero_shot (argmax) ─────
        argmax_ids = logit_b.argmax(dim=-1).tolist()
        zs_tags = []
        seen, pi = set(), 0
        for pos, wid in enumerate(wids):
            if amask[pos] == 1:
                if wid is not None and wid not in seen:
                    tag_id = argmax_ids[pi] if pi < len(argmax_ids) else 0
                    zs_tags.append(id2label.get(tag_id, "X"))
                    seen.add(wid)
                pi += 1
        results["zero_shot"].append(zs_tags)

        # ── Method 2: lookback ───────────────
        lb_tags = lookback(logit_b, wids, id2label)
        results["lookback"].append(lb_tags)

        # ── Method 3: lbs ────────────────────
        lbs_tags = lookback_with_score(logit_b, wids, id2label)
        results["lbs"].append(lbs_tags)

        # ── Method 3b: lbs_weighted ──────────
        lbs_w_tags = lookback_with_score_weighted(logit_b, wids, id2label, tokenizer, words)
        results["lbs_weighted"].append(lbs_w_tags)

    return results


# ─────────────────────────────────────────────
# WRITE CONLL
# ─────────────────────────────────────────────
def write_conll(path, sentences, predictions):
    with open(path, "w", encoding="utf-8") as f:
        for words, tags in zip(sentences, predictions):
            for w, t in zip(words, tags):
                f.write(f"{w}\t{t}\n")
            f.write("\n")
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    t0 = time.time()

    # ── Load checkpoint ──────────────────────
    ck        = torch.load(CKPT, map_location=DEVICE)
    label2id  = ck["label2id"]
    id2label  = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    print(f"Labels ({num_labels}): {list(label2id.keys())}")

    # ── Build model ──────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = MuRIL_Linear(num_labels).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print("Model loaded.")

    # ── Load data ────────────────────────────
    sentences, gold_labels = load_conll(DATA_DIR)
    print(f"Sentences loaded: {len(sentences)}")

    # ── Run inference ────────────────────────
    all_preds = {"zero_shot": [], "lookback": [], "lbs": [], "lbs_weighted": []}

    for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc="Inference (Linear)"):
        batch     = sentences[i: i + BATCH_SIZE]
        batch_res = run_batch(model, tokenizer, batch, id2label)
        for method, preds in batch_res.items():
            all_preds[method].extend(preds)

    # ── Save outputs ─────────────────────────
    for method, preds in all_preds.items():
        out_path = os.path.join(OUT_DIR, f"pipeline_linear_{method}.conll")
        write_conll(out_path, sentences, preds)

    # ── Save stats ───────────────────────────
    stats = {
        "total_sentences": len(sentences),
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
        "elapsed_sec": round(time.time() - t0, 2),
        "methods": list(all_preds.keys()),
    }
    stats_path = os.path.join(OUT_DIR, "pipeline_linear_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats → {stats_path}")
    print(f"\nDone in {stats['elapsed_sec']}s")


if __name__ == "__main__":
    main()