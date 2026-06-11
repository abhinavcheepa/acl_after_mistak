"""
Train_model_crf.py
==================
Pipeline 1 — MuRIL + 4-layer fusion + MLP(768→768→17) + CRF

Architecture:
    MuRIL (google/muril-base-cased)
           ↓
    4-layer hidden states weighted fusion
           ↓
    MLP(768 → 768 → 17) + GELU + Dropout
           ↓
    CRF Decoder (torchcrf — batch_first, viterbi_decode)

FIXES applied vs original:
    1. Consistent CRF mask  — training loss and viterbi decode now both use
       `(labels != -100)` mask, eliminating the divergence that froze F1.
    2. Unified forward()    — returns (loss, preds) in one pass; evaluate()
       no longer calls the model twice per batch.
    3. scheduler.step() order — scheduler steps AFTER optimizer.step() to
       silence the PyTorch 1.1+ warning and correctly apply warmup LR.
"""

import os
import sys
import math
import argparse
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "dataset")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

TRAIN_FILE = os.path.join(DATA_DIR, "train.conll")
VAL_FILE   = os.path.join(DATA_DIR, "dev.conll")
TEST_FILE  = os.path.join(DATA_DIR, "test.conll")

MODEL_NAME    = "google/muril-base-cased"
MAX_LEN       = 128
BATCH_SIZE    = 16
ACCUM_STEPS   = 8        # effective batch = 128
EPOCHS        = 30
LR            = 2e-5
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.05
FREEZE_LAYERS = 6
DROPOUT       = 0.3
EMA_DECAY     = 0.999
PATIENCE      = 7
LABEL_SMOOTH  = 0.1

TAG2IDX = {
    'NN':0,'NST':1,'NNP':2,'PRP':3,'DEM':4,
    'VM':5,'VAUX':6,'JJ':7,'RB':8,'PSP':9,
    'RP':10,'CC':11,'WQ':12,'QF':13,'QC':14,
    'QO':15,'SYM':16
}
IDX2TAG = {v: k for k, v in TAG2IDX.items()}
NUM_TAGS = len(TAG2IDX)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
def read_conll(path):
    sentences = []
    words, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                words.append(parts[0])
                tags.append(parts[-1])
            else:
                if words:
                    sentences.append((words, tags))
                    words, tags = [], []
    if words:
        sentences.append((words, tags))
    return sentences


class POSDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len):
        self.data      = sentences
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        word_ids = encoding.word_ids(batch_index=0)
        labels   = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word_id:
                tag = tags[wid] if wid < len(tags) else "NN"
                labels.append(TAG2IDX.get(tag, 0))
            else:
                labels.append(-100)
            prev_word_id = wid

        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels":         labels,
        }


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
class MuRIL_MLP_CRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True,
            low_cpu_mem_usage=True,
        )
        # Freeze bottom layers
        modules_to_freeze = [self.bert.embeddings] + \
                            list(self.bert.encoder.layer[:FREEZE_LAYERS])
        for m in modules_to_freeze:
            for p in m.parameters():
                p.requires_grad = False

        hidden = self.bert.config.hidden_size  # 768

        # Learnable layer-fusion weights (last 4 hidden states)
        self.layer_weights = nn.Parameter(torch.ones(4) / 4)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, NUM_TAGS),
        )

        # CRF — batch_first=True
        self.crf = CRF(NUM_TAGS)

    def _encode(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Fuse last 4 hidden states with learned weights
        hidden_states = out.hidden_states          # tuple: (num_layers+1) x (B,T,H)
        last4   = torch.stack(hidden_states[-4:], dim=0)   # (4, B, T, H)
        weights = torch.softmax(self.layer_weights, dim=0)
        fused   = (weights[:, None, None, None] * last4).sum(0)  # (B, T, H)
        emissions = self.mlp(fused)                # (B, T, NUM_TAGS)
        return emissions

    # ------------------------------------------------------------------
    # FIX 1 & 2: unified forward — returns (loss, preds) always.
    #   • When labels are given: computes CRF loss AND viterbi preds
    #     using the SAME `labels != -100` mask for both.
    #   • When labels are None (pure inference): loss=None, mask from
    #     attention_mask.
    # ------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        emissions = self._encode(input_ids, attention_mask, token_type_ids)

        if labels is not None:
            # ── consistent mask: positions where a real label exists ──
            mask = (labels != -100).bool()          # (B, T)

            # CRF requires labels in [0, NUM_TAGS); replace -100 with 0
            crf_labels = labels.clone()
            crf_labels[~mask] = 0

            # log_likelihood shape: (B,)  →  scalar loss
            log_likelihood = self.crf(emissions, crf_labels, mask=mask)
            loss = -log_likelihood.mean()

            # Decode with the SAME mask — keeps train/eval behaviour identical
            preds = self.crf.viterbi_decode(emissions, mask=mask)
            return loss, preds

        else:
            # Inference only (no labels available)
            mask  = (attention_mask == 1).bool()    # (B, T)
            preds = self.crf.viterbi_decode(emissions, mask=mask)
            return None, preds


# ──────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay):
        self.decay  = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self, model):
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self._backup)


# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────
def compute_f1(preds, labels):
    return f1_score(labels, preds, average="weighted", zero_division=0)

def compute_accuracy(preds, labels):
    arr = np.array(preds)
    lbl = np.array(labels)
    return (arr == lbl).mean() if len(arr) else 0.0


# ──────────────────────────────────────────────
# EVALUATE
# FIX 2: single forward pass per batch (loss + preds together)
# ──────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            # Single forward call — returns both loss and viterbi preds
            loss, preds = model(input_ids, attention_mask, token_type_ids, labels)
            total_loss += loss.item()

            # Collect predictions aligned with real (non -100) label positions
            for i, pred_seq in enumerate(preds):
                label_seq      = labels[i].tolist()
                valid_positions = [t for t, l in enumerate(label_seq) if l != -100]
                for j, t in enumerate(valid_positions):
                    if j >= len(pred_seq):
                        break
                    all_preds.append(pred_seq[j])
                    all_labels.append(label_seq[t])

    avg_loss = total_loss / len(loader)
    f1       = compute_f1(all_preds, all_labels)
    acc      = compute_accuracy(all_preds, all_labels)
    return avg_loss, f1, acc


# ──────────────────────────────────────────────
# TRAINING CURVES PLOT
# ──────────────────────────────────────────────
def save_training_curves(history, save_dir):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"CRF Training Curves — {date_str}", fontsize=12)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_f1"],  label="Val F1",  color="green")
    axes[1].plot(epochs, history["test_f1"], label="Test F1", color="orange")
    axes[1].set_title("F1 Score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["lr"], label="LR", color="red")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"crf_training_curves_{date_str}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊 Training curves saved: {out_path}")


# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────
def train(resume=False):
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Data ──────────────────────────────────
    print("Loading data...")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data = read_conll(TRAIN_FILE)
    val_data   = read_conll(VAL_FILE)
    test_data  = read_conll(TEST_FILE)
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    train_loader = DataLoader(POSDataset(train_data, tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(POSDataset(val_data,   tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(POSDataset(test_data,  tokenizer, MAX_LEN),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────
    print("Building model...")
    model = MuRIL_MLP_CRF().to(device)

    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    trainable = total - frozen
    print(f"  Frozen params   : {frozen:,}")
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")

    ema = EMA(model, EMA_DECAY)

    # ── Optimizer ─────────────────────────────
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LR)

    total_steps  = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / \
                   float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler('cuda') if device.type == "cuda" else GradScaler()

    best_val_f1 = 0.0
    best_ckpt   = os.path.join(CHECKPOINT_DIR, "best_crf.pt")
    no_improve  = 0
    start_epoch = 1

    # ── Resume ────────────────────────────────
    if resume and os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        best_val_f1 = ckpt.get("val_f1", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch-1}, best F1={best_val_f1:.4f}")

    print(f"Training: {EPOCHS} epochs, effective batch = {BATCH_SIZE * ACCUM_STEPS}")
    print("-" * 60)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "test_f1": [], "lr": []}

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}")
        for step, batch in enumerate(pbar, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            with autocast('cuda' if device.type == "cuda" else 'cpu'):
                # FIX 2: forward now returns (loss, preds); ignore preds during training
                loss, _ = model(input_ids, attention_mask, token_type_ids, labels)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()
            total_loss += loss.item() * ACCUM_STEPS

            if step % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # FIX 3: optimizer.step() BEFORE scheduler.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                scheduler.step()   # ← now correctly AFTER optimizer step

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}", lr=f"{current_lr:.2e}")

        avg_train_loss = total_loss / len(train_loader)

        # ── Evaluate with EMA weights ──────────
        ema.apply(model)
        val_loss,  val_f1,  val_acc  = evaluate(model, val_loader,  device)
        test_loss, test_f1, test_acc = evaluate(model, test_loader, device)
        ema.restore(model)

        elapsed    = (time.time() - t0) / 60
        eta        = elapsed * (EPOCHS - epoch)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"TrainLoss={avg_train_loss:.4f} | "
              f"ValLoss={val_loss:.4f} ValF1={val_f1:.4f} ValAcc={val_acc:.4f} | "
              f"TestLoss={test_loss:.4f} TestF1={test_f1:.4f} TestAcc={test_acc:.4f} | "
              f"LR={current_lr:.2e} | Time={elapsed:.1f}m ETA={eta:.1f}m")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["test_f1"].append(test_f1)
        history["lr"].append(current_lr)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve  = 0
            torch.save({
                "model":  model.state_dict(),
                "ema":    ema.shadow,
                "val_f1": val_f1,
                "epoch":  epoch,
            }, best_ckpt)
            print(f"  ✅ New best saved: F1={val_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  ⏹ Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Training complete.")
    print(f"   Best Val F1 : {best_val_f1:.4f}")
    print(f"   Best checkpoint: {best_ckpt}")

    save_training_curves(history, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(resume=args.resume)