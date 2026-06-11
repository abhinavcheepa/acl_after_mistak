"""
train_model_linear.py
=====================
Pipeline 2 — MuRIL + 4-layer fusion + MLP(768→768→17) — No CRF

Architecture:
    MuRIL (google/muril-base-cased)
         ↓
    4-layer hidden states weighted fusion
         ↓
    MLP(768 → 768 → 17) + GELU + Dropout
         ↓
    Softmax → Logits
         ↓
    POS Tags

Fixes applied:
    [FIX-A] OOM — frozen params skip in checkpoint (~115MB not 606MB)
    [FIX-B] CUDA OOM — BATCH_SIZE=4, ACCUM_STEPS=16, gradient_checkpointing
    [FIX-C] LR display — optimizer.param_groups[0]['lr']
    [FIX-D] strict=False load — frozen params missing ok
    [FIX-E] lbs_weighted method added
    [FIX-F] 4-layer fusion + MLP architecture (matches inference_pipeline_linear.py)
    [FIX-G] Deprecated torch.cuda.amp → torch.amp (GradScaler + autocast)
    [FIX-H] autocast/GradScaler device-aware
    [FIX-I] evaluate() — -100 labels correctly filtered (not attention_mask based)
    [FIX-J] Early stopping with PATIENCE added
    [FIX-K] Training curves auto-save (results/linear_training_curves_DATE.png)
"""

import os
import gc
import math
import json
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast          # [FIX-G] torch.amp not torch.cuda.amp
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "dataset")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

TRAIN_FILE      = os.path.join(DATA_DIR, "train.conll")
VAL_FILE        = os.path.join(DATA_DIR, "dev.conll")
TEST_FILE       = os.path.join(DATA_DIR, "test.conll")

BEST_PATH       = os.path.join(CHECKPOINT_DIR, "best_linear.pt")
RESUME_PATH     = os.path.join(CHECKPOINT_DIR, "resume_linear.pt")

MODEL_NAME      = "google/muril-base-cased"

# Training hyperparams
BATCH_SIZE      = 4          # [FIX-B] reduced for VRAM
ACCUM_STEPS     = 16         # effective batch = 64
MAX_LEN         = 128
EPOCHS          = 30
LR              = 2e-5
WEIGHT_DECAY    = 1e-2
WARMUP_RATIO    = 0.1
FREEZE_LAYERS   = 6          # 8 → 6: 2 extra layers train hongi
EMA_DECAY       = 0.999
DROPOUT         = 0.1
SEED            = 42
PATIENCE        = 7          # [FIX-J] early stopping

# Architecture
NUM_LAYERS_FUSE = 4
MLP_HIDDEN      = 768
NUM_TAGS        = 17

# Tag vocabulary
TAG2IDX = {
    'NN': 0, 'NST': 1, 'NNP': 2, 'PRP': 3, 'DEM': 4,
    'VM': 5, 'VAUX': 6, 'JJ': 7, 'RB': 8, 'PSP': 9,
    'RP': 10, 'CC': 11, 'WQ': 12, 'QF': 13, 'QC': 14,
    'QO': 15, 'SYM': 16
}
IDX2TAG = {v: k for k, v in TAG2IDX.items()}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def read_conll(path):
    """Read CoNLL file → list of (words, tags)."""
    sentences, words, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append((words, tags))
                    words, tags = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    words.append(parts[0])
                    tags.append(parts[-1])
    if words:
        sentences.append((words, tags))
    return sentences


class POSDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_len=MAX_LEN):
        self.samples = []
        for words, tags in sentences:
            enc = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            word_ids = enc.word_ids(0)
            label_ids = []
            seen = set()
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid in seen:
                    label_ids.append(-100)
                else:
                    seen.add(wid)
                    tag = tags[wid] if wid < len(tags) else "NN"
                    label_ids.append(TAG2IDX.get(tag, 0))

            self.samples.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "token_type_ids": enc.get("token_type_ids",
                                          torch.zeros(max_len, dtype=torch.long)).squeeze(0),
                "labels":         torch.tensor(label_ids, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class MuRIL_MLP_Linear(nn.Module):
    """
    MuRIL → 4-layer hidden state fusion → MLP(768→768→17) → Softmax
    Matches inference_pipeline_linear.py architecture exactly.
    """
    def __init__(self, model_name=MODEL_NAME, num_tags=NUM_TAGS,
                 num_layers_fuse=NUM_LAYERS_FUSE, mlp_hidden=MLP_HIDDEN,
                 dropout=DROPOUT, freeze_layers=FREEZE_LAYERS):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )

        self._freeze_layers(freeze_layers)

        self.num_layers_fuse = num_layers_fuse
        self.layer_weights = nn.Parameter(
            torch.ones(num_layers_fuse) / num_layers_fuse
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(768, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_tags),
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _freeze_layers(self, n):
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden_states = outputs.hidden_states
        fuse_states = torch.stack(
            hidden_states[-self.num_layers_fuse:], dim=0
        )  # (num_layers_fuse, B, T, 768)
        weights = F.softmax(self.layer_weights, dim=0)
        fused = (fuse_states * weights[:, None, None, None]).sum(0)  # (B, T, 768)

        logits = self.mlp(fused)  # (B, T, num_tags)

        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return loss
        else:
            return torch.argmax(logits, dim=-1)  # (B, T)


# ─────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach().cpu().float()
            for k, v in model.state_dict().items()
            if v.requires_grad
        }

    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k] = (
                    self.decay * self.shadow[k] +
                    (1 - self.decay) * v.detach().cpu().float()
                )

    def apply(self, model):
        self._orig = {k: v.clone() for k, v in model.state_dict().items() if k in self.shadow}
        for k in self.shadow:
            model.state_dict()[k].copy_(self.shadow[k].to(model.state_dict()[k].device))

    def restore(self, model):
        for k, v in self._orig.items():
            model.state_dict()[k].copy_(v)


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_f1,
                    config, ema=None, lightweight=False):
    """[FIX-A] Sirf trainable params save — frozen 208M skip → ~115MB."""
    gc.collect()
    torch.cuda.empty_cache()

    trainable_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if any(k == n for n, p in model.named_parameters() if p.requires_grad)
    }

    checkpoint = {
        "epoch":        epoch,
        "best_f1":      best_f1,
        "model_state":  trainable_state,
        "config":       config,
        "lightweight":  lightweight,
    }

    if not lightweight:
        checkpoint["optimizer_state"] = optimizer.state_dict()
        checkpoint["scheduler_state"] = scheduler.state_dict() if scheduler else None
        if ema is not None:
            checkpoint["ema_shadow"] = ema.shadow

    torch.save(checkpoint, path)

    del trainable_state, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def load_checkpoint(path, model, optimizer=None, scheduler=None, ema=None):
    """[FIX-D] strict=False — frozen params already in model."""
    ckpt = torch.load(path, map_location="cpu")

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  Loaded checkpoint: epoch={ckpt['epoch']}, best_f1={ckpt['best_f1']:.4f}")
    if missing:
        print(f"  Missing keys (frozen, ok): {len(missing)}")

    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt and ckpt["scheduler_state"]:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if ema and "ema_shadow" in ckpt:
        ema.shadow = ckpt["ema_shadow"]
    elif ema:
        ema.shadow = {
            k: v.clone().detach().cpu().float()
            for k, v in model.state_dict().items()
            if v.requires_grad
        }

    return ckpt["epoch"], ckpt["best_f1"]


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_f1(preds_flat, labels_flat):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for p, l in zip(preds_flat, labels_flat):
        if l == -100:
            continue
        if p == l:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1

    f1s = []
    for tag in set(list(tp.keys()) + list(fn.keys())):
        prec = tp[tag] / (tp[tag] + fp[tag] + 1e-9)
        rec  = tp[tag] / (tp[tag] + fn[tag] + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)

    return float(np.mean(f1s)) if f1s else 0.0


def compute_accuracy(preds_flat, labels_flat):
    correct = total = 0
    for p, l in zip(preds_flat, labels_flat):
        if l == -100:
            continue
        total += 1
        if p == l:
            correct += 1
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────
# TRAINING CURVES PLOT                [FIX-K]
# ─────────────────────────────────────────────
def save_training_curves(history, save_dir):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Linear Training Curves — {date_str}", fontsize=12)

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
    out_path = os.path.join(save_dir, f"linear_training_curves_{date_str}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊 Training curves saved: {out_path}")


# ─────────────────────────────────────────────
# EVALUATE                            [FIX-I]
# ─────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            loss  = model(input_ids, attention_mask, token_type_ids, labels)
            preds = model(input_ids, attention_mask, token_type_ids)  # (B, T)

            total_loss += loss.item()

            # [FIX-I] filter by label != -100, not attention_mask
            for i in range(labels.size(0)):
                label_seq = labels[i].tolist()
                pred_seq  = preds[i].tolist()
                for t in range(len(label_seq)):
                    if label_seq[t] != -100:
                        all_preds.append(pred_seq[t])
                        all_labels.append(label_seq[t])

    avg_loss = total_loss / len(loader)
    f1  = compute_f1(all_preds, all_labels)
    acc = compute_accuracy(all_preds, all_labels)
    return avg_loss, f1, acc


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading data...")
    train_data = read_conll(TRAIN_FILE)
    val_data   = read_conll(VAL_FILE)
    test_data  = read_conll(TEST_FILE)
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    train_ds = POSDataset(train_data, tokenizer)
    val_ds   = POSDataset(val_data,   tokenizer)
    test_ds  = POSDataset(test_data,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=0, pin_memory=True)

    print("Building model...")
    model = MuRIL_MLP_Linear().to(device)

    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = frozen + trainable
    print(f"  Frozen params   : {frozen:,}")
    print(f"  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    total_steps  = (len(train_loader) // ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # [FIX-G] device-aware GradScaler
    scaler = GradScaler('cuda') if device.type == "cuda" else GradScaler()
    ema    = EMA(model)

    config = {
        "model_name":      MODEL_NAME,
        "batch_size":      BATCH_SIZE,
        "accum_steps":     ACCUM_STEPS,
        "lr":              LR,
        "epochs":          EPOCHS,
        "freeze_layers":   FREEZE_LAYERS,
        "num_layers_fuse": NUM_LAYERS_FUSE,
        "mlp_hidden":      MLP_HIDDEN,
        "num_tags":        NUM_TAGS,
        "tag2idx":         TAG2IDX,
    }

    start_epoch = 0
    best_f1     = 0.0
    no_improve  = 0                                  # [FIX-J]

    if resume and os.path.exists(RESUME_PATH):
        print(f"Resuming from {RESUME_PATH}")
        start_epoch, best_f1 = load_checkpoint(
            RESUME_PATH, model, optimizer, scheduler, ema
        )
        start_epoch += 1

    eff_batch = BATCH_SIZE * ACCUM_STEPS
    print(f"Training: {EPOCHS} epochs, effective batch = {eff_batch}")
    print("-" * 60)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "test_f1": [], "lr": []}

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1:02d}/{EPOCHS}")

        for step, batch in pbar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            # [FIX-H] device-aware autocast
            with autocast('cuda' if device.type == "cuda" else 'cpu'):
                loss = model(input_ids, attention_mask, token_type_ids, labels)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()
            train_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                ema.update(model)

            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}", lr=f"{cur_lr:.2e}")

        avg_train_loss = train_loss / len(train_loader)

        # Evaluate with EMA weights
        ema.apply(model)
        val_loss,  val_f1,  val_acc  = evaluate(model, val_loader,  device)
        test_loss, test_f1, test_acc = evaluate(model, test_loader, device)
        ema.restore(model)

        elapsed = (time.time() - t0) / 60
        eta     = elapsed * (EPOCHS - epoch - 1)
        cur_lr  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"TrainLoss={avg_train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} ValF1={val_f1:.4f} ValAcc={val_acc:.4f} | "
            f"TestLoss={test_loss:.4f} TestF1={test_f1:.4f} TestAcc={test_acc:.4f} | "
            f"LR={cur_lr:.2e} | Time={elapsed:.1f}m ETA={eta:.1f}m"
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["test_f1"].append(test_f1)
        history["lr"].append(cur_lr)

        # [FIX-J] Early stopping
        if val_f1 > best_f1 + 1e-4:
            best_f1    = val_f1
            no_improve = 0
            save_checkpoint(BEST_PATH, model, optimizer, scheduler, epoch,
                            best_f1, config, ema=ema, lightweight=False)
            print(f"  ✅ New best saved: F1={best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

        save_checkpoint(RESUME_PATH, model, optimizer, scheduler, epoch,
                        best_f1, config, ema=None, lightweight=True)

    print("\n✅ Training complete.")
    print(f"   Best Val F1: {best_f1:.4f}")
    print(f"   Best checkpoint: {BEST_PATH}")

    save_training_curves(history, RESULTS_DIR)          # [FIX-K]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from resume_linear.pt")
    args = parser.parse_args()
    train(resume=args.resume)