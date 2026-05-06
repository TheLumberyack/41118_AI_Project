"""
train_lstm.py
Stage 1: Train the LSTM move predictor (Model 1).

Saves:
  checkpoints/lstm_best.pt     — best val F1 checkpoint
  logs/lstm_train_log.csv      — epoch-level metrics
  plots/lstm_confusion.png     — confusion matrix on test set
  plots/lstm_roc.png           — per-class ROC curves

Usage:
  python train_lstm.py --epochs 3 --batch 256 --lr 1e-3
"""

import argparse, os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.lstm_predictor import MovePredictor, SEQ_LEN
from training.dataset import LSTMDataset, match_split
from env.fighter_env import N_ACTIONS, ACTION_NAMES
from utils.eval_metrics import (
    compute_f1_macro, confusion_matrix_plot, roc_curves_plot
)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for seq, lbl in loader:
        seq, lbl = seq.to(device), lbl.to(device)
        optimizer.zero_grad()
        logits = model(seq)
        loss   = loss_fn(logits, lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(lbl)
        correct    += (logits.argmax(1) == lbl).sum().item()
        total      += len(lbl)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for seq, lbl in loader:
        seq, lbl = seq.to(device), lbl.to(device)
        logits = model(seq)
        loss   = loss_fn(logits, lbl)
        probs  = torch.softmax(logits, dim=-1)
        total_loss += loss.item() * len(lbl)
        correct    += (logits.argmax(1) == lbl).sum().item()
        total      += len(lbl)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    f1 = compute_f1_macro(all_labels, all_preds, N_ACTIONS)
    return (total_loss / total, correct / total, f1,
            np.array(all_labels), np.array(all_preds), np.array(all_probs))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir = args.data
    features   = np.load(os.path.join(data_dir, "raw_frames.npy"))
    p1_actions = np.load(os.path.join(data_dir, "p1_actions.npy"))
    boundaries = np.load(os.path.join(data_dir, "match_boundaries.npy"))

    train_idx, val_idx, test_idx = match_split(boundaries, seed=args.seed)
    print(f"Split: {len(train_idx):,} train / {len(val_idx):,} val / "
          f"{len(test_idx):,} test frames")

    train_ds = LSTMDataset(features, p1_actions, boundaries,
                           seq_len=SEQ_LEN, indices=train_idx)
    val_ds   = LSTMDataset(features, p1_actions, boundaries,
                           seq_len=SEQ_LEN, indices=val_idx)
    test_ds  = LSTMDataset(features, p1_actions, boundaries,
                           seq_len=SEQ_LEN, indices=test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model + training setup ────────────────────────────────────────────────
    model     = MovePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Class weights to handle imbalance (IDLE is very frequent)
    counts    = np.bincount(p1_actions[train_idx], minlength=N_ACTIONS)
    weights   = 1.0 / (counts + 1e-6)
    weights   = weights / weights.sum() * N_ACTIONS
    loss_fn   = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)
    os.makedirs("plots",       exist_ok=True)

    log_path = "logs/lstm_train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","train_acc",
                         "val_loss","val_acc","val_f1","lr"])

    best_val_f1 = 0.0
    print(f"\nTraining LSTM for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        va_loss, va_acc, va_f1, *_ = evaluate(
            model, val_loader, loss_fn, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss {tr_loss:.4f} → {va_loss:.4f} | "
              f"acc {tr_acc:.3f} → {va_acc:.3f} | "
              f"val F1 {va_f1:.3f} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, tr_loss, tr_acc, va_loss, va_acc, va_f1, lr])

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), "checkpoints/lstm_best.pt")
            print(f"  ✓ Saved best model (val F1={va_f1:.4f})")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\n── Test set evaluation ──────────────────────────────────────")
    model.load_state_dict(torch.load("checkpoints/lstm_best.pt",
                                     map_location=device))
    te_loss, te_acc, te_f1, labels, preds, probs = evaluate(
        model, test_loader, loss_fn, device)

    print(f"Test loss:     {te_loss:.4f}")
    print(f"Test accuracy: {te_acc:.4f}")
    print(f"Test F1 macro: {te_f1:.4f}")

    action_labels = [ACTION_NAMES[i] for i in range(N_ACTIONS)]
    confusion_matrix_plot(labels, preds, action_labels,
                          save_path="plots/lstm_confusion.png",
                          title="LSTM — move prediction confusion matrix")
    roc_curves_plot(labels, probs, action_labels,
                    save_path="plots/lstm_roc.png",
                    title="LSTM — per-class ROC curves")
    print("Plots saved to plots/")

    # ── Pre-compute LSTM probs for CNN training ───────────────────────────────
    print("\nPre-computing LSTM probs over full dataset for CNN training...")
    all_loader = DataLoader(
        LSTMDataset(features, p1_actions, boundaries, seq_len=SEQ_LEN),
        batch_size=512, shuffle=False, num_workers=2)

    model.eval()
    all_probs_list = []
    with torch.no_grad():
        for seq, _ in all_loader:
            seq = seq.to(device)
            probs = torch.softmax(model(seq), dim=-1)
            all_probs_list.append(probs.cpu().numpy())

    # Align back to full frame count (first seq_len frames have no window)
    # We pad the front with uniform probs
    full_probs = np.full((len(features), N_ACTIONS),
                         1.0 / N_ACTIONS, dtype=np.float32)
    valid_ds   = LSTMDataset(features, p1_actions, boundaries, seq_len=SEQ_LEN)
    valid_idx  = valid_ds.valid_indices
    concatenated = np.concatenate(all_probs_list, axis=0)
    full_probs[valid_idx] = concatenated

    np.save(os.path.join(data_dir, "lstm_probs_cache.npy"), full_probs)
    print(f"Saved lstm_probs_cache.npy → {data_dir}/")
    print("\nStage 1 complete. Run train_cnn.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=30)
    parser.add_argument("--batch",  type=int,   default=256)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--data",   type=str,   default="data")
    main(parser.parse_args())
