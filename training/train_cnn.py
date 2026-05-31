"""
train_cnn.py
Stage 2: Train the CNN counter-action classifier (Model 2).

Requires lstm_probs_cache.npy produced by train_lstm.py.

Saves:
  checkpoints/cnn_best.pt
  logs/cnn_train_log.csv
  plots/cnn_confusion.png

Usage:
  python train_cnn.py --epochs 25 --batch 512 --lr 5e-4
"""

import argparse, os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.cnn_classifier import CounterClassifier
from training.dataset import CNNDataset, match_split
from env.fighter_env import N_ACTIONS, N_AI_ACTIONS, N_AI_ACTIONS, ACTION_NAMES
from utils.eval_metrics import (
    compute_f1_macro, confusion_matrix_plot
)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for game_state, lstm_probs, lbl in loader:
        game_state = game_state.to(device)
        lstm_probs = lstm_probs.to(device)
        lbl        = lbl.to(device)
        optimizer.zero_grad()
        logits = model(game_state, lstm_probs)
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
    all_preds, all_labels = [], []
    for game_state, lstm_probs, lbl in loader:
        game_state = game_state.to(device)
        lstm_probs = lstm_probs.to(device)
        lbl        = lbl.to(device)
        logits     = model(game_state, lstm_probs)
        loss       = loss_fn(logits, lbl)
        total_loss += loss.item() * len(lbl)
        correct    += (logits.argmax(1) == lbl).sum().item()
        total      += len(lbl)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())
    f1 = compute_f1_macro(all_labels, all_preds, N_ACTIONS)
    return total_loss / total, correct / total, f1, \
           np.array(all_labels), np.array(all_preds)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir   = args.data
    features   = np.load(os.path.join(data_dir, "raw_frames.npy"))
    p2_actions = np.load(os.path.join(data_dir, "p2_actions.npy"))
    boundaries = np.load(os.path.join(data_dir, "match_boundaries.npy"))
    lstm_probs = np.load(os.path.join(data_dir, "lstm_probs_cache.npy"))

    train_idx, val_idx, test_idx = match_split(boundaries, seed=args.seed)

    train_ds = CNNDataset(features, p2_actions, lstm_probs, indices=train_idx)
    val_ds   = CNNDataset(features, p2_actions, lstm_probs, indices=val_idx)
    test_ds  = CNNDataset(features, p2_actions, lstm_probs, indices=test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | "
          f"Test: {len(test_ds):,} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = CounterClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 5,
        steps_per_epoch=len(train_loader), epochs=args.epochs)

    counts  = np.bincount(p2_actions[train_idx], minlength=N_AI_ACTIONS)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * N_ACTIONS
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)
    os.makedirs("plots",       exist_ok=True)

    log_path = "logs/cnn_train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","train_acc",
                                 "val_loss","val_acc","val_f1"])

    best_val_f1 = 0.0
    print(f"\nTraining CNN for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        va_loss, va_acc, va_f1, *_ = evaluate(
            model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss {tr_loss:.4f} → {va_loss:.4f} | "
              f"acc {tr_acc:.3f} → {va_acc:.3f} | "
              f"val F1 {va_f1:.3f} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, tr_loss, tr_acc, va_loss, va_acc, va_f1])

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), "checkpoints/cnn_best.pt")
            print(f"  ✓ Saved best CNN (val F1={va_f1:.4f})")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\n── Test set evaluation ──────────────────────────────────────")
    model.load_state_dict(torch.load("checkpoints/cnn_best.pt",
                                     map_location=device))
    te_loss, te_acc, te_f1, labels, preds = evaluate(
        model, test_loader, loss_fn, device)
    print(f"Test loss: {te_loss:.4f} | acc: {te_acc:.4f} | F1: {te_f1:.4f}")

    action_labels = [ACTION_NAMES[i] for i in range(N_ACTIONS)]
    confusion_matrix_plot(labels, preds, action_labels,
                          save_path="plots/cnn_confusion.png",
                          title="CNN — counter-action confusion matrix")
    print("Done. Run the game with: python game/play.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=25)
    parser.add_argument("--batch",  type=int,   default=512)
    parser.add_argument("--lr",     type=float, default=5e-4)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--data",   type=str,   default="data")
    main(parser.parse_args())
