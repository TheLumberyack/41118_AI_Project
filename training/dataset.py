"""
dataset.py
PyTorch Dataset classes for both models.

LSTMDataset   : sliding-window sequences → next P1 action label
CNNDataset    : (game_state, lstm_probs, counter_action) triples
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    """
    For each frame i, takes the window [i-seq_len .. i-1] of feature
    vectors and labels it with the P1 action at frame i.

    Respects match boundaries so windows never span two matches.
    """

    def __init__(
        self,
        features:    np.ndarray,   # (N, 13)
        p1_actions:  np.ndarray,   # (N,)
        boundaries:  np.ndarray,   # match start indices
        seq_len:     int = 10,
        indices:     np.ndarray | None = None,   # frame indices to use
    ):
        self.features   = torch.tensor(features,   dtype=torch.float32)
        self.p1_actions = torch.tensor(p1_actions, dtype=torch.long)
        self.seq_len    = seq_len

        # Build valid sample indices (frame index i where a full window exists
        # within the same match)
        valid = []
        boundaries_set = set(boundaries.tolist())

        frame_to_match = np.zeros(len(features), dtype=np.int32)
        for m_idx in range(len(boundaries) - 1):
            frame_to_match[boundaries[m_idx]:boundaries[m_idx+1]] = m_idx

        candidate_range = np.arange(seq_len, len(features)) \
            if indices is None else indices[indices >= seq_len]

        for i in candidate_range:
            # All frames in window must be from the same match
            if frame_to_match[i] == frame_to_match[i - seq_len]:
                valid.append(i)

        self.valid_indices = np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i   = self.valid_indices[idx]
        seq = self.features[i - self.seq_len : i]   # (seq_len, 13)
        lbl = self.p1_actions[i]                     # scalar
        return seq, lbl


class CNNDataset(Dataset):
    """
    For training the CNN counter-classifier.

    We use the LSTM's *training-time* predictions (run offline) as the
    lstm_probs input, and the P2 bot action as the counter label.
    This lets us train the CNN without the LSTM being live.

    lstm_probs_cache : (N, 8) pre-computed softmax outputs from trained LSTM
    """

    def __init__(
        self,
        features:         np.ndarray,          # (N, 13)
        p2_actions:       np.ndarray,          # (N,)   counter-action labels
        lstm_probs_cache: np.ndarray,          # (N, 8)
        indices:          np.ndarray | None = None,
    ):
        idx = indices if indices is not None else np.arange(len(features))
        self.features   = torch.tensor(features[idx],         dtype=torch.float32)
        self.p2_actions = torch.tensor(p2_actions[idx],       dtype=torch.long)
        self.lstm_probs = torch.tensor(lstm_probs_cache[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.lstm_probs[idx], self.p2_actions[idx]


# ── Train / val / test split by match ────────────────────────────────────────

def match_split(boundaries: np.ndarray, train=0.70, val=0.15, seed=42):
    """
    Returns (train_frames, val_frames, test_frames) as numpy arrays of
    valid frame indices, split by full matches to prevent data leakage.
    """
    rng = np.random.default_rng(seed)
    n_matches = len(boundaries) - 1
    match_ids = rng.permutation(n_matches)

    n_train = int(n_matches * train)
    n_val   = int(n_matches * val)

    train_m = match_ids[:n_train]
    val_m   = match_ids[n_train:n_train + n_val]
    test_m  = match_ids[n_train + n_val:]

    def get_frames(match_set):
        idx = []
        for m in match_set:
            idx.extend(range(boundaries[m], boundaries[m + 1]))
        return np.array(idx, dtype=np.int64)

    return get_frames(train_m), get_frames(val_m), get_frames(test_m)
