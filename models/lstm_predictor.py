"""
lstm_predictor.py
Model 1: Bidirectional LSTM that predicts the human's next move
given the last N frames of state features.

Input  : (batch, seq_len=10, feature_dim=13)
Output : (batch, N_ACTIONS=8)   — raw logits, apply softmax for probs
"""

import torch
import torch.nn as nn
from utils.features import FEATURE_DIM
from env.fighter_env import N_ACTIONS

SEQ_LEN    = 10
HIDDEN_DIM = 128


class MovePredictor(nn.Module):
    def __init__(
        self,
        input_size:  int = FEATURE_DIM,
        hidden_size: int = HIDDEN_DIM,
        num_layers:  int = 2,
        num_classes: int = N_ACTIONS,
        dropout:     float = 0.3,
    ):
        super().__init__()

        # Bidirectional LSTM — goes beyond the tutorial (uni-directional)
        # hidden output dim = hidden_size * 2 because of bidirectionality
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        lstm_out_dim = hidden_size * 2   # bidirectional doubles the output

        # Two-layer classification head with residual-style dropout
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, feature_dim)
        returns logits : (batch, num_classes)
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq, hidden*2)
        last_step   = lstm_out[:, -1, :]    # take final timestep
        return self.head(last_step)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns softmax probabilities."""
        return torch.softmax(self.forward(x), dim=-1)


# ── Convenience factory ───────────────────────────────────────────────────────
def build_move_predictor(device: str = "cpu") -> MovePredictor:
    model = MovePredictor()
    return model.to(device)
