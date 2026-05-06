"""
cnn_classifier.py
Model 2: 1D-CNN counter-action classifier.

Takes the current game state feature vector concatenated with the LSTM's
predicted move probabilities, and outputs the best counter-action for
the AI fighter.

Input  : game_state (batch, FEATURE_DIM=13)
         lstm_probs  (batch, N_ACTIONS=8)
         → concatenated to (batch, 21)
Output : (batch, N_COUNTER_ACTIONS=12) — raw logits
"""

import torch
import torch.nn as nn
from utils.features import FEATURE_DIM
from env.fighter_env import N_ACTIONS

# AI fighter can use all 8 base actions + 4 positional moves (walk L/R + dash)
N_COUNTER_ACTIONS = N_ACTIONS   # keep same action space for simplicity
COMBINED_DIM      = FEATURE_DIM + N_ACTIONS   # 13 + 8 = 21


class CounterClassifier(nn.Module):
    def __init__(
        self,
        input_dim:   int = COMBINED_DIM,
        num_actions: int = N_COUNTER_ACTIONS,
    ):
        super().__init__()

        # 1D CNN over the feature "channels" — treat the 21 features as a
        # 1-channel sequence of length 21. Two conv layers extract local
        # feature interactions (e.g. health × distance, action × facing).
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),   # → (batch, 64, 1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),                           # → (batch, 64)
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        game_state:  torch.Tensor,   # (batch, 13)
        lstm_probs:  torch.Tensor,   # (batch,  8)
    ) -> torch.Tensor:
        """Returns logits of shape (batch, N_COUNTER_ACTIONS)."""
        x = torch.cat([game_state, lstm_probs], dim=-1)   # (batch, 21)
        x = x.unsqueeze(1)                                 # (batch, 1, 21)
        x = self.conv(x)                                   # (batch, 64, 1)
        return self.head(x)                                # (batch, n_actions)

    def predict_action(
        self,
        game_state:  torch.Tensor,
        lstm_probs:  torch.Tensor,
        temperature: float = 1.0,
    ) -> int:
        """
        Sample an action from the softmax distribution.
        temperature < 1.0 → sharper (smarter AI)
        temperature > 1.0 → more random (easier AI)
        """
        logits = self.forward(game_state, lstm_probs)
        probs  = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


def build_counter_classifier(device: str = "cpu") -> CounterClassifier:
    model = CounterClassifier()
    return model.to(device)
