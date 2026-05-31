"""
cnn_classifier.py
Model 2: 1D-CNN counter-action classifier.

Takes the current game state feature vector concatenated with the LSTM's
predicted move probabilities, and outputs the best counter-action for
the AI fighter.

Input  : game_state (batch, FEATURE_DIM=13)
         lstm_probs  (batch, N_ACTIONS=8)
         → concatenated to (batch, 21)
Output : (batch, N_COUNTER_ACTIONS=10) — raw logits

RL addition:
  log_prob(game_state, lstm_probs, action) — returns log probability of a
  specific action, used by REINFORCE to compute the policy gradient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.features import FEATURE_DIM
from env.fighter_env import N_ACTIONS, N_AI_ACTIONS

N_COUNTER_ACTIONS = N_AI_ACTIONS        # 10 actions including walk toward/away
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
        game_state: torch.Tensor,   # (batch, 13)
        lstm_probs: torch.Tensor,   # (batch,  8)
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
        Used during gameplay (no gradient tracking needed).
        """
        logits = self.forward(game_state, lstm_probs)
        probs  = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def sample_action_with_log_prob(
        self,
        game_state: torch.Tensor,   # (1, 13)
        lstm_probs: torch.Tensor,   # (1,  8)
    ) -> tuple[int, torch.Tensor]:
        """
        Sample an action AND return its log probability.
        Used during RL training — the log_prob is needed for REINFORCE.

        Returns:
            action   : int — the sampled action index
            log_prob : scalar tensor — log P(action | state)
                       kept in the computation graph so gradients
                       can flow back through the CNN during backprop.
        """
        logits = self.forward(game_state, lstm_probs)      # (1, 10)
        probs  = torch.softmax(logits, dim=-1)             # (1, 10)

        # Sample from the distribution (exploration)
        action = torch.multinomial(probs, num_samples=1)   # (1, 1)

        # log_prob of the chosen action — this is what REINFORCE differentiates
        log_prob = torch.log(probs.squeeze(0)[action.item()] + 1e-8)

        return action.item(), log_prob


def build_counter_classifier(device: str = "cpu") -> CounterClassifier:
    model = CounterClassifier()
    return model.to(device)
