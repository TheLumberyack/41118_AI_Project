"""
features.py
Converts raw env observation dicts into normalised tensors.
All values mapped to [0, 1] or one-hot encoded.
"""

import numpy as np
import torch
from env.fighter_env import N_ACTIONS, W, H, GROUND_Y

FEATURE_DIM = 13   # must match architecture definitions


def extract_features(obs: dict) -> np.ndarray:
    """
    Convert raw observation dict → normalised float32 array of shape (13,).

    Features:
      0   p1_action      one-hot index / N_ACTIONS
      1   p1_x           normalised by arena width
      2   p1_y           normalised by arena height
      3   p1_health      / 100
      4   p1_facing      mapped {-1→0, 1→1}
      5   p2_x           normalised by arena width
      6   p2_y           normalised by arena height
      7   p2_health      / 100
      8   p2_action      / N_ACTIONS
      9   distance       / arena width
      10  health_delta   / 100  (range -1..1)
      11  round_time     already 0..1
      12  p1_is_attacking  0 or 1
    """
    return np.array([
        obs["p1_action"]      / N_ACTIONS,
        obs["p1_x"]           / W,
        obs["p1_y"]           / GROUND_Y,
        obs["p1_health"]      / 100.0,
        (obs["p1_facing"] + 1) / 2.0,        # {-1,1} → {0,1}
        obs["p2_x"]           / W,
        obs["p2_y"]           / GROUND_Y,
        obs["p2_health"]      / 100.0,
        obs["p2_action"]      / N_ACTIONS,
        obs["distance"]       / W,
        obs["health_delta"]   / 100.0,        # can be negative
        obs["round_time"],
        float(obs["p1_is_attacking"]),
    ], dtype=np.float32)


def obs_to_tensor(obs: dict, device="cpu") -> torch.Tensor:
    """Single observation → (1, FEATURE_DIM) tensor."""
    return torch.tensor(extract_features(obs),
                        dtype=torch.float32,
                        device=device).unsqueeze(0)
