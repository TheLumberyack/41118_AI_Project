"""
collect_data.py
Runs scripted bots against each other headlessly to generate training data.

Produces:
  data/raw_frames.npy    — shape (N, 13) normalised feature vectors
  data/p1_actions.npy    — shape (N,)    ground-truth P1 action each frame
  data/p2_actions.npy    — shape (N,)    ground-truth P2 action each frame
  data/match_boundaries.npy — indices where new matches start (for split)

Usage:
  python collect_data.py --matches 300 --seed 42
"""

import argparse
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.fighter_env import FighterEnv, N_ACTIONS, IDLE, PUNCH, KICK, BLOCK, JUMP, CROUCH, SPECIAL_A, SPECIAL_B
from utils.features import extract_features


# ── Scripted bots ─────────────────────────────────────────────────────────────

def aggressive_bot(obs):
    """Attacks whenever in range, jumps to close distance."""
    dist = obs["distance"]
    if dist < 80:
        return np.random.choice([PUNCH, KICK, SPECIAL_A], p=[0.4, 0.4, 0.2])
    if dist < 160:
        return np.random.choice([PUNCH, SPECIAL_B], p=[0.6, 0.4])
    return JUMP

def defensive_bot(obs):
    """Blocks often, retaliates with specials."""
    dist = obs["distance"]
    hp   = obs["p1_health"] if "p1_health" in obs else 100
    if hp < 30:
        return np.random.choice([SPECIAL_A, SPECIAL_B], p=[0.5, 0.5])
    if dist < 100:
        return np.random.choice([BLOCK, PUNCH], p=[0.6, 0.4])
    return np.random.choice([CROUCH, IDLE], p=[0.5, 0.5])

def combo_bot(obs):
    """Spams combo-like sequences."""
    return np.random.choice(
        [PUNCH, KICK, SPECIAL_A, IDLE],
        p=[0.35, 0.35, 0.2, 0.1]
    )

def random_bot(obs):
    return np.random.randint(N_ACTIONS)

BOTS = [aggressive_bot, defensive_bot, combo_bot, random_bot]


# ── Collection loop ───────────────────────────────────────────────────────────

def collect(num_matches: int, seed: int, out_dir: str):
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    env = FighterEnv(render_mode=None)

    all_features   = []
    all_p1_actions = []
    all_p2_actions = []
    match_boundaries = [0]

    print(f"Collecting {num_matches} matches...")

    for match_idx in range(num_matches):
        # Randomly pair two bots
        p1_bot = np.random.choice(BOTS)
        p2_bot = np.random.choice(BOTS)

        obs, _ = env.reset()
        done   = False
        match_frames = 0

        while not done:
            feat = extract_features(obs)
            p1_action = p1_bot(obs)
            p2_action = p2_bot(obs)

            all_features.append(feat)
            all_p1_actions.append(p1_action)
            all_p2_actions.append(p2_action)

            obs, _, done, _, info = env.step(p2_action)
            match_frames += 1

        match_boundaries.append(len(all_features))

        if (match_idx + 1) % 50 == 0:
            pct = 100 * (match_idx + 1) / num_matches
            print(f"  {match_idx+1}/{num_matches} ({pct:.0f}%) — "
                  f"{len(all_features):,} frames so far")

    env.close()

    features   = np.array(all_features,   dtype=np.float32)
    p1_actions = np.array(all_p1_actions, dtype=np.int64)
    p2_actions = np.array(all_p2_actions, dtype=np.int64)
    boundaries = np.array(match_boundaries, dtype=np.int64)

    np.save(os.path.join(out_dir, "raw_frames.npy"),       features)
    np.save(os.path.join(out_dir, "p1_actions.npy"),       p1_actions)
    np.save(os.path.join(out_dir, "p2_actions.npy"),       p2_actions)
    np.save(os.path.join(out_dir, "match_boundaries.npy"), boundaries)

    print(f"\nDone. Saved {len(features):,} frames from {num_matches} matches.")
    print(f"Files written to: {out_dir}/")
    print(f"\nAction distribution (P1):")
    from env.fighter_env import ACTION_NAMES
    for a in range(N_ACTIONS):
        cnt = (p1_actions == a).sum()
        print(f"  {ACTION_NAMES[a]:12s}: {cnt:6d}  ({100*cnt/len(p1_actions):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=300,
                        help="Number of matches to simulate (default 300)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out",     type=str, default="data",
                        help="Output directory for .npy files")
    args = parser.parse_args()
    collect(args.matches, args.seed, args.out)
