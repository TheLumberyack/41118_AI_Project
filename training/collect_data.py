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

from env.fighter_env import FighterEnv, N_ACTIONS, N_AI_ACTIONS, IDLE, PUNCH, KICK, BLOCK, JUMP, CROUCH, SPECIAL_A, SPECIAL_B, WALK_TOWARD, WALK_AWAY
from utils.features import extract_features


# ── Scripted bots ─────────────────────────────────────────────────────────────
# P1 bots: human-style actions only (0–7, no walking — LSTM predicts these)
# P2 bots: full AI actions (0–9, including WALK_TOWARD/WALK_AWAY)

def aggressive_bot_human(obs):
    """P1 human-style: attacks in range, jumps to close distance."""
    dist = obs["distance"]
    if dist < 80:
        return np.random.choice([PUNCH, KICK, SPECIAL_A], p=[0.4, 0.4, 0.2])
    if dist < 160:
        return np.random.choice([PUNCH, SPECIAL_B], p=[0.6, 0.4])
    return JUMP

def defensive_bot_human(obs):
    """P1 human-style: blocks often, retaliates with specials."""
    dist = obs["distance"]
    hp   = obs["p1_health"] if "p1_health" in obs else 100
    if hp < 30:
        return np.random.choice([SPECIAL_A, SPECIAL_B], p=[0.5, 0.5])
    if dist < 100:
        return np.random.choice([BLOCK, PUNCH], p=[0.6, 0.4])
    return np.random.choice([CROUCH, IDLE], p=[0.5, 0.5])

def combo_bot_human(obs):
    """P1 human-style: spams combo-like sequences."""
    return np.random.choice(
        [PUNCH, KICK, SPECIAL_A, IDLE],
        p=[0.35, 0.35, 0.2, 0.1]
    )

def random_bot_human(obs):
    """P1 human-style: random from 0–7 only."""
    return np.random.randint(N_ACTIONS)

# P2 AI bots — use full action space including walking
def aggressive_bot_ai(obs):
    """P2 AI-style: walks to close distance, attacks in range."""
    dist = obs["distance"]
    if dist < 80:
        return np.random.choice([PUNCH, KICK, SPECIAL_A], p=[0.4, 0.4, 0.2])
    if dist < 160:
        return np.random.choice([PUNCH, SPECIAL_B, WALK_TOWARD], p=[0.4, 0.3, 0.3])
    return np.random.choice([WALK_TOWARD, JUMP], p=[0.8, 0.2])

def defensive_bot_ai(obs):
    """AI-style: blocks, retaliates, retreats when threatened. Uses distance only so works as P1 or P2."""
    dist = obs["distance"]
    if dist < 100:
        return np.random.choice([BLOCK, PUNCH, WALK_AWAY], p=[0.5, 0.3, 0.2])
    return np.random.choice([WALK_TOWARD, CROUCH, IDLE], p=[0.4, 0.3, 0.3])

def combo_bot_ai(obs):
    """P2 AI-style: walks into range then combos."""
    dist = obs["distance"]
    if dist > 150:
        return np.random.choice([WALK_TOWARD, WALK_AWAY], p=[0.8, 0.2])
    return np.random.choice(
        [PUNCH, KICK, SPECIAL_A, WALK_TOWARD, IDLE],
        p=[0.30, 0.30, 0.15, 0.15, 0.10]
    )

def random_bot_ai(obs):
    """P2 AI-style: random from full 0–9 action space."""
    return np.random.randint(N_AI_ACTIONS)

P1_BOTS = [aggressive_bot_human, defensive_bot_human,
           combo_bot_human,      random_bot_human]
P2_BOTS = [aggressive_bot_ai,   defensive_bot_ai,
           combo_bot_ai,         random_bot_ai]


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
        # P1 uses human-style bots (actions 0–7, for LSTM training)
        # P2 uses AI-style bots (actions 0–9, for CNN training)
        p1_bot = np.random.choice(P1_BOTS)
        p2_bot = np.random.choice(P2_BOTS)

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

            obs, _, done, _, info = env.step(p2_action, p1_action)
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
    action_names_full = {**ACTION_NAMES, 8: "Walk Toward", 9: "Walk Away"}
    for a in range(N_AI_ACTIONS):
        cnt = (p1_actions == a).sum()
        print(f"  {action_names_full[a]:12s}: {cnt:6d}  ({100*cnt/len(p1_actions):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=300,
                        help="Number of matches to simulate (default 300)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out",     type=str, default="data",
                        help="Output directory for .npy files")
    args = parser.parse_args()
    collect(args.matches, args.seed, args.out)