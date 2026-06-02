"""
train_rl.py
Phase 2: Reinforcement learning self-play fine-tuning.

Two CNN agents (P1 and P2) play against each other using the REINFORCE
algorithm. The LSTM is frozen — it acts as a feature extractor only.
Both CNNs start from the supervised checkpoint and improve through
self-play.

Reward signal:
  Every frame : +0.1 per HP of damage dealt, -0.1 per HP damage received
  End of round: +5.0 for winning, -5.0 for losing, 0.0 for draw

Key concepts:
  - Policy gradient: update CNN weights to make winning actions more likely
  - Discounted returns: earlier actions get slightly less credit (gamma=0.99)
  - Self-play: both agents improve together, creating an arms race
  - LSTM frozen: only the CNN is updated by RL

Usage:
  python training/train_rl.py --episodes 2000 --lr 1e-4
  python training/train_rl.py --episodes 500  --lr 1e-4  # quick test
"""

import argparse
import os
import sys
import csv
import time
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from env.fighter_env import FighterEnv, N_ACTIONS, N_AI_ACTIONS
from models.lstm_predictor import MovePredictor, SEQ_LEN
from models.cnn_classifier import CounterClassifier
from utils.features import extract_features


# ── Reward constants ──────────────────────────────────────────────────────────
REWARD_DAMAGE_SCALE   = 0.1     # per HP point dealt / received
REWARD_WIN            =  5.0
REWARD_LOSS           = -5.0
REWARD_DRAW           =  0.0
GAMMA                 = 0.99    # discount factor for future rewards
REWARD_APPROACH_SCALE = 0.003   # reward per pixel closed toward opponent
REWARD_IDLE_PENALTY   = -0.005  # small penalty for idling


# ── Agent — wraps LSTM + CNN for one player ───────────────────────────────────

class RLAgent:
    """
    Encapsulates one player's LSTM + CNN.
    The LSTM is frozen. Only the CNN is trained.

    Maintains:
      - move_buffer : sliding window of last SEQ_LEN opponent observations
      - episode_log_probs : log probabilities of actions taken this episode
      - episode_rewards   : rewards received this episode
    """

    def __init__(self, lstm: MovePredictor, cnn: CounterClassifier,
                 device: str, name: str = "agent"):
        self.lstm   = lstm    # frozen
        self.cnn    = cnn     # trained
        self.device = device
        self.name   = name
        self.reset_episode()

    def reset_episode(self):
        self.move_buffer      = collections.deque(maxlen=SEQ_LEN)
        self.episode_log_probs = []
        self.episode_rewards   = []

    def select_action(self, obs: dict) -> int:
        """
        Given an observation, run LSTM → CNN and sample an action.
        Stores the log_prob for later REINFORCE update.
        """
        feat = extract_features(obs)
        self.move_buffer.append(feat)

        # LSTM forward (no gradient — frozen)
        with torch.no_grad():
            if len(self.move_buffer) == SEQ_LEN:
                seq = torch.tensor(
                    np.array(self.move_buffer),
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)                          # (1, 10, 13)
                lstm_probs = self.lstm.predict_probs(seq)  # (1, 8)
            else:
                lstm_probs = torch.full(
                    (1, N_ACTIONS), 1.0 / N_ACTIONS,
                    device=self.device
                )

        # CNN forward (WITH gradient — being trained)
        game_state = torch.tensor(
            feat, dtype=torch.float32, device=self.device
        ).unsqueeze(0)                                  # (1, 13)

        action, log_prob = self.cnn.sample_action_with_log_prob(
            game_state, lstm_probs
        )

        self.episode_log_probs.append(log_prob)
        return action

    def store_reward(self, reward: float):
        self.episode_rewards.append(reward)

    def compute_returns(self) -> torch.Tensor:
        """
        Convert raw rewards into discounted returns.

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

        Actions near the end of the episode (that contributed to winning
        or losing) get higher returns. Earlier actions are discounted
        because we're less certain they caused the outcome.

        Returns are normalised (zero mean, unit std) to stabilise training.
        """
        returns = []
        G = 0.0
        for r in reversed(self.episode_rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalise — prevents very large or very small gradient updates
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def compute_loss(self) -> torch.Tensor:
        """
        REINFORCE loss:
          L = -sum( log_prob(action_t) * G_t )

        Negative because optimisers minimise loss, but we want to
        maximise expected return. Multiplying by G_t means:
          - Positive return → increase log_prob → make action more likely
          - Negative return → decrease log_prob → make action less likely

        Note: episode_rewards has one extra terminal reward appended after
        the loop (win/loss/draw), so we truncate returns to match log_probs.
        """
        returns   = self.compute_returns()
        log_probs = torch.stack(self.episode_log_probs)

        # Truncate returns to match log_probs length (terminal reward has
        # no corresponding action, so we drop the last return entry)
        min_len = min(len(log_probs), len(returns))
        log_probs = log_probs[:min_len]
        returns   = returns[:min_len]

        loss = -(log_probs * returns).mean()
        return loss


# ── Frame-level reward ────────────────────────────────────────────────────────

def compute_frame_reward(obs_before: dict, obs_after: dict,
                         player: str, action: int) -> float:
    """
    Reward for one frame.
    - Damage dealt/received (main signal)
    - Approach reward: closing distance toward opponent
    - Idle penalty: discourage doing nothing
    obs_before and obs_after are raw env observation dicts (not feature arrays).
    player = 'p1' or 'p2'
    """
    from env.fighter_env import IDLE as IDLE_ACTION

    if player == "p1":
        hp_dealt    = obs_before["p2_health"] - obs_after["p2_health"]
        hp_received = obs_before["p1_health"] - obs_after["p1_health"]
    else:
        hp_dealt    = obs_before["p1_health"] - obs_after["p1_health"]
        hp_received = obs_before["p2_health"] - obs_after["p2_health"]

    damage_reward = (max(0.0, hp_dealt) - max(0.0, hp_received)) * REWARD_DAMAGE_SCALE

    # Approach reward — raw pixel distance, reward closing the gap
    dist_before     = obs_before["distance"]
    dist_after      = obs_after["distance"]
    approach_reward = (dist_before - dist_after) * REWARD_APPROACH_SCALE

    # Idle penalty
    idle_penalty = REWARD_IDLE_PENALTY if action == IDLE_ACTION else 0.0

    return damage_reward + approach_reward + idle_penalty


# ── Self-play episode ─────────────────────────────────────────────────────────

def run_episode(env: FighterEnv,
                agent1: RLAgent,
                agent2: RLAgent) -> dict:
    """
    Run one complete round of self-play.
    Both agents select actions, receive rewards, store log_probs.
    Returns info dict with outcome stats.
    """
    obs, _ = env.reset()
    agent1.reset_episode()
    agent2.reset_episode()

    done          = False
    total_frames  = 0
    p1_damage     = 0.0
    p2_damage     = 0.0

    while not done:
        obs_before = {k: v for k, v in obs.items()}  # snapshot before step

        # Both agents select actions simultaneously
        p1_action = agent1.select_action(obs)
        p2_action = agent2.select_action(obs)

        # Step environment with both agents' actions
        obs, env_reward, done, truncated, info = env.step(p2_action, p1_action)

        # We need P1 to also act — patch: rebuild obs for P1 perspective
        # The env already stepped, so we compute frame rewards from health delta
        frame_r1 = compute_frame_reward(obs_before, obs, "p1", p1_action)
        frame_r2 = compute_frame_reward(obs_before, obs, "p2", p2_action)

        p1_damage += max(0, obs_before["p2_health"] - obs["p2_health"])
        p2_damage += max(0, obs_before["p1_health"] - obs["p1_health"])

        agent1.store_reward(frame_r1)
        agent2.store_reward(frame_r2)
        total_frames += 1

    # Terminal reward based on who won
    if obs["p1_health"] > obs["p2_health"]:
        agent1.store_reward(REWARD_WIN)
        agent2.store_reward(REWARD_LOSS)
        winner = "agent1"
    elif obs["p2_health"] > obs["p1_health"]:
        agent1.store_reward(REWARD_LOSS)
        agent2.store_reward(REWARD_WIN)
        winner = "agent2"
    else:
        agent1.store_reward(REWARD_DRAW)
        agent2.store_reward(REWARD_DRAW)
        winner = "draw"

    return {
        "winner":      winner,
        "frames":      total_frames,
        "p1_hp_left":  obs["p1_health"],
        "p2_hp_left":  obs["p2_health"],
        "p1_damage":   p1_damage,
        "p2_damage":   p2_damage,
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load pretrained supervised checkpoints ────────────────────────────────
    lstm_ckpt = "checkpoints/lstm_best.pt"
    cnn_ckpt  = "checkpoints/cnn_best.pt"

    if not os.path.exists(lstm_ckpt) or not os.path.exists(cnn_ckpt):
        raise FileNotFoundError(
            "Supervised checkpoints not found.\n"
            "Run supervised training first:\n"
            "  python training/collect_data.py\n"
            "  python training/train_lstm.py\n"
            "  python training/train_cnn.py"
        )

    # Shared frozen LSTM — one instance, used by both agents
    lstm = MovePredictor().to(device)
    lstm.load_state_dict(torch.load(lstm_ckpt, map_location=device))
    lstm.eval()
    for param in lstm.parameters():
        param.requires_grad = False   # freeze completely
    print("LSTM loaded and frozen.")

    # Two independent CNN agents — start from same supervised checkpoint
    cnn1 = CounterClassifier().to(device)
    cnn2 = CounterClassifier().to(device)
    cnn1.load_state_dict(torch.load(cnn_ckpt, map_location=device))
    cnn2.load_state_dict(torch.load(cnn_ckpt, map_location=device))
    print("Both CNN agents loaded from supervised checkpoint.")

    # Separate optimisers for each agent
    opt1 = torch.optim.Adam(cnn1.parameters(), lr=args.lr)
    opt2 = torch.optim.Adam(cnn2.parameters(), lr=args.lr)

    # Agents
    agent1 = RLAgent(lstm, cnn1, device, name="agent1")
    agent2 = RLAgent(lstm, cnn2, device, name="agent2")

    # Headless env for RL training
    env = FighterEnv(render_mode=None)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    log_path = "logs/rl_train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "winner", "frames",
            "p1_hp_left", "p2_hp_left",
            "p1_damage", "p2_damage",
            "loss1", "loss2",
            "agent1_wins", "agent2_wins", "draws"
        ])

    agent1_wins = 0
    agent2_wins = 0
    draws       = 0
    best_win_rate = 0.0

    print(f"\nStarting RL self-play for {args.episodes} episodes...\n")
    print(f"{'Episode':>8} {'Winner':>8} {'Frames':>7} "
          f"{'Loss1':>8} {'Loss2':>8} "
          f"{'A1 Win%':>8} {'A2 Win%':>8}")
    print("-" * 70)

    for episode in range(1, args.episodes + 1):
        t0 = time.time()

        # ── Run self-play episode ─────────────────────────────────────────────
        result = run_episode(env, agent1, agent2)

        # Track wins
        if result["winner"] == "agent1":
            agent1_wins += 1
        elif result["winner"] == "agent2":
            agent2_wins += 1
        else:
            draws += 1

        # ── Compute losses ────────────────────────────────────────────────────
        loss1 = agent1.compute_loss()
        loss2 = agent2.compute_loss()

        # ── Update CNN 1 ──────────────────────────────────────────────────────
        opt1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(cnn1.parameters(), max_norm=1.0)
        opt1.step()

        # ── Update CNN 2 ──────────────────────────────────────────────────────
        opt2.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(cnn2.parameters(), max_norm=1.0)
        opt2.step()

        # ── Logging ───────────────────────────────────────────────────────────
        total_ep    = agent1_wins + agent2_wins + draws
        a1_win_rate = agent1_wins / total_ep
        a2_win_rate = agent2_wins / total_ep

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                episode,
                result["winner"],
                result["frames"],
                f"{result['p1_hp_left']:.1f}",
                f"{result['p2_hp_left']:.1f}",
                f"{result['p1_damage']:.1f}",
                f"{result['p2_damage']:.1f}",
                f"{loss1.item():.4f}",
                f"{loss2.item():.4f}",
                agent1_wins, agent2_wins, draws
            ])

        if episode % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"{episode:>8} {result['winner']:>8} "
                  f"{result['frames']:>7} "
                  f"{loss1.item():>8.4f} {loss2.item():>8.4f} "
                  f"{a1_win_rate:>8.1%} {a2_win_rate:>8.1%}")

        # ── Save best agent checkpoint ────────────────────────────────────────
        # Save agent1's CNN as the "best" when it's winning more than 55%
        if episode % args.save_every == 0:
            if a1_win_rate >= a2_win_rate:
                torch.save(cnn1.state_dict(), "checkpoints/cnn_rl_best.pt")
            else:
                torch.save(cnn2.state_dict(), "checkpoints/cnn_rl_best.pt")

            # Always save both for reference
            torch.save(cnn1.state_dict(), "checkpoints/cnn_rl_agent1.pt")
            torch.save(cnn2.state_dict(), "checkpoints/cnn_rl_agent2.pt")

    env.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    total_ep = agent1_wins + agent2_wins + draws
    print("\n" + "=" * 70)
    print("RL SELF-PLAY COMPLETE")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Agent 1 wins: {agent1_wins} ({agent1_wins/total_ep:.1%})")
    print(f"  Agent 2 wins: {agent2_wins} ({agent2_wins/total_ep:.1%})")
    print(f"  Draws:        {draws}       ({draws/total_ep:.1%})")
    print(f"\nCheckpoints saved:")
    print(f"  checkpoints/cnn_rl_best.pt    ← use this for gameplay")
    print(f"  checkpoints/cnn_rl_agent1.pt")
    print(f"  checkpoints/cnn_rl_agent2.pt")
    print(f"\nLogs saved to: {log_path}")
    print("\nTo play with the RL-trained AI:")
    print("  python game/play.py --checkpoint checkpoints/cnn_rl_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int,   default=2000,
                        help="Number of self-play episodes (default 2000)")
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Learning rate for both CNN optimisers (default 1e-4)")
    parser.add_argument("--log-every",  type=int,   default=10,
                        help="Print progress every N episodes (default 10)")
    parser.add_argument("--save-every", type=int,   default=100,
                        help="Save checkpoint every N episodes (default 100)")
    train(parser.parse_args())