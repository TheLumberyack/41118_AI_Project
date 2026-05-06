"""
ablation.py
HD evaluation: compare CNN-alone vs CNN+LSTM pipeline.

Runs N complete rounds against three opponent types and reports:
  - Win rate
  - Average health remaining
  - Inference latency (ms/frame)

Usage:
  python ablation.py --rounds 100
"""

import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import collections

from env.fighter_env import FighterEnv, N_ACTIONS
from models.lstm_predictor import MovePredictor, SEQ_LEN
from models.cnn_classifier import CounterClassifier
from utils.features import extract_features
from training.collect_data import aggressive_bot, defensive_bot, random_bot


def load_models(device):
    lstm = MovePredictor().to(device)
    lstm.load_state_dict(torch.load("checkpoints/lstm_best.pt",
                                    map_location=device))
    lstm.eval()

    cnn = CounterClassifier().to(device)
    cnn.load_state_dict(torch.load("checkpoints/cnn_best.pt",
                                   map_location=device))
    cnn.eval()
    return lstm, cnn


@torch.no_grad()
def run_episodes(use_lstm, lstm, cnn, p1_bot, n_rounds, device,
                 temperature=0.8):
    """Returns (win_rate, avg_ai_health_remaining, avg_latency_ms)."""
    env    = FighterEnv(render_mode=None)
    wins   = 0
    health_remaining = []
    latencies = []
    buffer = collections.deque(maxlen=SEQ_LEN)

    uniform_probs = torch.full((1, N_ACTIONS), 1.0 / N_ACTIONS,
                               device=device)

    for _ in range(n_rounds):
        obs, _ = env.reset()
        buffer.clear()
        done = False

        while not done:
            feat = extract_features(obs)
            buffer.append(feat)

            t0 = time.perf_counter()

            if use_lstm and len(buffer) == SEQ_LEN:
                seq  = torch.tensor(np.array(buffer),
                                    dtype=torch.float32,
                                    device=device).unsqueeze(0)
                lstm_probs = lstm.predict_probs(seq)
            else:
                lstm_probs = uniform_probs

            game_state = torch.tensor(feat, dtype=torch.float32,
                                      device=device).unsqueeze(0)
            action = cnn.predict_action(game_state, lstm_probs,
                                        temperature=temperature)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            obs, reward, done, _, _ = env.step(action)

        if reward > 0:   # AI won
            wins += 1
        health_remaining.append(obs["p2_health"])

    env.close()
    return wins / n_rounds, np.mean(health_remaining), np.mean(latencies)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lstm, cnn = load_models(device)

    bots = [
        ("Random agent",    random_bot),
        ("Aggressive bot",  aggressive_bot),
        ("Defensive bot",   defensive_bot),
    ]

    print("=" * 60)
    print("ABLATION STUDY: CNN alone vs CNN + LSTM")
    print(f"Rounds per condition: {args.rounds}")
    print("=" * 60)

    results = {}
    for bot_name, bot_fn in bots:
        print(f"\nOpponent: {bot_name}")
        for use_lstm in [False, True]:
            label = "CNN + LSTM" if use_lstm else "CNN alone"
            wr, hp, lat = run_episodes(use_lstm, lstm, cnn, bot_fn,
                                       args.rounds, device)
            key = (bot_name, label)
            results[key] = (wr, hp, lat)
            print(f"  {label:12s} | win rate: {wr:.1%} | "
                  f"avg HP left: {hp:.1f} | latency: {lat:.2f}ms/frame")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print(f"{'Opponent':<18} {'Model':<14} {'Win %':>7} "
          f"{'HP left':>9} {'ms/frame':>10}")
    print("-" * 60)
    for (bot_name, label), (wr, hp, lat) in results.items():
        print(f"{bot_name:<18} {label:<14} {wr:>6.1%} "
              f"{hp:>9.1f} {lat:>10.2f}")

    # Save results
    os.makedirs("logs", exist_ok=True)
    import csv
    with open("logs/ablation_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opponent", "model", "win_rate",
                    "avg_hp_remaining", "avg_latency_ms"])
        for (bot_name, label), (wr, hp, lat) in results.items():
            w.writerow([bot_name, label, f"{wr:.4f}",
                        f"{hp:.2f}", f"{lat:.4f}"])
    print("\nSaved: logs/ablation_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=100)
    main(parser.parse_args())
