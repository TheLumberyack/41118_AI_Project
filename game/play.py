"""
play.py
Human vs AI fighter — the interactive game.

Controls (keyboard):
  J = Punch      K = Kick       L = Block
  Space = Jump   S = Crouch     U = Fireball   I = Uppercut

HUD shows the LSTM's top-3 predicted moves + probabilities in real time.
Use the +/- keys to adjust AI difficulty (temperature).

Usage:
  python game/play.py
  python game/play.py --temperature 0.5   # sharper/harder AI
  python game/play.py --temperature 2.0   # easier AI
"""

import argparse, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pygame
import numpy as np
import torch

from env.fighter_env import FighterEnv, N_ACTIONS, ACTION_NAMES, W, H, FPS
from models.lstm_predictor import MovePredictor, SEQ_LEN
from models.cnn_classifier import CounterClassifier
from utils.features import extract_features


# ── HUD colours (hardcoded — physical-realism display, not themed) ────────────
HUD_BG     = (15, 10, 30, 180)    # semi-transparent dark
BAR_FULL   = (80, 200, 120)
BAR_MED    = (220, 180, 40)
BAR_LOW    = (220, 60,  40)
TEXT_WHITE = (240, 240, 255)
TEXT_GRAY  = (150, 150, 180)
ACCENT     = (100, 160, 255)


def load_models(device):
    lstm = MovePredictor().to(device)
    lstm_ckpt = "checkpoints/lstm_best.pt"
    if not os.path.exists(lstm_ckpt):
        raise FileNotFoundError(
            "checkpoints/lstm_best.pt not found.\n"
            "Run training first:\n"
            "  python training/collect_data.py\n"
            "  python training/train_lstm.py\n"
            "  python training/train_cnn.py"
        )
    lstm.load_state_dict(torch.load(lstm_ckpt, map_location=device))
    lstm.eval()

    cnn = CounterClassifier().to(device)
    cnn.load_state_dict(torch.load("checkpoints/cnn_best.pt",
                                   map_location=device))
    cnn.eval()
    return lstm, cnn


def draw_hud(screen, lstm_probs_np, temperature, round_stats, sfont, tfont):
    """
    Draws the live LSTM prediction HUD in the bottom-left corner.
    Shows top-3 predicted moves with probability bars.
    """
    # HUD panel
    panel = pygame.Surface((260, 130), pygame.SRCALPHA)
    panel.fill(HUD_BG)
    screen.blit(panel, (10, H - 140))

    # Title
    title = sfont.render("LSTM prediction", True, ACCENT)
    screen.blit(title, (18, H - 136))

    # Temp indicator
    temp_str = f"difficulty  {'+' if temperature < 1.0 else '-' * int(temperature)}"
    tsf = sfont.render(f"temp={temperature:.1f}  [+/-]", True, TEXT_GRAY)
    screen.blit(tsf, (18, H - 120))

    # Top-3 bars
    top3_idx  = np.argsort(lstm_probs_np)[::-1][:3]
    for row, idx in enumerate(top3_idx):
        prob   = lstm_probs_np[idx]
        name   = ACTION_NAMES[idx]
        bar_w  = int(prob * 180)
        y_base = H - 105 + row * 28

        col = BAR_FULL if prob > 0.5 else (BAR_MED if prob > 0.25 else BAR_LOW)
        pygame.draw.rect(screen, (40, 30, 60), (18, y_base + 6, 180, 14),
                         border_radius=3)
        if bar_w > 0:
            pygame.draw.rect(screen, col, (18, y_base + 6, bar_w, 14),
                             border_radius=3)
        label = sfont.render(f"{name:<10} {prob:.0%}", True, TEXT_WHITE)
        screen.blit(label, (18, y_base + 5))

    # Round stats
    stats_str = (f"Round {round_stats['round']}  "
                 f"P1 wins: {round_stats['p1_wins']}  "
                 f"AI wins: {round_stats['ai_wins']}")
    st = sfont.render(stats_str, True, TEXT_GRAY)
    screen.blit(st, (W // 2 - st.get_width() // 2, H - 40))


def play(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    lstm_model, cnn_model = load_models(device)

    env  = FighterEnv(render_mode="human")
    obs, _ = env.reset()

    move_buffer = collections.deque(maxlen=SEQ_LEN)
    temperature = args.temperature

    uniform_probs_np = np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32)
    lstm_probs_np    = uniform_probs_np.copy()

    sfont = pygame.font.SysFont("monospace", 13)
    tfont = pygame.font.SysFont("monospace", 18, bold=True)

    round_stats = {"round": 1, "p1_wins": 0, "ai_wins": 0}
    done = False

    print("Game started! Controls: J=Punch K=Kick L=Block Space=Jump "
          "S=Crouch U=Fireball I=Uppercut   +/- = AI difficulty")

    clock = pygame.time.Clock()

    with torch.no_grad():
        while True:
            # Handle quit + difficulty keys directly here (env reads others)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_EQUALS:
                        temperature = max(0.2, temperature - 0.1)
                        print(f"Temperature → {temperature:.1f} (harder)")
                    if event.key == pygame.K_MINUS:
                        temperature = min(3.0, temperature + 0.1)
                        print(f"Temperature → {temperature:.1f} (easier)")
                    if event.key == pygame.K_r and done:
                        obs, _ = env.reset()
                        move_buffer.clear()
                        done = False

            if done:
                # Show "Press R to restart" message
                msg = tfont.render("Press R to play again", True, (220, 220, 80))
                env._screen.blit(
                    msg, (W // 2 - msg.get_width() // 2, H // 2))
                pygame.display.flip()
                clock.tick(FPS)
                continue

            # Feature extraction
            feat = extract_features(obs)
            move_buffer.append(feat)

            # Model 1 — LSTM
            if len(move_buffer) == SEQ_LEN:
                seq = torch.tensor(
                    np.array(move_buffer), dtype=torch.float32,
                    device=device).unsqueeze(0)
                lstm_probs    = lstm_model.predict_probs(seq)
                lstm_probs_np = lstm_probs.cpu().numpy()[0]
            else:
                lstm_probs    = torch.tensor(
                    uniform_probs_np, device=device).unsqueeze(0)
                lstm_probs_np = uniform_probs_np

            # Model 2 — CNN
            game_state = torch.tensor(
                feat, dtype=torch.float32, device=device).unsqueeze(0)
            ai_action = cnn_model.predict_action(
                game_state, lstm_probs, temperature=temperature)

            # Step
            obs, reward, done, truncated, info = env.step(ai_action)

            if done:
                if reward > 0:
                    round_stats["ai_wins"] += 1
                elif reward < 0:
                    round_stats["p1_wins"] += 1
                round_stats["round"] += 1

            # Draw HUD on top of the env frame
            draw_hud(env._screen, lstm_probs_np, temperature,
                     round_stats, sfont, tfont)
            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="AI randomness: lower=harder, higher=easier (default 0.8)")
    play(parser.parse_args())
