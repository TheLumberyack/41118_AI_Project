"""
play.py
Human vs AI fighter — the interactive game.

Controls:
  A / D    = Walk left / right
  J        = Punch        K = Kick
  L        = Block        Space = Jump
  S        = Crouch       U = Fireball    I = Uppercut
  + / -    = AI harder / easier
  R        = Restart after round ends

Usage:
  python game/play.py
  python game/play.py --temperature 0.5   # harder AI
  python game/play.py --temperature 2.0   # easier AI
  python game/play.py --no-servo          # disable Arduino servo
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
from servo_controller import ServoController

# ── HUD colours ───────────────────────────────────────────────────────────────
HUD_BG     = (15, 10, 30, 180)
BAR_FULL   = (80, 200, 120)
BAR_MED    = (220, 180, 40)
BAR_LOW    = (220, 60,  40)
TEXT_WHITE = (240, 240, 255)
TEXT_GRAY  = (150, 150, 180)
ACCENT     = (100, 160, 255)
YELLOW     = (255, 220, 0)


def load_models(device):
    lstm_ckpt = "checkpoints/lstm_best.pt"
    cnn_ckpt  = "checkpoints/cnn_best.pt"
    if not os.path.exists(lstm_ckpt) or not os.path.exists(cnn_ckpt):
        raise FileNotFoundError(
            "Trained model checkpoints not found.\n"
            "Run training first:\n"
            "  python training/collect_data.py\n"
            "  python training/train_lstm.py\n"
            "  python training/train_cnn.py"
        )
    lstm = MovePredictor().to(device)
    lstm.load_state_dict(torch.load(lstm_ckpt, map_location=device))
    lstm.eval()

    cnn = CounterClassifier().to(device)
    cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device))
    cnn.eval()
    return lstm, cnn


def draw_hud(screen, lstm_probs_np, temperature, round_stats, sfont, tfont):
    # Semi-transparent panel
    panel = pygame.Surface((265, 135), pygame.SRCALPHA)
    panel.fill(HUD_BG)
    screen.blit(panel, (10, H - 145))

    screen.blit(sfont.render("LSTM prediction", True, ACCENT), (18, H - 141))
    screen.blit(sfont.render(f"temp={temperature:.1f}  [+/-]", True, TEXT_GRAY),
                (18, H - 125))

    # Top-3 predicted moves
    top3 = np.argsort(lstm_probs_np)[::-1][:3]
    for row, idx in enumerate(top3):
        prob   = lstm_probs_np[idx]
        bar_w  = int(prob * 185)
        y_base = H - 108 + row * 28
        col    = BAR_FULL if prob > 0.5 else (BAR_MED if prob > 0.25 else BAR_LOW)

        pygame.draw.rect(screen, (40, 30, 60), (18, y_base + 8, 185, 14), border_radius=3)
        if bar_w > 0:
            pygame.draw.rect(screen, col, (18, y_base + 8, bar_w, 14), border_radius=3)
        screen.blit(
            sfont.render(f"{ACTION_NAMES[idx]:<10} {prob:.0%}", True, TEXT_WHITE),
            (18, y_base + 7)
        )

    # Round stats bottom-centre
    stats = (f"Round {round_stats['round']}   "
             f"P1 wins: {round_stats['p1_wins']}   "
             f"AI wins: {round_stats['ai_wins']}")
    st = sfont.render(stats, True, TEXT_GRAY)
    screen.blit(st, (W // 2 - st.get_width() // 2, H - 40))


def show_round_end(screen, tfont, winner_text, color):
    """Flash a winner message in the centre of the screen."""
    msg = tfont.render(winner_text, True, color)
    screen.blit(msg, (W // 2 - msg.get_width() // 2, H // 2 - 30))
    r   = tfont.render("Press R to play again", True, (200, 200, 80))
    screen.blit(r,   (W // 2 - r.get_width()   // 2, H // 2 + 10))


def play(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    lstm_model, cnn_model = load_models(device)

    # Servo — skips gracefully if Arduino not connected or --no-servo passed
    servo = ServoController() if not args.no_servo else None
    if servo and not servo.is_connected():
        print("No Arduino detected — servo disabled.")
        servo = None

    env  = FighterEnv(render_mode="human")
    obs, _ = env.reset()

    move_buffer      = collections.deque(maxlen=SEQ_LEN)
    temperature      = args.temperature
    uniform_probs_np = np.full(N_ACTIONS, 1.0 / N_ACTIONS, dtype=np.float32)
    lstm_probs_np    = uniform_probs_np.copy()

    sfont = pygame.font.SysFont("monospace", 13)
    tfont = pygame.font.SysFont("monospace", 20, bold=True)

    round_stats  = {"round": 1, "p1_wins": 0, "ai_wins": 0}
    done         = False
    winner_text  = ""
    winner_color = (255, 255, 255)

    print("Game started!")
    print("Controls: A/D=Walk  J=Punch  K=Kick  L=Block  "
          "Space=Jump  S=Crouch  U=Fireball  I=Uppercut  +/-=Difficulty")

    clock = pygame.time.Clock()

    with torch.no_grad():
        while True:
            # ── Event handling ────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if servo:
                        servo.close()
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_EQUALS:
                        temperature = max(0.2, temperature - 0.1)
                        print(f"Temperature → {temperature:.1f}  (harder AI)")
                    if event.key == pygame.K_MINUS:
                        temperature = min(3.0, temperature + 0.1)
                        print(f"Temperature → {temperature:.1f}  (easier AI)")
                    if event.key == pygame.K_r and done:
                        obs, _ = env.reset()
                        move_buffer.clear()
                        lstm_probs_np = uniform_probs_np.copy()
                        done          = False
                        winner_text   = ""

            # ── Round-end screen ──────────────────────────────────────────────
            if done:
                show_round_end(env._screen, tfont, winner_text, winner_color)
                pygame.display.flip()
                clock.tick(FPS)
                continue

            # ── Feature extraction ────────────────────────────────────────────
            feat = extract_features(obs)
            move_buffer.append(feat)

            # ── Model 1: LSTM predicts human's next move ──────────────────────
            if len(move_buffer) == SEQ_LEN:
                seq = torch.tensor(
                    np.array(move_buffer), dtype=torch.float32,
                    device=device).unsqueeze(0)
                lstm_probs    = lstm_model.predict_probs(seq)
                lstm_probs_np = lstm_probs.cpu().numpy()[0]
            else:
                lstm_probs = torch.tensor(
                    uniform_probs_np, device=device).unsqueeze(0)

            # ── Model 2: CNN picks counter-action ─────────────────────────────
            game_state = torch.tensor(
                feat, dtype=torch.float32, device=device).unsqueeze(0)
            ai_action = cnn_model.predict_action(
                game_state, lstm_probs, temperature=temperature)

            # ── Step environment ──────────────────────────────────────────────
            obs, reward, done, truncated, info = env.step(ai_action)

            # ── Round result ──────────────────────────────────────────────────
            if done:
                round_stats["round"] += 1
                if reward < 0:
                    # Human won — AI died → trigger servo
                    round_stats["p1_wins"] += 1
                    winner_text  = "You win!"
                    winner_color = (80, 200, 120)
                    if servo:
                        servo.trigger_death()
                elif reward > 0:
                    # AI won
                    round_stats["ai_wins"] += 1
                    winner_text  = "AI wins!"
                    winner_color = (220, 80, 60)
                else:
                    winner_text  = "Draw!"
                    winner_color = (200, 200, 80)

            # ── Draw HUD then flip once ───────────────────────────────────────
            # env._render_frame() already drew the game to the surface.
            # We draw the HUD on top, then flip exactly once.
            draw_hud(env._screen, lstm_probs_np, temperature,
                     round_stats, sfont, tfont)
            pygame.display.flip()
            clock.tick(FPS)

    if servo:
        servo.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Softmax temperature: lower=harder, higher=easier")
    parser.add_argument("--no-servo", action="store_true",
                        help="Disable Arduino servo integration")
    play(parser.parse_args())