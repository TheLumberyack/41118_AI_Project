#!/usr/bin/env bash
# run.sh — Fighter AI full pipeline
# Run from inside the fighter_ai/ directory.
#
# Usage:
#   bash run.sh              # full pipeline: collect → train → rl → play
#   bash run.sh collect      # Step 1: generate training data
#   bash run.sh train        # Step 2: supervised training (LSTM + CNN)
#   bash run.sh rl           # Step 3: RL self-play fine-tuning
#   bash run.sh ablation     # Step 4: ablation study
#   bash run.sh play         # Step 5: play with supervised model
#   bash run.sh play-rl      # Step 5: play with RL-trained model

set -e
cd "$(dirname "$0")"

STEP=${1:-all}

collect_data() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Step 1: Collecting training data"
    echo "  Simulating 300 bot matches (~2 min)"
    echo "══════════════════════════════════════════"
    python3 training/collect_data.py --matches 300 --seed 42
}

train_lstm() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Step 2a: Training LSTM (Model 1)"
    echo "══════════════════════════════════════════"
    python3 training/train_lstm.py --epochs 3 --batch 256 --lr 1e-3
}

train_cnn() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Step 2b: Training CNN supervised"
    echo "══════════════════════════════════════════"
    python3 training/train_cnn.py --epochs 25 --batch 512 --lr 5e-4
}

train_rl() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Step 3: RL self-play fine-tuning"
    echo "  LSTM frozen — CNN trained via REINFORCE"
    echo "  2000 episodes (~20-40 min on CPU)"
    echo "══════════════════════════════════════════"
    python3 training/train_rl.py --episodes 2000 --lr 1e-4
}

run_ablation() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Step 4: Ablation study"
    echo "══════════════════════════════════════════"
    python3 training/ablation.py --rounds 100
}

launch_game() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Launching game — Supervised AI"
    echo "  Controls:"
    echo "    A/D=Walk  J=Punch  K=Kick  L=Block"
    echo "    Space=Jump  S=Crouch"
    echo "    U=Fireball  I=Uppercut"
    echo "    +/- = AI difficulty"
    echo "══════════════════════════════════════════"
    python3 game/play.py --checkpoint checkpoints/cnn_best.pt
}

launch_game_rl() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Launching game — RL-trained AI"
    echo "  Controls:"
    echo "    A/D=Walk  J=Punch  K=Kick  L=Block"
    echo "    Space=Jump  S=Crouch"
    echo "    U=Fireball  I=Uppercut"
    echo "    +/- = AI difficulty"
    echo "══════════════════════════════════════════"
    python3 game/play.py --checkpoint checkpoints/cnn_rl_best.pt
}

case "$STEP" in
    collect)   collect_data ;;
    train)     train_lstm && train_cnn ;;
    rl)        train_rl ;;
    ablation)  run_ablation ;;
    play)      launch_game ;;
    play-rl)   launch_game_rl ;;
    all)
        collect_data
        train_lstm
        train_cnn
        train_rl
        run_ablation
        launch_game_rl
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Usage: bash run.sh [collect|train|rl|ablation|play|play-rl|all]"
        exit 1
        ;;
esac
