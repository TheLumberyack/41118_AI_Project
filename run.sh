#!/usr/bin/env bash
# run.sh — Fighter AI full pipeline
# Run from inside the fighter_ai/ directory.
#
# Usage:
#   bash run.sh           # runs all steps in order
#   bash run.sh collect   # only collect data
#   bash run.sh train     # only train both models (requires data)
#   bash run.sh play      # launch the game (requires trained models)
#   bash run.sh ablation  # run ablation study (requires trained models)

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
    echo "  Step 2b: Training CNN (Model 2)"
    echo "══════════════════════════════════════════"
    python3 training/train_cnn.py --epochs 25 --batch 512 --lr 5e-4
}

run_ablation() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Step 3: Ablation study"
    echo "══════════════════════════════════════════"
    python3 training/ablation.py --rounds 100
}

launch_game() {
    echo ""
    echo "══════════════════════════════════════════"
    echo "  Launching game — Human vs AI"
    echo "  Controls:"
    echo "    J=Punch  K=Kick  L=Block"
    echo "    Space=Jump  S=Crouch"
    echo "    U=Fireball  I=Uppercut"
    echo "    +/- = AI difficulty"
    echo "══════════════════════════════════════════"
    python3 game/play.py
}

case "$STEP" in
    collect)   collect_data ;;
    train)     train_lstm && train_cnn ;;
    ablation)  run_ablation ;;
    play)      launch_game ;;
    all)
        collect_data
        train_lstm
        train_cnn
        run_ablation
        launch_game
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Usage: bash run.sh [collect|train|ablation|play|all]"
        exit 1
        ;;
esac
