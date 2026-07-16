#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="results/phase3/pure_oai_retry_4way_$(date +%Y%m%d_%H%M%S)"
SEED="${SEED:-42}"
HORIZON="${HORIZON:-24}"
STEP_SECONDS="${STEP_SECONDS:-1}"
DQN_CKPT="${DQN_CKPT:-results/offline_checkpoints/dqn_3ue_live_finetune.pt}"
A2C_CKPT="${A2C_CKPT:-results/offline_checkpoints/a2c_3ue_live_finetune.pt}"
TARGET_IP="${TARGET_IP:-12.1.1.1}"
HYBRID_BASE_WEIGHT="${HYBRID_BASE_WEIGHT:-0.40}"
HYBRID_DQN_WEIGHT="${HYBRID_DQN_WEIGHT:-0.30}"
HYBRID_A2C_WEIGHT="${HYBRID_A2C_WEIGHT:-0.30}"
CONTROLLERS=(rule_based dqn a2c rule_based_drl_hybrid)

if docker ps --format '{{.Names}}' | grep -Eiq 'ueransim'; then
  echo "ERROR: UERANSIM container detected" >&2
  exit 2
fi

for controller in "${CONTROLLERS[@]}"; do
  echo "running ${controller}"
  bash scripts/start_ue_traffic_profiles.sh --target-ip "$TARGET_IP" --seed "$SEED" >/dev/null
  sleep 2
  PYTHONPATH=. /usr/bin/python3 scripts/train_drl.py \
    --config configs/openran_live_prom_strict_3ue_congestion_600step.yaml \
    --controllers "$controller" \
    --controller-checkpoint "dqn=$DQN_CKPT" \
    --controller-checkpoint "a2c=$A2C_CKPT" \
    --require-inference-checkpoints \
    --hybrid-dqn-checkpoint "$DQN_CKPT" \
    --hybrid-a2c-checkpoint "$A2C_CKPT" \
    --hybrid-base-weight "$HYBRID_BASE_WEIGHT" \
    --hybrid-dqn-weight "$HYBRID_DQN_WEIGHT" \
    --hybrid-a2c-weight "$HYBRID_A2C_WEIGHT" \
    --hybrid-require-checkpoints \
    --seeds "$SEED" \
    --horizon-steps "$HORIZON" \
    --step-seconds "$STEP_SECONDS" \
    --output-dir "$OUT_DIR" >/dev/null

done

PYTHONPATH=. /usr/bin/python3 scripts/train_drl.py \
  --output-dir "$OUT_DIR" \
  --controllers "${CONTROLLERS[@]}" \
  --compare >/dev/null

/usr/bin/python3 - "$OUT_DIR/comparison_summary.csv" "$OUT_DIR" <<'PY'
import csv
import sys

summary_path, out_dir = sys.argv[1], sys.argv[2]
rows = {r['controller']: r for r in csv.DictReader(open(summary_path))}
rb = float(rows['rule_based']['reward_mean'])
dq = float(rows['dqn']['reward_mean'])
a2 = float(rows['a2c']['reward_mean'])
hy = float(rows['rule_based_drl_hybrid']['reward_mean'])
rbv = float(rows['rule_based']['violations_per_step_mean'])
dqv = float(rows['dqn']['violations_per_step_mean'])
a2v = float(rows['a2c']['violations_per_step_mean'])
hyv = float(rows['rule_based_drl_hybrid']['violations_per_step_mean'])
all_beat = int(dq > rb and a2 > rb and hy > rb)
print(f"OUT_DIR={out_dir}")
print(f"rule_based_reward={rb:.6f}")
print(f"dqn_reward={dq:.6f}")
print(f"a2c_reward={a2:.6f}")
print(f"hybrid_reward={hy:.6f}")
print(f"rule_based_violps={rbv:.6f}")
print(f"dqn_violps={dqv:.6f}")
print(f"a2c_violps={a2v:.6f}")
print(f"hybrid_violps={hyv:.6f}")
print(f"all_beat_rule={all_beat}")
PY
