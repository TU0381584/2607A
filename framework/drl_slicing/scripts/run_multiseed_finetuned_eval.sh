#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

OUT_DIR="${1:-results/phase3/congestion_superiority_finetuned_multiseed_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-42 123 456}"
HORIZON_STEPS="${HORIZON_STEPS:-600}"
TRAFFIC_TARGET="${TRAFFIC_TARGET:-12.1.1.1}"
DQN_CKPT="${DQN_CKPT:-results/offline_checkpoints/dqn_3ue_live_finetune.pt}"
A2C_CKPT="${A2C_CKPT:-results/offline_checkpoints/a2c_3ue_live_finetune.pt}"

if [[ "$DQN_CKPT" != /* ]]; then
  DQN_CKPT="$BASE_DIR/$DQN_CKPT"
fi
if [[ "$A2C_CKPT" != /* ]]; then
  A2C_CKPT="$BASE_DIR/$A2C_CKPT"
fi

if [[ ! -f "$DQN_CKPT" ]]; then
  echo "Missing DQN checkpoint: $DQN_CKPT"
  exit 2
fi
if [[ ! -f "$A2C_CKPT" ]]; then
  echo "Missing A2C checkpoint: $A2C_CKPT"
  exit 2
fi

CONTROLLERS=(dqn a2c rule_based)

bash scripts/verify_live_e2e_stack.sh \
  --expected-ue-count 3 \
  --min-sessions 3 \
  --prom-url "http://127.0.0.1:9090" \
  --target-ip "$TRAFFIC_TARGET"

echo "Running multi-seed finetuned evaluation"
echo "  Output : $OUT_DIR"
echo "  Seeds  : $SEEDS"
echo "  Horizon: $HORIZON_STEPS"

for seed in $SEEDS; do
  echo "=== Seed $seed ==="
  for controller in "${CONTROLLERS[@]}"; do
    echo "--- controller=$controller seed=$seed ---"
    bash scripts/start_ue_traffic_profiles.sh --target-ip "$TRAFFIC_TARGET" --seed "$seed" >/dev/null
    sleep 2

    PYTHONPATH=. /usr/bin/python3 scripts/train_drl.py \
      --config configs/openran_live_prom_strict_3ue_congestion_600step.yaml \
      --controllers "$controller" \
      --controller-checkpoint "dqn=$DQN_CKPT" \
      --controller-checkpoint "a2c=$A2C_CKPT" \
      --require-inference-checkpoints \
      --seeds "$seed" \
      --horizon-steps "$HORIZON_STEPS" \
      --output-dir "$OUT_DIR"
  done
done

OUT_DIR="$OUT_DIR" /usr/bin/python3 - <<'PY'
import csv
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"]).resolve()
for controller_dir in sorted(out_dir.iterdir()):
    if not controller_dir.is_dir():
        continue

    summaries = []
    for seed_dir in sorted(controller_dir.glob("seed_*")):
        run_log = seed_dir / "run_log.csv"
        if not run_log.exists():
            continue

        with run_log.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue

        rewards = [float(row.get("reward", 0.0)) for row in rows]
        violations = [float(row.get("sla_violations", 0.0)) for row in rows]
        seed = int(seed_dir.name.split("_")[-1])
        summaries.append(
            {
                "seed": seed,
                "avg_reward": float(sum(rewards) / len(rewards)),
                "total_reward": float(sum(rewards)),
                "total_sla_violations": float(sum(violations)),
                "avg_sla_violations": float(sum(violations) / len(violations)),
                "num_steps": len(rows),
            }
        )

    if not summaries:
        continue

    summaries.sort(key=lambda item: int(item["seed"]))
    summary_path = controller_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
PY

PYTHONPATH=. /usr/bin/python3 scripts/train_drl.py \
  --output-dir "$OUT_DIR" \
  --controllers "${CONTROLLERS[@]}" \
  --compare

for seed in $SEEDS; do
  seed_plot_dir="$OUT_DIR/plots/seed_${seed}"

  PYTHONPATH=. /usr/bin/python3 scripts/plot_slice_priority_congestion.py \
    --output-dir "$OUT_DIR" \
    --controllers "${CONTROLLERS[@]}" \
    --seed "$seed" \
    --plots-dir "$seed_plot_dir" \
    --ma-window 10 \
    --shock-start 200 \
    --shock-interval 200 \
    --shock-duration 20

  PYTHONPATH=. /usr/bin/python3 scripts/plot_sla_superiority_live.py \
    --output-dir "$OUT_DIR" \
    --controllers "${CONTROLLERS[@]}" \
    --seed "$seed" \
    --plots-dir "$seed_plot_dir" \
    --ma-window 10 \
    --shock-start 200 \
    --shock-interval 200 \
    --shock-duration 20
done

echo "FINAL_OUT_DIR=$OUT_DIR"
