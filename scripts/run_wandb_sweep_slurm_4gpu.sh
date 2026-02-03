#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <sweep_id> [extra wandb agent args...]"
  echo "Example: $0 entity/project/abc123"
  exit 1
fi

SWEEP_ID="$1"
shift

# Launch one agent per GPU. Slurm assigns GPUs to each srun.
for _ in $(seq 1 4); do
  srun --exclusive --gres=gpu:1 --cpus-per-task=4 --mem=16G \
    wandb agent "${SWEEP_ID}" "$@" &
done

wait
