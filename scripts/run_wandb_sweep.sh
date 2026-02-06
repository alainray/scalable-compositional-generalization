#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH -t 2-00:00
#SBATCH -o /workspace1/asoto/araymond/scalable-compositional-generalization/exp_logs/%x_%j.out
#SBATCH -e /workspace1/asoto/araymond/scalable-compositional-generalization/exp_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=afraymon@uc.cl
#SBATCH --chdir=/home/araymond
#SBATCH --partition=ialab
#SBATCH --nodelist=ventress
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --mem=60G

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <sweep_id|entity/project/sweep_id> [extra wandb agent args...]"
  echo "Example: $0 entity/project/abc123"
  echo "Example (short ID with env): WANDB_ENTITY=my_entity WANDB_PROJECT=visgen $0 abc123"
  exit 1
fi

SWEEP_ID="$1"
WANDB_ENTITY="alainray_puc"
shift

# Activa tu env si aplica
# source activate tu_env

cd /workspace1/asoto/araymond/scalable-compositional-generalization


FULL_SWEEP_ID="${SWEEP_ID}"
if [[ "${SWEEP_ID}" != */*/* ]]; then
  if [ -z "${WANDB_ENTITY:-}" ]; then
    echo "Error: SWEEP_ID is missing entity/project. Set WANDB_ENTITY or pass full entity/project/sweep_id."
    exit 1
  fi
  WANDB_PROJECT="${WANDB_PROJECT:-visgen}"
  FULL_SWEEP_ID="${WANDB_ENTITY}/${WANDB_PROJECT}/${SWEEP_ID}"
fi

# Launch one agent per GPU. Slurm assigns GPUs to each srun.
for _ in $(seq 1 4); do
  srun --exclusive --gpus-per-task=1 --gpu-bind=single:1 --cpus-per-task=3 --mem=15G \
    wandb agent "${FULL_SWEEP_ID}" "$@" &
done

wait
