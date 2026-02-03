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
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <sweep_id> [extra wandb agent args...]"
  echo "Example: $0 entity/project/abc123"
  exit 1
fi

SWEEP_ID="$1"
shift

# Activa tu env si aplica
# source activate tu_env

cd /workspace1/asoto/araymond/scalable-compositional-generalization

# Launch one agent per GPU. Slurm assigns GPUs to each srun.
for _ in $(seq 1 4); do
  srun --exclusive --gres=gpu:1 --cpus-per-task=8 --mem=20G \
    wandb agent "${SWEEP_ID}" "$@" &
done

wait
