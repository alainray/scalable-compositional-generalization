#!/bin/bash
#SBATCH --job-name=lambda
#SBATCH --output=exp_logs/%x_%j.out
#SBATCH --error=exp_logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --account=defaultacc
#SBATCH --partition=peteroa-default
#SBATCH --nodelist=peteroa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
# Activar entorno si es necesario
source ~/storage/peteroa/bin/activate
cd ~/investigacion/scalable-compositional-generalization

WANDB_DEBUG=1,WANDB_SILENT=false ./ain_alg_lambda_runner.sh $ds metrics $lambda
wait

echo "Finished with job $SLURM_JOBID (seed=$SEED)"
