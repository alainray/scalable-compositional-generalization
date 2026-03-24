#!/bin/bash
#SBATCH --job-name=run_comp
#SBATCH --output=exp_logs/%x_%j.out
#SBATCH --error=exp_logs/%x_%j.err
#SBATCH --time=02-00:00:00
#SBATCH --account=forgetting_pixels_learnin_vidt5
#SBATCH --partition=debug
#SBATCH --nodelist=peteroa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
# Activar entorno si es necesario
source ~/storage/torch/bin/activate

cd ~/investigacion/scalable-compositional-generalization

./compositional_orth.sh $ds metrics resnet18

wait

echo "Finished with job $SLURM_JOBID (seed=$SEED)"
