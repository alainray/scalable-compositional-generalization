#!/bin/bash
#SBATCH --job-name=run_comp
#SBATCH --output=exp_logs/%x_%j.out
#SBATCH --error=exp_logs/%x_%j.err
#SBATCH --time=01-00:00:00
#SBATCH --account=forgetting_pixels_learnin_vidt5
#SBATCH --partition=peteroa-default
#SBATCH --nodelist=peteroa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
# Activar entorno si es necesario
source ~/storage/peteroa/bin/activate

cd ~/investigacion/scalable-compositional-generalization

./crm_runner.sh $ds crm $model

wait

echo "Finished with job $SLURM_JOBID (seed=$SEED)"
