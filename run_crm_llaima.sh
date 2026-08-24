#!/bin/bash
#SBATCH --job-name=run_crm
#SBATCH --output=exp_logs/%x_%j.out
#SBATCH --error=exp_logs/%x_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --account=araymond
#SBATCH --partition=ialab
#SBATCH --nodelist=llaima
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

source ~/storage/torch/bin/activate
cd ~/investigacion/scalable-compositional-generalization

# En llaima la salida va al scratch local, no al de peteroa.
export OUT_BASE=/workspace1/asoto/araymond/scalable-compositional-generalization/out

./crm_runner.sh $ds crm ${model:-crm_split_resnet_mixer_l15} $seeds
wait

echo "Finished with job $SLURM_JOBID"
