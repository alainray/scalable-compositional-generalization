#!/bin/bash
#SBATCH --job-name=run_comp
#SBATCH --output=exp_logs/%x_%j.out
#SBATCH --error=exp_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=araymond
#SBATCH --partition=ialab
#SBATCH --nodelist=llaima
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --comment="Experimento: CRM+ED"
#SBATCH --qos=regular

source ~/storage/torch/bin/activate
cd ~/investigacion/scalable-compositional-generalization

# En llaima la salida va al scratch local, no al de peteroa.
export OUT_BASE=/workspace1/asoto/araymond/scalable-compositional-generalization/out

# --time se sobreescribe por dataset desde la linea de sbatch.
eval "./pending_runner.sh $ds metrics $model $seeds ${flavor:-non_iid} ${nw:-0}"
wait

echo "Finished with job $SLURM_JOBID (ds=$ds model=$model seed=$seeds)"
