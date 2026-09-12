#!/bin/bash
#SBATCH --job-name=run_comp
#SBATCH --output=exp_logs/%x_%j.out
#SBATCH --error=exp_logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=forgetting_pixels_learnin_vidt5
#SBATCH --partition=peteroa-default
#SBATCH --nodelist=peteroa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --comment="Experimento: CRM+ED"
#SBATCH --qos=regular
# El /tmp del nodo se llena con archivos de otros usuarios y los workers del
# DataLoader mueren con OSError 28; los temporales van al scratch.
export TMPDIR=/workspace1/araymond/tmp
mkdir -p "$TMPDIR"
source ~/storage/peteroa/bin/activate
cd ~/investigacion/scalable-compositional-generalization

# --time se sobreescribe por dataset desde la linea de sbatch (backfill premia
# los limites cortos), igual que ds/model/seeds/flavor/nw via --export.
eval "WANDB_DEBUG=1,WANDB_SILENT=false ./pending_runner.sh $ds metrics $model $seeds ${flavor:-non_iid} ${nw:-0}"
wait

echo "Finished with job $SLURM_JOBID (ds=$ds model=$model seed=$seeds)"
