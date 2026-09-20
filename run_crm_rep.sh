#!/bin/bash
#SBATCH --job-name=run_comp
#SBATCH --output=exp_logs/%x_%j.out
#SBATCH --error=exp_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=forgetting_pixels_learnin_vidt5
#SBATCH --partition=peteroa-default
#SBATCH --nodelist=peteroa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --comment="Experimento: CRM+ED"
#SBATCH --qos=regular
# Activar entorno si es necesario
# El /tmp del nodo se llena con archivos de otros usuarios y los workers
# del DataLoader mueren con OSError 28; los temporales van al scratch.
export TMPDIR=/workspace1/araymond/tmp
mkdir -p "$TMPDIR"
source ~/storage/peteroa/bin/activate

cd ~/investigacion/scalable-compositional-generalization

./crm_runner.sh $ds crm_rep ${model:-all} $seeds

wait

echo "Finished with job $SLURM_JOBID (seed=$SEED)"
