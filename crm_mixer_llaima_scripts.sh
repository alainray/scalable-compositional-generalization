#!/bin/bash
# CRM + mixer algebraico (loss_weight 1.5) en llaima. Una semilla por job.
# Orden: modelo, dataset, semilla. Correr desde ~/investigacion/scalable-compositional-generalization

# ===================== crm_resnet18_mixer_l15 =====================

sbatch --export=ds=cars3d,model=crm_resnet18_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=cars3d,model=crm_resnet18_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=cars3d,model=crm_resnet18_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=cars3d,model=crm_resnet18_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=cars3d,model=crm_resnet18_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=iraven,model=crm_resnet18_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=iraven,model=crm_resnet18_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=iraven,model=crm_resnet18_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=iraven,model=crm_resnet18_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=iraven,model=crm_resnet18_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=dsprites,model=crm_resnet18_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=dsprites,model=crm_resnet18_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=dsprites,model=crm_resnet18_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=dsprites,model=crm_resnet18_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=dsprites,model=crm_resnet18_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=shapes3d,model=crm_resnet18_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=shapes3d,model=crm_resnet18_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=shapes3d,model=crm_resnet18_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=shapes3d,model=crm_resnet18_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=shapes3d,model=crm_resnet18_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=clevr,model=crm_resnet18_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=clevr,model=crm_resnet18_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=clevr,model=crm_resnet18_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=clevr,model=crm_resnet18_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=clevr,model=crm_resnet18_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=mpi3d,model=crm_resnet18_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=mpi3d,model=crm_resnet18_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=mpi3d,model=crm_resnet18_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=mpi3d,model=crm_resnet18_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=mpi3d,model=crm_resnet18_mixer_l15,seeds=5 run_crm_llaima.sh

# ===================== crm_split_resnet_mixer_l15 =====================

sbatch --export=ds=cars3d,model=crm_split_resnet_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=cars3d,model=crm_split_resnet_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=cars3d,model=crm_split_resnet_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=cars3d,model=crm_split_resnet_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=cars3d,model=crm_split_resnet_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=iraven,model=crm_split_resnet_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=iraven,model=crm_split_resnet_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=iraven,model=crm_split_resnet_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=iraven,model=crm_split_resnet_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=iraven,model=crm_split_resnet_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=dsprites,model=crm_split_resnet_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=dsprites,model=crm_split_resnet_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=dsprites,model=crm_split_resnet_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=dsprites,model=crm_split_resnet_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=dsprites,model=crm_split_resnet_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=shapes3d,model=crm_split_resnet_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=shapes3d,model=crm_split_resnet_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=shapes3d,model=crm_split_resnet_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=shapes3d,model=crm_split_resnet_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=shapes3d,model=crm_split_resnet_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=clevr,model=crm_split_resnet_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=clevr,model=crm_split_resnet_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=clevr,model=crm_split_resnet_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=clevr,model=crm_split_resnet_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=clevr,model=crm_split_resnet_mixer_l15,seeds=5 run_crm_llaima.sh

sbatch --export=ds=mpi3d,model=crm_split_resnet_mixer_l15,seeds=1 run_crm_llaima.sh
sbatch --export=ds=mpi3d,model=crm_split_resnet_mixer_l15,seeds=2 run_crm_llaima.sh
sbatch --export=ds=mpi3d,model=crm_split_resnet_mixer_l15,seeds=3 run_crm_llaima.sh
sbatch --export=ds=mpi3d,model=crm_split_resnet_mixer_l15,seeds=4 run_crm_llaima.sh
sbatch --export=ds=mpi3d,model=crm_split_resnet_mixer_l15,seeds=5 run_crm_llaima.sh
