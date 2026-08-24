#!/bin/bash

# Compositional Risk Minimization (CRM) over the orthotopic splits.
# Mirrors compositional_orth.sh so CRM runs are directly comparable to the
# baselines; the only differences are the model list and that the trainer is
# forced to `crm` (a config with `trainer: base` would silently run the plain
# trainer with no group head, no B_hat and no post-hoc step).
#
# Usage: ./crm_runner.sh <dataset> [experiment] [model] [seeds]
#   dataset:    iraven | cars3d | dsprites | shapes3d | mpi3d | clevr
#   experiment: config name under configs/experiments (default: crm)
#   model:      config name under configs/models (default: all CRM models)
#   seeds:      comma-separated list of seeds to run (default: 5)
#               e.g. ./crm_runner.sh dsprites crm all 1,2,3

dataset=$1
experiment=${2:-"crm"}
model=${3:-"all"}
seeds=${4:-"5"}

IFS=',' read -ra SEEDS <<< "$seeds"
split=general_composition

# DSPRITES
if [ "$dataset" = "dsprites" ]; then
    C=(1)
    D=("[2,3,14,14]")
    split_attributes="scale_shape_x-position_y-position"

# IRAVEN
elif [ "$dataset" = "iraven" ]; then
    C=(1)
    D=("[6,3,3]")
    split_attributes="size_type_color"

# CARS3D
elif [ "$dataset" = "cars3d" ]; then
    C=(1)
    D=("[15,2,113]")
    split_attributes="elevation_type_orientation"

# SHAPES3D
elif [ "$dataset" = "shapes3d" ]; then
    C=(1)
    D=("[7,7,7,6,3]")
    split_attributes="wall_floor_object_scale_shape"

# CLEVR
elif [ "$dataset" = "clevr" ]; then
    C=(1)
    D=("[2,2,1,7]")
    split_attributes="shape_size_material_color"

# MPI3D
elif [ "$dataset" = "mpi3d" ]; then
    C=(1)
    D=("[5,4,2,2,34,34]")
    split_attributes="color_shape_height_bgcolor_x-axis_y-axis"

else
    echo "Unknown dataset: $dataset"
    exit 1
fi

crm_models=("crm_resnet18" "crm_split_resnet" "crm_split_resnet_mixer")
if [ "$model" = "all" ]; then
    all_models=("${crm_models[@]}")
else
    all_models=("${model}")
fi

for c in "${C[@]}"; do
    for model in "${all_models[@]}"; do
        # the analogical term needs 4-view batches, which only the non_iid
        # dataset configs produce
        case "$model" in
            *_mixer|*_mixer_*) data_cfg="configs/datasets/${dataset}_non_iid.yml" ;;
            *)       data_cfg="configs/datasets/${dataset}.yml" ;;
        esac
        for seed in "${SEEDS[@]}"; do
            difficulty=${D[0]}
            python main.py --experiment-cfg configs/experiments/${experiment}.yml \
            --data-cfg "$data_cfg" --model-cfg configs/models/${model}.yml \
            data.training.targets=$split_attributes data.training.split_attributes=$split_attributes \
            data.training.split=$split --seed=$seed  data.training.c=$c data.testing.c=$c \
            data.training.attr_difficulty=$difficulty data.testing.attr_difficulty=$difficulty data.training.num_workers=0 data.testing.num_workers=0 \
            training.trainer=crm \
            logger.name=wandb path.base="${OUT_BASE:-$HOME/storage/investigacion/licg/scalable-compositional-generalization/out}"
        done
    done
done
