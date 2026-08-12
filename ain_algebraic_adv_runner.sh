#!/bin/bash

# get command line args
dataset=$1
experiment=$2
sampling=$3 # adversarial/unpredictable_target_1/unpredictable_target_2
seeds=${4:-"1,2,3"} # comma- or space-separated list, e.g. 1,2,3 o "4 5"
model="split_resnet_algebraic_non_iid"

if [ -z "$dataset" ] || [ -z "$experiment" ]; then
    echo "Usage: $0 <dataset> <experiment> [sampling] [seeds]"
    echo "  seeds: comma-separated list of seeds to run (default: 1,2,3)"
    echo "         e.g. $0 dsprites adversarial adversarial 4,5"
    exit 1
fi

# set number of experiment repetitions
IFS=', ' read -ra SEEDS <<< "$seeds"

# define dataset-specific split attributes
if [ "$dataset" = "dsprites" ]; then
    C=(1)
    D=("[2,3,14,14]")
    split_attributes="scale_shape_x-position_y-position"
elif [ "$dataset" = "iraven" ]; then
    C=(1)
    D=("[6,3,3]")
    split_attributes="size_type_color"
elif [ "$dataset" = "cars3d" ]; then
    C=(1)
    D=("[15,2,113]")
    split_attributes="elevation_type_orientation"
elif [ "$dataset" = "shapes3d" ]; then
    C=(1)
    D=("[7,7,7,6,3]")
    split_attributes="wall_floor_object_scale_shape"
elif [ "$dataset" = "clevr" ]; then
    C=(1)
    D=("[2,2,1,7]")
    split_attributes="shape_size_material_color"
elif [ "$dataset" = "mpi3d" ]; then
    C=(1)
    D=("[5,4,2,2,34,34]")
    split_attributes="color_shape_height_bgcolor_x-axis_y-axis"
else
    echo "Unknown dataset: $dataset"
    exit 1
fi

split=general_composition
data_cfg="configs/datasets/${dataset}_${sampling}.yml"
model_cfg="configs/models/${model}.yml"

for c in "${C[@]}"; do
    for seed in "${SEEDS[@]}"; do
        difficulty=${D[0]}
        python main.py --experiment-cfg "configs/experiments/${experiment}.yml" \
        --data-cfg "$data_cfg" --model-cfg "$model_cfg"  --model-dir-suffix "${sampling}" \
        data.training.targets=$split_attributes data.training.split_attributes=$split_attributes \
        data.training.split=$split data.testing.split=$split \
        data.training.c=$c data.testing.c=$c \
        data.training.attr_difficulty=$difficulty data.testing.attr_difficulty=$difficulty \
        --seed=$seed data.training.num_workers=0 data.testing.num_workers=4 logger.name=wandb \
	
    done
done
