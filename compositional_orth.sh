#!/bin/bash

# get command line args
dataset=$1
experiment=$2
model=${3:-"all"} # default to "all" if not provided

# set number of experiment repetitions
SEEDS=(1 2 3)
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

fi

# define list of models to test
pretrained=("resnet101_pretrained" "resnet152_pretrained" "densenet121_pretrained")
resnet=("resnet18")
densenet=("densenet121" "densenet201" "densenet161")
wideresnet=("wideresnet")
convnext=("convnext_tiny" "convnext_small" "convnext_base")
vit=("swin_tiny" "swin_base")
mlp=("mlp")
ed=("ed" "split")
if [ "$model" = "all" ]; then
    all_models=("${resnet[@]}" "${densenet[@]}" "${pretrained[@]}" "${vit[@]}" "${convnext[@]}" "${mlp[@]}" "${ed[@]}" "${wideresnet[@]}")
else
    all_models=("${model}")
fi

for c in "${C[@]}"; do
    for model in "${all_models[@]}"; do
        for seed in "${SEEDS[@]}"; do
            difficulty=${D[0]}
            python main.py --experiment-cfg configs/experiments/${experiment}.yml \
            --data-cfg configs/datasets/${dataset}.yml --model-cfg configs/models/${model}.yml \
            data.training.targets=$split_attributes data.training.split_attributes=$split_attributes \
            data.training.split=$split --seed=$seed  data.training.c=$c data.testing.c=$c logger.name=base \
            data.training.attr_difficulty=$difficulty data.testing.attr_difficulty=$difficulty data.training.num_workers=0 data.testing.num_workers=0
        done
    done
done
