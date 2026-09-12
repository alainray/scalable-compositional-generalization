#!/bin/bash

# Runner de las filas que faltan del grid de p-score:
#   split                        -> AIN sin LATTICE  (baseline de lattice_ain_alg_l_1.5)
#   split_4                      -> AIN con 4 bloques exclusivos (falta mpi3d)
#   lattice_resnet18_alg_l_1.5   -> ResNet18 + LATTICE, lambda=1.5
#
# Deja registradas las metricas de representacion en val_4cases y testing. El
# trainer guarda en best_ams el logams de la epoca que gano por selection_metric,
# asi que con every_n_epochs=1 (el default de base.yml) los *_pscore_mean que
# quedan en results.json son los del model_best: mismo protocolo que
# recompute_pscore.py, y por lo tanto comparables con las filas ya medidas.

dataset=$1
experiment=$2
model=$3
seeds=${4:-"1,2,3,4,5"}
flavor=${5:-non_iid}          # non_iid | plain
nw_train=${6:-0}

if [ -z "$dataset" ] || [ -z "$experiment" ] || [ -z "$model" ]; then
    echo "Usage: $0 <dataset> <experiment> <model> [seeds] [non_iid|plain] [nw_train]"
    exit 1
fi

IFS=', ' read -ra SEEDS <<< "$seeds"

if [ "$dataset" = "dsprites" ]; then
    C=(1); D=("[2,3,14,14]"); split_attributes="scale_shape_x-position_y-position"
elif [ "$dataset" = "iraven" ]; then
    C=(1); D=("[6,3,3]"); split_attributes="size_type_color"
elif [ "$dataset" = "cars3d" ]; then
    C=(1); D=("[15,2,113]"); split_attributes="elevation_type_orientation"
elif [ "$dataset" = "shapes3d" ]; then
    C=(1); D=("[7,7,7,6,3]"); split_attributes="wall_floor_object_scale_shape"
elif [ "$dataset" = "clevr" ]; then
    C=(1); D=("[2,2,1,7]"); split_attributes="shape_size_material_color"
elif [ "$dataset" = "mpi3d" ]; then
    C=(1); D=("[5,4,2,2,34,34]"); split_attributes="color_shape_height_bgcolor_x-axis_y-axis"
else
    echo "Unknown dataset: $dataset"; exit 1
fi

split=general_composition
if [ "$flavor" = "plain" ]; then
    data_cfg="configs/datasets/${dataset}.yml"
else
    data_cfg="configs/datasets/${dataset}_non_iid.yml"
fi
model_cfg="configs/models/${model}.yml"

for c in "${C[@]}"; do
    for seed in "${SEEDS[@]}"; do
        difficulty=${D[0]}
        python main.py --experiment-cfg "configs/experiments/${experiment}.yml" \
        --data-cfg "$data_cfg" --model-cfg "$model_cfg" \
        data.training.targets=$split_attributes data.training.split_attributes=$split_attributes \
        data.training.split=$split data.testing.split=$split \
        data.training.c=$c data.testing.c=$c \
        data.training.attr_difficulty=$difficulty data.testing.attr_difficulty=$difficulty \
        training.representation_metrics.enabled=True \
        "training.representation_metrics.split=[val_4cases,testing]" \
        training.representation_metrics.every_n_epochs=1 \
        training.representation_metrics.max_samples=5000 \
        --seed=$seed data.training.num_workers=$nw_train data.testing.num_workers=4 logger.name=wandb \
        path.base="${OUT_BASE:-$HOME/storage/investigacion/licg/scalable-compositional-generalization/out}"
    done
done
