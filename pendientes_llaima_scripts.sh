#!/bin/bash
# ==========================================================================
# LLAIMA: los 4 datasets que faltan de las dos filas nuevas del grid.
# cars3d, dsprites, shapes3d, iraven  x  {AIN sin LATTICE, ResNet18+LATTICE}
# = 40 trabajos, una semilla por trabajo.
#
# Correr desde ~/investigacion/scalable-compositional-generalization en llaima
# (ssh kraken -> ssh llaima), despues de hacer git pull: necesita
# pending_runner.sh, run_pending_llaima.sh y los dos configs nuevos
# (lattice_ain_alg_l_0.yml, lattice_resnet18_alg_l_1.5.yml).
#
# La salida va a OUT_BASE=/workspace1/asoto/araymond/.../out, igual que los CRM.
#
# --time: anclado en corridas medidas EN LLAIMA (A40), no en peteroa:
#   lattice_ain_alg_l_1        (rep=0, ms=20000)  -> anclaje de AIN lambda=0
#   resnet18_algebraic_non_iid (rep=1, ms=20000)  -> anclaje de ResNet18+LATTICE
# llaima tiene A40 y peteroa H100 80GB: el H100 corre 1.3-2.0x mas rapido
# (medido comparando crm_resnet18_mixer_l15 en peteroa contra
# resnet18_algebraic_non_iid en llaima, en clevr y mpi3d). Por eso NO sirve
# anclar tiempos de llaima en corridas de peteroa.
#
#   dataset    epocas   AIN l=0        R18+LATTICE     --time
#   cars3d     500      1.39 h (rep=0) 2.12 h (rep=1)  6 h
#   iraven      50      1.58 h (rep=0) 1.99 h (rep=1)  4 h
#   shapes3d    50      3.91 h (rep=0) 6.13 h (rep=1)  8 h
#   dsprites    50      4.70 h (rep=0) 5.97 h (rep=1)  8 h
#
# El anclaje de AIN tiene rep=0, asi que hay que sumarle las metricas: ~25 s por
# epoca en peteroa, del orden de 40-50 s en el A40. Con 50 epocas es ~0.6 h y se
# pierde en el margen; en cars3d son 500 epocas, o sea ~2.5-3 h sobre 1.39 h de
# entrenamiento. Ese es el unico caso donde las metricas dominan el costo, y es
# la razon de que cars3d (el dataset mas chico) no tenga el --time mas corto.
# ==========================================================================

# ============ AIN sin LATTICE (lattice_ain_alg_l_0, lambda=0) =============

for s in 1 2 3 4 5; do
  sbatch --time=06:00:00 --export=ALL,ds=cars3d,model=lattice_ain_alg_l_0,seeds=$s,flavor=non_iid,nw=0 run_pending_llaima.sh
done

for s in 1 2 3 4 5; do
  sbatch --time=04:00:00 --export=ALL,ds=iraven,model=lattice_ain_alg_l_0,seeds=$s,flavor=non_iid,nw=0 run_pending_llaima.sh
done

for s in 1 2 3 4 5; do
  sbatch --time=08:00:00 --export=ALL,ds=dsprites,model=lattice_ain_alg_l_0,seeds=$s,flavor=non_iid,nw=0 run_pending_llaima.sh
done

for s in 1 2 3 4 5; do
  sbatch --time=08:00:00 --export=ALL,ds=shapes3d,model=lattice_ain_alg_l_0,seeds=$s,flavor=non_iid,nw=0 run_pending_llaima.sh
done

# ====== ResNet-18 + LATTICE (lattice_resnet18_alg_l_1.5, lambda=1.5) ======

for s in 1 2 3 4 5; do
  sbatch --time=06:00:00 --export=ALL,ds=cars3d,model=lattice_resnet18_alg_l_1.5,seeds=$s,flavor=non_iid,nw=0 run_pending_llaima.sh
done

for s in 1 2 3 4 5; do
  sbatch --time=04:00:00 --export=ALL,ds=iraven,model=lattice_resnet18_alg_l_1.5,seeds=$s,flavor=non_iid,nw=0 run_pending_llaima.sh
done

for s in 1 2 3 4 5; do
  sbatch --time=08:00:00 --export=ALL,ds=dsprites,model=lattice_resnet18_alg_l_1.5,seeds=$s,flavor=non_iid,nw=0 run_pending_llaima.sh
done

for s in 1 2 3 4 5; do
  sbatch --time=08:00:00 --export=ALL,ds=shapes3d,model=lattice_resnet18_alg_l_1.5,seeds=$s,flavor=non_iid,nw=0 run_pending_llaima.sh
done
