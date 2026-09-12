# ==========================================================================
# Filas que faltan del grid de p-score. TU decides que se envia.
# Un trabajo por semilla (backfill premia los --time cortos).
# Todos: qos=regular, account=forgetting_pixels_learnin_vidt5,
# partition=peteroa-default, nodelist=peteroa, TMPDIR en scratch,
# y metricas de representacion activadas en val_4cases + testing.
# Limite de la cola: 16 encolados / 4 corriendo -> por eso van en lotes.
# ==========================================================================

# --- LOTE 1 (10 trabajos) -------------------------------------------------
# 1) split_4 en mpi3d: unica celda que faltaba de la fila ED (split_4).
#    data plana + nw=4 + batch 1024, igual que split_1..split_4 ya corridos.
#    Anclaje: los 5 runs de abril tardaron 8.05-8.26 h, pero (a) con
#    representation_metrics APAGADAS y (b) en LLAIMA (A40), no en peteroa.
#    El H100 de peteroa corre 1.3-2.0x mas rapido, asi que 8.08/1.3 = 6.2 h
#    + ~0.5 h de metricas = ~6.7 h. Las 11 h pedidas sobran; quedaron asi
#    porque los jobs ya estaban encolados y resubmitirlos pierde la edad.
for s in 1 2 3 4 5; do
  sbatch --time=11:00:00 --export=ALL,ds=mpi3d,model=split_4,seeds=$s,flavor=plain,nw=4 run_pending.sh
done

# 2) AIN sin LATTICE en mpi3d (50 epocas).
#    Anclaje: lattice_ain_alg_l_1 (misma arch, mismo batch) 8.83-9.32 h, pero
#    medido en llaima (A40). En H100: ~7.0 h + 0.5 de metricas = ~7.5 h.
for s in 1 2 3 4 5; do
  sbatch --time=12:00:00 --export=ALL,ds=mpi3d,model=lattice_ain_alg_l_0,seeds=$s,flavor=non_iid,nw=0 run_pending.sh
done

# --- LOTE 2 (10 trabajos) -------------------------------------------------
# 3) AIN sin LATTICE en clevr (250 epocas).
#    Anclaje: lattice_ain_alg_l_1 clevr 5.19 h, medido EN PETEROA (H100), el
#    unico anclaje nativo que tengo. + ~1.7 h de metricas (250 epocas) = ~6.9 h.
for s in 1 2 3 4 5; do
  sbatch --time=08:00:00 --export=ALL,ds=clevr,model=lattice_ain_alg_l_0,seeds=$s,flavor=non_iid,nw=0 run_pending.sh
done

# 4) ResNet-18 + LATTICE (lambda=1.5) en mpi3d (50 epocas).
#    Anclaje: resnet18_algebraic_non_iid (lambda=1.0, mismo batch, metricas ya
#    encendidas) 9.3-9.8 h EN LLAIMA (A40). Con el ratio medido en vivo sobre
#    el job 7373 (A40 581 s/ep sin metricas -> H100 270 s/ep CON metricas,
#    >=2.15x), en H100 son ~5.3 h. Uso 1.8x para dejar margen.
for s in 1 2 3 4 5; do
  sbatch --time=08:00:00 --export=ALL,ds=mpi3d,model=lattice_resnet18_alg_l_1.5,seeds=$s,flavor=non_iid,nw=0 run_pending.sh
done

# --- LOTE 3 (5 trabajos) --------------------------------------------------
# 5) ResNet-18 + LATTICE (lambda=1.5) en clevr (250 epocas).
#    Anclaje: resnet18_algebraic_non_iid clevr 6.73 h en A40 -> ~3.7 h en H100 (1.8x).
for s in 1 2 3 4 5; do
  sbatch --time=06:00:00 --export=ALL,ds=clevr,model=lattice_resnet18_alg_l_1.5,seeds=$s,flavor=non_iid,nw=0 run_pending.sh
done

# ==========================================================================
# ALTERNATIVA para el AIN, si prefieres el modelo `split` literal.
# Ojo: con data non_iid el batch queda en min(1024,2048)=1024, no en 256,
# asi que NO parea con lattice_ain_alg_l_1.5 (que corrio en 256) y el
# override de batch por CLI lo pisa custom_cfg_conflict_resolution.
# Con data plana si parea con split_1..split_4, pero no con la fila LATTICE.
# ==========================================================================
# for s in 1 2 3 4 5; do
#   sbatch --time=08:00:00 --export=ALL,ds=clevr,model=split,seeds=$s,flavor=plain,nw=4 run_pending.sh
#   sbatch --time=12:00:00 --export=ALL,ds=mpi3d,model=split,seeds=$s,flavor=plain,nw=4 run_pending.sh
# done

# ==========================================================================
# LLAIMA: los otros 4 datasets. Cambia path.base si alla es otro.
# ==========================================================================
# for ds in cars3d dsprites shapes3d iraven; do
#   ./pending_runner.sh $ds metrics lattice_ain_alg_l_0        1,2,3,4,5 non_iid 0
#   ./pending_runner.sh $ds metrics lattice_resnet18_alg_l_1.5 1,2,3,4,5 non_iid 0
# done
