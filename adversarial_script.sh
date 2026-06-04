sbatch --export=ds=cars3d,sampling=adversarial run_ain_algebraic_adv.sh
sbatch --export=ds=cars3d,sampling=unpredictable_target_1 run_ain_algebraic_adv.sh
sbatch --export=ds=cars3d,sampling=unpredictable_target_2 run_ain_algebraic_adv.sh

