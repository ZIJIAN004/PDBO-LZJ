@echo off
cd /d "%~dp0"
D:\apps\Python313\python.exe scripts\compare_spectral_initializations.py ^
  --gset_ids 8 9 10 ^
  --seeds 0 1 2 ^
  --modes spectral_subspace_random ^
  --batch 100 ^
  --max_iters 5000 ^
  --check_every 10 ^
  --optimizer rmsprop ^
  --init_radius 0.5 ^
  --dual_init_mode curvature ^
  --hessian_level 0 ^
  --subspace_dim 16 ^
  --subspace_power_min 0.5 ^
  --subspace_power_max 1.5 ^
  --out spectral_subspace_random_boundary_g8_g10_seeds0_2_patch.csv ^
  > spectral_subspace_random_boundary_g8_g10_seeds0_2_patch.out.log ^
  2> spectral_subspace_random_boundary_g8_g10_seeds0_2_patch.err.log
