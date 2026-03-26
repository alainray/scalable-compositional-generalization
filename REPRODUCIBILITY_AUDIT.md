# Reproducibility audit (seed-level)

This note summarizes whether training runs are expected to be repeatable when using the same `--seed`.

## Verdict

- **Not fully deterministic today**.
- There are at least two concrete variability sources that can change results across runs with the same seed:
  1. CUDA backend config sets `cudnn.benchmark=True` together with `cudnn.deterministic=True`.
  2. `NonIIDWrapper` uses its own NumPy RNG with `seed=None` unless explicitly configured, so it can ignore the global experiment seed.

## Where seeds are fixed correctly

- `main.py` calls `fix_random(cfg.seed)` before dataloaders and model creation.
- `fix_random` seeds Python `random`, NumPy, and Torch CPU/CUDA.
- Dataset split helpers use seeded generators in some places (e.g., `visgen/datasets/splits.py`).

## Main nondeterminism risks

1. **cuDNN benchmark enabled**
   - In `visgen/utils/general/random.py`, `torch.backends.cudnn.benchmark=True` is enabled.
   - This can introduce run-to-run variability on GPU even with deterministic mode enabled.

2. **Non-IID wrapper RNG not tied to experiment seed by default**
   - `visgen/datasets/non_iid.py` constructs `np.random.default_rng(seed)`.
   - In dataset configs, `non_iid.seed` is usually omitted, so wrapper seed becomes `None`.
   - That makes non-IID sampling vary between runs regardless of `--seed`.

3. **No strict deterministic-algorithms guard**
   - The code does not call `torch.use_deterministic_algorithms(True)`.
   - If any nondeterministic CUDA ops are used by models/ops, results can still drift.

## Practical expectation

- **CPU-only, IID setup, same software stack:** often repeatable.
- **GPU training:** can vary due to cuDNN benchmark and potentially nondeterministic kernels.
- **Any run using `non_iid` wrapper without explicit `non_iid.seed`:** expected to vary across reruns.

## Hardening checklist

- Set `torch.backends.cudnn.benchmark = False` when reproducibility is required.
- Add `torch.use_deterministic_algorithms(True)` (with clear fallback/error handling).
- Bind `non_iid.seed` to `${seed}` in dataset configs (or default it from global seed in code).
- Optionally pass an explicit `generator` to `DataLoader` and/or `random_split` wherever randomness is used.
