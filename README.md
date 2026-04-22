# PhaseRiskNet

This repository contains companion code for the paper **PhaseRiskNet: Phase-Aware Risk-Adaptive Seismic Phase Picking with Uncertainty-Guided Selective Prediction**.

## Environment

- Recommended **Python 3.10+**.
- Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install CUDA-enabled PyTorch. Please follow the command on the [PyTorch official site](https://pytorch.org/get-started/locally/) to match your local CUDA version (you can override/extend the `torch` entry in `requirements.txt`).

## Running

From the repository root directory (the folder containing this `README.md`), run:

```bash
python phase_run.py --include-ablation
```

Common arguments:

| Argument | Description |
|------|------|
| `--include-ablation` | Run the ablation configurations defined in `phase_run.py` (e.g., `phasenet_full_big`, `phasenet_full_small`) |
| `--quick` | Quick test run (fewer epochs; relies on the quick-path logic in `phase_core`) |
| `--gpu 0` | Use a specific GPU only (sets `CUDA_VISIBLE_DEVICES`) |
| `--data-source ceed` | Explicitly select the dataset backend (`ceed`, `h5_three_channel`, or `npz`) |
| `--ceed-local-dir /path/to/ceed_local` | Explicit CEED local directory containing `*.h5` files (overrides `CEED_LOCAL_DIR`) |
| `--ceed-cache-dir /path/to/hf_cache` | Explicit Hugging Face cache directory (overrides `CEED_CACHE_DIR`) |
| `--h5-root /path/to/h5_root` | Explicit three-channel H5 root directory (overrides `H5_THREE_CHANNEL_ROOT`) |
| `--ablation-keys phasenet_full_big,phasenet_full_small` | Run only the listed configuration keys (comma-separated) |
| `--skip-baseline` | Skip the baseline (if `phasenet_baseline` is not defined in configs, it will be skipped automatically) |
| `--seed` | Random seed (defaults to the value of `PHASENET_SEED` below) |

Example:

```bash
python phase_run.py --include-ablation --quick --gpu 0
```

Example (CEED from a local directory, with an explicit Hugging Face cache path):

```bash
python phase_run.py --include-ablation --gpu 0 \
  --data-source ceed \
  --ceed-local-dir "/data/ceed_local" \
  --ceed-cache-dir "/data/hf_cache"
```

Example (three-channel H5 backend):

```bash
python phase_run.py --include-ablation --gpu 0 \
  --data-source h5_three_channel \
  --h5-root "/data/diting_h5_root"
```

Example (`nohup` on a server):

```bash
nohup python phase_run.py --include-ablation --gpu 1 \
  --data-source ceed \
  --ceed-local-dir "/data/ceed_local" \
  --ceed-cache-dir "/data/hf_cache" \
  > run.log 2>&1 &
```

## Data

- **CEED**: loaded and (if needed) downloaded via the Hugging Face `datasets` script `CEED.py` in this repository. When `DATA_SOURCE == "ceed"` (in `phase_core.py`), CEED will be accessed through this script, and Hugging Face will handle caching via `CEED_CACHE_DIR` / default cache paths.
- **Diting / three-channel H5**: the Diting three-channel H5 dataset should be downloaded from its public release link (follow the official Diting dataset instructions), then placed under the directory pointed to by `H5_THREE_CHANNEL_ROOT`. When `DATA_SOURCE == "h5_three_channel"`, training and evaluation will use this local H5 data.

## Configuration and Privacy (Environment Variables)

To avoid hard-coding local paths and random seeds in code, set environment variables before running (or export them in your shell / `.env`):

| Variable | Description |
|------|------|
| `PHASENET_SEED` | Global default random seed; defaults to `42` if not set (`phase_core.SEED`) |
| `PHASENET_OUTPUT_DIR` | Root directory for training outputs and metrics; if not set, it prefers the same disk as `CEED_CACHE_DIR`, otherwise uses the current directory |
| `CEED_CACHE_DIR` | Hugging Face dataset cache directory; if empty, uses the system default cache location |
| `CEED_LOCAL_DIR` | Local directory containing CEED `.h5` data; if empty, relies on `datasets` to download/cache-load |
| `H5_THREE_CHANNEL_ROOT` | Root directory of three-channel H5 data (required when `phase_core.py` uses `DATA_SOURCE == "h5_three_channel"`) |

Notes:

- If you pass `--data-source/--ceed-local-dir/--ceed-cache-dir/--h5-root`, these CLI flags take priority over environment variables.
- If you do not pass CLI flags, the project falls back to environment variables and the defaults in `phase_core.py`.
