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
| `--ablation-keys phasenet_full_big,phasenet_full_small` | Run only the listed configuration keys (comma-separated) |
| `--skip-baseline` | Skip the baseline (if `phasenet_baseline` is not defined in configs, it will be skipped automatically) |
| `--seed` | Random seed (defaults to the value of `PHASENET_SEED` below) |

Example:

```bash
python phase_run.py --include-ablation --quick --gpu 0
```

## Configuration and Privacy (Environment Variables)

To avoid hard-coding local paths and random seeds in code, set environment variables before running (or export them in your shell / `.env`):

| Variable | Description |
|------|------|
| `PHASENET_SEED` | Global default random seed; defaults to `42` if not set (`phase_core.SEED`) |
| `PHASENET_OUTPUT_DIR` | Root directory for training outputs and metrics; if not set, it prefers the same disk as `CEED_CACHE_DIR`, otherwise uses the current directory |
| `CEED_CACHE_DIR` | Hugging Face dataset cache directory; if empty, uses the system default cache location |
| `CEED_LOCAL_DIR` | Local directory containing CEED `.h5` data; if empty, relies on `datasets` to download/cache-load |
| `H5_THREE_CHANNEL_ROOT` | Root directory of three-channel H5 data (required when `phase_core.py` uses `DATA_SOURCE == "h5_three_channel"`) |

Data source types such as `DATA_ROOT` and `DATA_SOURCE` are still configured in **`phase_core.py`** as needed.
