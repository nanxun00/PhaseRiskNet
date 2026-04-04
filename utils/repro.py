"""
Reproducibility helpers for PyTorch experiments.

Usage:
    from utils.repro import seed_everything, seed_worker, torch_generator
    seed_everything(42)
    g = torch_generator(42)
    DataLoader(..., worker_init_fn=seed_worker, generator=g)
"""
from __future__ import annotations
import os
import random
from typing import Optional

def seed_everything(seed: int=42, deterministic: bool=True, benchmark: Optional[bool]=None) -> None:
    """Seed Python, NumPy, and PyTorch. Optionally enforce deterministic behavior.

    - Sets `PYTHONHASHSEED` for stable hashing.
    - Seeds `random`, `numpy`, `torch`, and CUDA.
    - If `deterministic` is True, configures cuDNN and (best-effort) PyTorch deterministic algorithms.
    - If `benchmark` is None and deterministic is True, disables cuDNN benchmark to avoid nondeterministic kernels.
    """
    os.environ['PYTHONHASHSEED'] = str(int(seed))
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                import torch.backends.cudnn as cudnn
                if deterministic:
                    cudnn.deterministic = True
                    cudnn.benchmark = False if benchmark is None else bool(benchmark)
                    cudnn.allow_tf32 = False
                elif benchmark is not None:
                    cudnn.benchmark = bool(benchmark)
                else:
                    cudnn.allow_tf32 = False
            except Exception:
                pass
            if deterministic:
                os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
    random.seed(seed)

def seed_worker(worker_id: int) -> None:
    """Per-worker init function for DataLoader to set NumPy and random seeds deterministically.

    Uses PyTorch's initial seed to derive a 32-bit worker seed.
    """
    try:
        import torch
        worker_seed = torch.initial_seed() % 2 ** 32
    except Exception:
        worker_seed = int(os.environ.get('PYTHONHASHSEED', '0')) % 2 ** 32
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except Exception:
        pass
    random.seed(worker_seed)

def torch_generator(seed: int):
    """Return a CPU torch.Generator seeded with `seed`.

    This can be passed to DataLoader(generator=...).
    """
    import torch
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
