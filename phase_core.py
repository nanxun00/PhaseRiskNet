from __future__ import annotations
import os
import json
import inspect
from datetime import datetime
CEED_CACHE_DIR = os.environ.get('CEED_CACHE_DIR', '').strip()
if CEED_CACHE_DIR:
    os.environ['HF_DATASETS_CACHE'] = CEED_CACHE_DIR
    os.environ['HF_HUB_CACHE'] = os.path.join(CEED_CACHE_DIR, 'hub')
    os.environ['HF_HOME'] = CEED_CACHE_DIR
import csv
import math
import random
from typing import Dict, Any
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils.repro import seed_everything, seed_worker, torch_generator
from phase_model import PhaseNetUNet
from data import WaveformDataset
from ceed_data import CEEDDataset
from three_channel_h5_dataset import ThreeChannelH5Dataset
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None
from single_ablation_visualization import plot_losses, estimate_snr, plot_one_sample_visual, plot_time_residual_distribution, plot_representative_waveforms_grid, plot_pca_visualization, plot_pr_curve, plot_snr_stratified, plot_max_prob_histogram, plot_uncertainty_overview_grid
DATA_SOURCE = 'ceed'
DATA_ROOT = 'dataset'
TRAIN_DIR = os.path.join(DATA_ROOT, 'waveform_train')
TRAIN_CSV = os.path.join(DATA_ROOT, 'waveform_train_split.csv')
VALID_CSV = os.path.join(DATA_ROOT, 'waveform_valid_split.csv')
CEED_LOCAL_DIR = os.environ.get('CEED_LOCAL_DIR', '').strip() or None
CEED_DATASET_NAME = 'CEED.py'
CEED_TRAIN_SPLIT = 'train'
CEED_VALID_SPLIT = 'train'
CEED_TEST_SPLIT = 'test'
CEED_LIMIT_TRAIN = 9000
CEED_VAL_RATIO = 0.2
CEED_LIMIT_TEST = 1000
CEED_WAVEFORM_KEY = None
CEED_P_KEY = None
CEED_S_KEY = None
H5_THREE_CHANNEL_ROOT = os.environ.get('H5_THREE_CHANNEL_ROOT', '').strip()
H5_TEST_RATIO = 0.08
H5_TRAIN_VAL_RATIO = 0.78
H5_LIMIT = 10000
H5_LIMIT_TRAIN = None
H5_LIMIT_VAL = None
H5_LIMIT_TEST = None
H5_ARRIVAL_RELATIVE_TO_SEGMENT = True
H5_FILTER_NATURAL_ONLY = False
H5_ALLOW_TYPES = None
H5_STRICT_CHECK = False
OUTPUT_BASE_DIR = os.environ.get('PHASENET_OUTPUT_DIR', CEED_CACHE_DIR if CEED_CACHE_DIR else '.')
OUT_ROOT = os.path.join(OUTPUT_BASE_DIR, 'PhaseNet', 'ablation_unet')
os.makedirs(OUT_ROOT, exist_ok=True)
METRICS_CSV = os.path.join(OUT_ROOT, 'metrics.csv')
METRICS_HEADER = ['name', 'best_val', 'train_last', 'valid_last', 'use_cbam', 'kernels', 'thr_p', 'thr_s', 'time_acc', 'mcc', 'p_prec', 'p_rec', 'p_f1', 's_prec', 's_rec', 's_f1', 'p_res_mean_sec', 'p_res_std_sec', 'p_res_mae_sec', 's_res_mean_sec', 's_res_std_sec', 's_res_mae_sec']
MODEL_CLASS_MAP: Dict[str, type[nn.Module]] = {'PhaseNetUNet': PhaseNetUNet}
EPOCHS = 120
BATCH_SIZE = 32
LR = 0.005
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 0
CROP_LEN = 3000
LABEL_WIDTH = 51
LABEL_SIGMA_SEC = 0.1
SAMPLE_RATE = 100.0
N_VIS = 4
SEED = int(os.environ.get('PHASENET_SEED', '42'))
_dataset_split_cache = {}
_base_dataset_cache = {}
_ceed_test_ds_cache = None

def _get_split_cache_key(use_waveform_aug: bool, aug_params: tuple) -> str:
    if DATA_SOURCE == 'h5_three_channel':
        return f'{DATA_SOURCE}_{H5_THREE_CHANNEL_ROOT}_{H5_TEST_RATIO}_{H5_TRAIN_VAL_RATIO}_{H5_LIMIT}_{SEED}'
    return f'{DATA_SOURCE}_{CEED_LIMIT_TRAIN}_{CEED_VAL_RATIO}_{CEED_LIMIT_TEST}_{SEED}_{use_waveform_aug}_{aug_params}'
RUN_TEST_EVAL = True
DYN_THRESH_GRID = [x / 100.0 for x in range(10, 96, 5)]
DYN_TOL_SAMPLES = 10
BEST_MODEL_BY_F1 = True
BEST_METRIC_WEIGHTS = (0.5, 0.5)
H5_EARLY_STOP_PATIENCE = 20
SCHEDULER = 'plateau'
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 8
PLATEAU_MIN_LR = 1e-06
PLATEAU_THRESHOLD = 0.0001

def _write_comparison_format(case_dir: str) -> None:
    fmt = {'description': '将对比方法的数据按下列格式写入同目录，运行 plot_paper_figures.py --compare 可绘制多方法对比图。', 'pr_snr_data': {'filename_pattern': 'pr_snr_data_<method>.json（如 pr_snr_data_ar.json）', 'keys': ['has_p', 'has_s', 'max_prob_p', 'max_prob_s', 'p_ok', 's_ok', 'snr'], 'key_descriptions': {'has_p': 'bool list, 每样本是否有 P 波真值', 'has_s': 'bool list, 每样本是否有 S 波真值', 'max_prob_p': 'float list, 每样本 P 通道最大概率', 'max_prob_s': 'float list, 每样本 S 通道最大概率', 'p_ok': 'bool list, 每样本 P 拾取是否在容差内正确', 's_ok': 'bool list, 每样本 S 拾取是否在容差内正确', 'snr': 'float list, 每样本 SNR (dB)'}}, 'time_residuals': {'filename_pattern': 'time_residuals_<method>.json（如 time_residuals_ar.json）', 'keys': ['p_residuals_signed', 's_residuals_signed'], 'key_descriptions': {'p_residuals_signed': 'float list, P 波预测索引 − 真值索引（样本数）', 's_residuals_signed': 'float list, S 波预测索引 − 真值索引（样本数）'}}, 'pca': {'filename_pattern': 'pca_<method>.npz（如 pca_ar.npz）', 'arrays': ['features', 'labels'], 'descriptions': {'features': 'shape (N, D), 每样本特征向量（12 维：max_p, max_s, mean_p, mean_s, std_p, std_s, entropy_p, entropy_s, peak_width_p, peak_width_s, margin_p, margin_s）', 'labels': 'shape (N,) int, 0=无震相 1=P_only 2=S_only 3=P+S'}}}
    with open(os.path.join(case_dir, 'comparison_data_format.json'), 'w', encoding='utf-8') as f:
        json.dump(fmt, f, ensure_ascii=False, indent=2)

def _select_phase_peaks(prob: np.ndarray, threshold: float, min_interval: int, cap: int) -> list[int]:
    thr = float(threshold)
    indices = np.where(prob >= thr)[0]
    if indices.size == 0:
        return []
    sorted_indices = sorted(indices, key=lambda idx: -prob[idx])
    peaks: list[int] = []
    for idx in sorted_indices:
        if all((abs(idx - prev) >= min_interval for prev in peaks)):
            peaks.append(idx)
            if len(peaks) >= cap:
                break
    return sorted(peaks)

def apply_global_cap(peaks: list[int], prob: np.ndarray, cap: int) -> tuple[list[int], int]:
    if cap <= 0 or not peaks:
        return (peaks, 0)
    sorted_by_prob = sorted(peaks, key=lambda idx: -prob[idx])
    kept = sorted(sorted_by_prob[:cap])
    removed = len(peaks) - len(kept)
    return (kept, removed)

def enforce_p_before_s(p_peaks: list[int], s_peaks: list[int], min_gap: int, p_prob: np.ndarray, s_prob: np.ndarray) -> tuple[list[int], int]:
    if min_gap <= 0 or not p_peaks or (not s_peaks):
        return (s_peaks, 0)
    threshold = max(p_peaks[:2]) + min_gap
    filtered = [s for s in s_peaks if s > threshold]
    if filtered:
        return (filtered, len(s_peaks) - len(filtered))
    if len(p_peaks) > 1:
        threshold = p_peaks[1] + min_gap
        filtered = [s for s in s_peaks if s > threshold]
        if filtered:
            return (filtered, len(s_peaks) - len(filtered))
    return (s_peaks[:min(2, len(s_peaks))], len(s_peaks) - min(2, len(s_peaks)))

def _apply_ps_spacing(peaks: list[int], other_peaks: list[int], min_gap: int) -> list[int]:
    if min_gap <= 0 or not other_peaks:
        return peaks
    filtered = []
    for idx in peaks:
        if all((abs(idx - prev) >= min_gap for prev in other_peaks)):
            filtered.append(idx)
    return filtered

def compute_uncertainty_from_probs(probs: torch.Tensor, mode: str='max_prob', time_window: int=0, aggregate: str='mean', eps: float=1e-08, fusion_alpha: float=0.5, use_phase_channels: bool=True) -> torch.Tensor:
    valid_modes = ('max_prob', 'entropy', 'margin', 'fusion')
    assert mode in valid_modes, f'Unknown uncertainty mode: {mode}'
    assert probs.dim() == 3, f'Expected [B,C,T], got {probs.shape}'
    B, C, T = probs.shape
    if C < 3:
        raise ValueError('Expected at least 3 channels: background, P, S')
    p_prob = probs[:, 1, :]
    s_prob = probs[:, 2, :]
    has_time = False
    if mode == 'max_prob':
        u = torch.stack([(1.0 - p_prob.max(dim=1)[0]).clamp(0.0, 1.0), (1.0 - s_prob.max(dim=1)[0]).clamp(0.0, 1.0)], dim=1)
    else:
        has_time = True
        if mode == 'entropy':
            if use_phase_channels:
                pb = probs[:, 0, :]
                pp = probs[:, 1, :]
                p2 = torch.stack([pb, pp], dim=1)
                ent_p = -(p2 * (p2 + eps).log()).sum(dim=1) / math.log(2.0)
                ps = probs[:, 2, :]
                s2 = torch.stack([pb, ps], dim=1)
                ent_s = -(s2 * (s2 + eps).log()).sum(dim=1) / math.log(2.0)
            else:
                ent = -(probs * (probs + eps).log()).sum(dim=1) / math.log(float(C))
                ent_p = ent_s = ent
            u = torch.stack([ent_p.clamp(0.0, 1.0), ent_s.clamp(0.0, 1.0)], dim=1)
        elif mode == 'margin':
            top2 = torch.topk(probs, k=2, dim=1).values
            margin = (top2[:, 0, :] - top2[:, 1, :]).clamp(0.0, 1.0)
            u = torch.stack([(1.0 - margin).clamp(0.0, 1.0)] * 2, dim=1)
        elif mode == 'fusion':
            ent_u = compute_uncertainty_from_probs(probs, mode='entropy', time_window=0, aggregate='none', eps=eps, use_phase_channels=use_phase_channels)
            mar_u = compute_uncertainty_from_probs(probs, mode='margin', time_window=0, aggregate='none', eps=eps, use_phase_channels=use_phase_channels)
            ent_u_p, ent_u_s = ent_u.unbind(dim=1)
            mar_u_p, mar_u_s = mar_u.unbind(dim=1)
            u = torch.stack([(fusion_alpha * ent_u_p + (1.0 - fusion_alpha) * mar_u_p).clamp(0.0, 1.0), (fusion_alpha * ent_u_s + (1.0 - fusion_alpha) * mar_u_s).clamp(0.0, 1.0)], dim=1)
        else:
            raise ValueError(f'Unknown uncertainty mode: {mode}')
    if has_time and time_window and (time_window > 1):
        pad = time_window // 2
        u = F.pad(u, (pad, pad), mode='replicate')
        u = F.avg_pool1d(u, kernel_size=time_window, stride=1)
    if has_time and aggregate and (aggregate != 'none'):
        if aggregate == 'mean':
            u = u.mean(dim=-1)
        elif aggregate == 'max':
            u = u.max(dim=-1).values
        elif aggregate == 'q90':
            u = torch.quantile(u, 0.9, dim=-1)
        else:
            raise ValueError(f'Unknown aggregate: {aggregate}')
    return u

def _u_to_thr_one(u: torch.Tensor, min_thr: float, max_thr: float, mapping: str, sigmoid_a: float, sigmoid_b: float, piecewise_thresholds: tuple, piecewise_high: float, piecewise_mid: float, piecewise_low: float) -> torch.Tensor:
    u = u.clamp(0.0, 1.0)
    if mapping == 'sigmoid':
        s = 1.0 / (1.0 + torch.exp(-sigmoid_a * (u - sigmoid_b)))
        thr = min_thr + (max_thr - min_thr) * s
    elif mapping == 'piecewise':
        t0, t1 = (piecewise_thresholds[0], piecewise_thresholds[1])
        low = torch.full_like(u, piecewise_low)
        mid = torch.full_like(u, piecewise_mid)
        high = torch.full_like(u, piecewise_high)
        thr = torch.where(u < t0, low, torch.where(u < t1, mid, high))
    else:
        thr = min_thr + (max_thr - min_thr) * u
    return thr.clamp(min_thr, max_thr)

def uncertainty_to_threshold(uncertainty: torch.Tensor, min_thr: float=0.42, max_thr: float=0.7, min_thr_s: float=None, max_thr_s: float=None, mapping: str='linear', use_lower_threshold_for_s: bool=False, sigmoid_a: float=10.0, sigmoid_b: float=0.5, piecewise_thresholds: tuple=(0.4, 0.6), piecewise_high: float=0.75, piecewise_mid: float=0.65, piecewise_low: float=0.55, p_piecewise_thresholds: tuple=None, p_piecewise_high: float=None, p_piecewise_mid: float=None, p_piecewise_low: float=None, s_piecewise_thresholds: tuple=None, s_piecewise_high: float=None, s_piecewise_mid: float=None, s_piecewise_low: float=None) -> torch.Tensor:
    u_p, u_s = (uncertainty[:, 0], uncertainty[:, 1])
    p_thrds = p_piecewise_thresholds or piecewise_thresholds
    s_thrds = s_piecewise_thresholds or piecewise_thresholds
    p_hi = piecewise_high if p_piecewise_high is None else p_piecewise_high
    p_md = piecewise_mid if p_piecewise_mid is None else p_piecewise_mid
    p_lo = piecewise_low if p_piecewise_low is None else p_piecewise_low
    s_hi = piecewise_high if s_piecewise_high is None else s_piecewise_high
    s_md = piecewise_mid if s_piecewise_mid is None else s_piecewise_mid
    s_lo = piecewise_low if s_piecewise_low is None else s_piecewise_low
    kw_base = dict(mapping=mapping, sigmoid_a=sigmoid_a, sigmoid_b=sigmoid_b)
    kw_p = dict(**kw_base, piecewise_thresholds=p_thrds, piecewise_high=p_hi, piecewise_mid=p_md, piecewise_low=p_lo)
    kw_s = dict(**kw_base, piecewise_thresholds=s_thrds, piecewise_high=s_hi, piecewise_mid=s_md, piecewise_low=s_lo)
    thr_p = _u_to_thr_one(u_p, min_thr, max_thr, **kw_p)
    if use_lower_threshold_for_s or (min_thr_s is not None or max_thr_s is not None):
        min_s = min_thr_s if min_thr_s is not None else min(0.38, min_thr - 0.04)
        max_s = max_thr_s if max_thr_s is not None else max(0.65, max_thr - 0.05)
        thr_s = _u_to_thr_one(u_s, min_s, max_s, **kw_s)
    else:
        thr_s = _u_to_thr_one(u_s, min_thr, max_thr, **kw_s)
    return torch.stack([thr_p, thr_s], dim=1)

def _get_uncertainty_threshold_kwargs(case_or_opts: dict) -> dict:
    o = case_or_opts or {}
    return {'min_thr': o.get('min_thr', 0.42), 'max_thr': o.get('max_thr', 0.7), 'time_window': o.get('uncertainty_time_window', 0), 'aggregate': o.get('uncertainty_aggregate', 'mean'), 'mapping': o.get('threshold_mapping', 'linear'), 'use_lower_threshold_for_s': o.get('use_lower_threshold_for_s', False), 'min_thr_s': o.get('min_thr_s'), 'max_thr_s': o.get('max_thr_s')}

def _stable_dynamic_threshold(uncertainty: torch.Tensor, state: dict | None, opts: dict) -> tuple[torch.Tensor, dict]:
    B = uncertainty.shape[0]
    thr_p = float(opts.get('base_thr_p', 0.45))
    thr_s = float(opts.get('base_thr_s', 0.42))
    suggested = torch.stack([uncertainty.new_full((B,), thr_p), uncertainty.new_full((B,), thr_s)], dim=1)
    return (suggested, state or {})

def _center_crop_time(y: torch.Tensor, target_T: int) -> torch.Tensor:
    T = y.shape[-1]
    if T == target_T:
        return y
    if T < target_T:
        pad = target_T - T
        left = pad // 2
        right = pad - left
        return F.pad(y, (left, right))
    start = (T - target_T) // 2
    return y[..., start:start + target_T]

def soft_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (-(y * torch.log_softmax(logits, dim=1))).mean()

def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float, beta: float, eps: float=1e-06) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)

def compute_phasewise_loss(logits: torch.Tensor, y: torch.Tensor, p_alpha: float, p_beta: float, s_alpha: float, s_beta: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    prob_p = probs[:, 1, :].clamp(0.0, 1.0)
    prob_s = probs[:, 2, :].clamp(0.0, 1.0)
    target_p = y[:, 1, :]
    target_s = y[:, 2, :]
    loss_p = tversky_loss(prob_p, target_p, alpha=p_alpha, beta=p_beta)
    loss_s = tversky_loss(prob_s, target_s, alpha=s_alpha, beta=s_beta)
    return loss_p + loss_s

def ttversky_time_start_loss(logits: torch.Tensor, y: torch.Tensor, temporal_att: torch.Tensor | None, alpha_p: float, beta_p: float, alpha_s: float, beta_s: float, time_weight: float=0.3, start_weight: float=0.4, temporal_att_weight: float=0.1, start_window: int=2, start_peak_threshold: float=0.2, eps: float=1e-06) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    pred_p = probs[:, 1, :]
    pred_s = probs[:, 2, :]
    target_p = y[:, 1, :]
    target_s = y[:, 2, :]

    def _tversky_1d(pred: torch.Tensor, target: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        tp = (pred * target).sum(dim=1)
        fp = (pred * (1.0 - target)).sum(dim=1)
        fn = ((1.0 - pred) * target).sum(dim=1)
        return (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    tver_p = _tversky_1d(pred_p, target_p, alpha=alpha_p, beta=beta_p)
    tver_s = _tversky_1d(pred_s, target_s, alpha=alpha_s, beta=beta_s)
    loss_tver = (1.0 - tver_p + 1.0 - tver_s) * 0.5
    grad_pred_p = (pred_p[:, 1:] - pred_p[:, :-1]).abs()
    grad_tgt_p = (target_p[:, 1:] - target_p[:, :-1]).abs()
    grad_pred_s = (pred_s[:, 1:] - pred_s[:, :-1]).abs()
    grad_tgt_s = (target_s[:, 1:] - target_s[:, :-1]).abs()
    loss_time = F.mse_loss(grad_pred_p, grad_tgt_p) + F.mse_loss(grad_pred_s, grad_tgt_s)
    B, T = target_p.shape
    idx = torch.arange(T, device=logits.device).unsqueeze(0).expand(B, -1)
    pos_p = target_p.argmax(dim=1)
    pos_s = target_s.argmax(dim=1)
    peak_p = target_p.gather(1, pos_p.unsqueeze(1)).squeeze(1)
    peak_s = target_s.gather(1, pos_s.unsqueeze(1)).squeeze(1)

    def _start_mask(pos: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
        lo = (pos - start_window).clamp(0, T - 1).unsqueeze(1)
        hi = (pos + start_window).clamp(0, T - 1).unsqueeze(1)
        mask = (idx >= lo) & (idx <= hi)
        mask = mask & (peak.unsqueeze(1) > start_peak_threshold)
        return mask.float()
    mask_p = _start_mask(pos_p, peak_p)
    mask_s = _start_mask(pos_s, peak_s)
    bce_p = F.binary_cross_entropy(pred_p, target_p, reduction='none')
    bce_s = F.binary_cross_entropy(pred_s, target_s, reduction='none')
    loss_start_p = (bce_p * mask_p).sum() / (mask_p.sum() + eps)
    loss_start_s = (bce_s * mask_s).sum() / (mask_s.sum() + eps)
    loss_start = 0.5 * (loss_start_p + loss_start_s)
    if temporal_att is not None and isinstance(temporal_att, torch.Tensor):
        att_mean = temporal_att.mean(dim=(1, 2))
        loss_tver = loss_tver * (1.0 + temporal_att_weight * att_mean)
    total = loss_tver.mean() + time_weight * loss_time + start_weight * loss_start
    return total

def combined_loss(logits: torch.Tensor, y: torch.Tensor, threshold_balance_weight: float=0.1, tol_samples: int=10, class_weights: torch.Tensor=None, optimal_thr_mse_weight: float=0.15, sample_level_balance: bool=True, thr_reg_weight: float=0.008, use_phasewise_loss: bool=False, phasewise_loss_weight: float=0.0, p_tversky_alpha: float=0.7, p_tversky_beta: float=0.3, s_tversky_alpha: float=0.3, s_tversky_beta: float=0.7, use_ttversky_loss: bool=False, tt_loss_weight: float=0.3, temporal_att: torch.Tensor | None=None, tt_time_weight: float=0.3, tt_start_weight: float=0.4, tt_temporal_att_weight: float=0.1, tt_start_window: int=2, tt_start_peak_threshold: float=0.2, tt_alpha_p: float=0.7, tt_beta_p: float=0.3, tt_alpha_s: float=0.8, tt_beta_s: float=0.2) -> torch.Tensor:
    base_loss = soft_ce(logits, y)
    total_loss = base_loss
    if use_ttversky_loss:
        tt_loss = ttversky_time_start_loss(logits=logits, y=y, temporal_att=temporal_att, alpha_p=tt_alpha_p, beta_p=tt_beta_p, alpha_s=tt_alpha_s, beta_s=tt_beta_s, time_weight=tt_time_weight, start_weight=tt_start_weight, temporal_att_weight=tt_temporal_att_weight, start_window=tt_start_window, start_peak_threshold=tt_start_peak_threshold)
        total_loss = total_loss + float(tt_loss_weight) * tt_loss
    if use_phasewise_loss:
        total_loss = total_loss + phasewise_loss_weight * compute_phasewise_loss(logits, y, p_alpha=p_tversky_alpha, p_beta=p_tversky_beta, s_alpha=s_tversky_alpha, s_beta=s_tversky_beta)
    return total_loss

def build_datasets(case: Dict[str, Any]=None):
    aug = case or {}
    use_waveform_aug = False
    aug_noise_min = float(aug.get('aug_noise_snr_db_min', 3.0))
    aug_noise_max = float(aug.get('aug_noise_snr_db_max', 20.0))
    aug_amp_min = float(aug.get('aug_amplitude_min', 0.5))
    aug_amp_max = float(aug.get('aug_amplitude_max', 2.0))
    aug_params = (aug_noise_min, aug_noise_max, aug_amp_min, aug_amp_max)
    if DATA_SOURCE == 'npz':
        train_ds = WaveformDataset(TRAIN_DIR, TRAIN_CSV, crop_len=CROP_LEN, label_width=LABEL_WIDTH, training=True, sampling_rate=SAMPLE_RATE, label_sigma_sec=LABEL_SIGMA_SEC)
        valid_ds = WaveformDataset(TRAIN_DIR, VALID_CSV, crop_len=CROP_LEN, label_width=LABEL_WIDTH, training=False, sampling_rate=SAMPLE_RATE, label_sigma_sec=LABEL_SIGMA_SEC)
        return (train_ds, valid_ds, None)
    elif DATA_SOURCE == 'ceed':
        cache_key = _get_split_cache_key(use_waveform_aug, aug_params)
        if cache_key in _dataset_split_cache:
            train_idx, val_idx = _dataset_split_cache[cache_key]
            print(f'[build_datasets] ✓ 使用缓存的 CEED 划分（train={len(train_idx)}, val={len(val_idx)}），test 使用独立 test split')
            full_train_ds_aug = CEEDDataset(dataset_name=CEED_DATASET_NAME, split=CEED_TRAIN_SPLIT, limit=CEED_LIMIT_TRAIN, waveform_key=CEED_WAVEFORM_KEY, p_key=CEED_P_KEY, s_key=CEED_S_KEY, sampling_rate=SAMPLE_RATE, crop_len=CROP_LEN, label_sigma_sec=LABEL_SIGMA_SEC, label_width=LABEL_WIDTH, training=True, local_dir=CEED_LOCAL_DIR, use_waveform_augmentation=use_waveform_aug, aug_noise_snr_db_min=aug_noise_min, aug_noise_snr_db_max=aug_noise_max, aug_amplitude_min=aug_amp_min, aug_amplitude_max=aug_amp_max)
        else:
            print(f'[build_datasets] CEED：从 train split 加载（limit={CEED_LIMIT_TRAIN}），9:1 划分训练/验证；test 使用 test split')
            full_train_ds_aug = CEEDDataset(dataset_name=CEED_DATASET_NAME, split=CEED_TRAIN_SPLIT, limit=CEED_LIMIT_TRAIN, waveform_key=CEED_WAVEFORM_KEY, p_key=CEED_P_KEY, s_key=CEED_S_KEY, sampling_rate=SAMPLE_RATE, crop_len=CROP_LEN, label_sigma_sec=LABEL_SIGMA_SEC, label_width=LABEL_WIDTH, training=True, local_dir=CEED_LOCAL_DIR, use_waveform_augmentation=use_waveform_aug, aug_noise_snr_db_min=aug_noise_min, aug_noise_snr_db_max=aug_noise_max, aug_amplitude_min=aug_amp_min, aug_amplitude_max=aug_amp_max)
            n_full = len(full_train_ds_aug)
            if n_full < 2:
                raise ValueError(f'CEED train 样本不足（{n_full}），至少需要 2 条')
            val_size = max(1, int(n_full * CEED_VAL_RATIO))
            train_size = n_full - val_size
            g = torch.Generator().manual_seed(SEED)
            perm = torch.randperm(n_full, generator=g)
            train_idx = perm[:train_size].tolist()
            val_idx = perm[train_size:].tolist()
            _dataset_split_cache[cache_key] = (train_idx, val_idx)
            print(f'[build_datasets] ✓ CEED train 划分完成并已缓存（train={train_size}, val={val_size}）')
        train_ds = Subset(full_train_ds_aug, train_idx)
        full_train_ds_clean = CEEDDataset(dataset_name=CEED_DATASET_NAME, split=CEED_TRAIN_SPLIT, limit=CEED_LIMIT_TRAIN, waveform_key=CEED_WAVEFORM_KEY, p_key=CEED_P_KEY, s_key=CEED_S_KEY, sampling_rate=SAMPLE_RATE, crop_len=CROP_LEN, label_sigma_sec=LABEL_SIGMA_SEC, label_width=LABEL_WIDTH, training=False, local_dir=CEED_LOCAL_DIR)
        valid_ds = Subset(full_train_ds_clean, val_idx)
        global _ceed_test_ds_cache
        if _ceed_test_ds_cache is None:
            _ceed_test_ds_cache = CEEDDataset(dataset_name=CEED_DATASET_NAME, split=CEED_TEST_SPLIT, limit=CEED_LIMIT_TEST, waveform_key=CEED_WAVEFORM_KEY, p_key=CEED_P_KEY, s_key=CEED_S_KEY, sampling_rate=SAMPLE_RATE, crop_len=CROP_LEN, label_sigma_sec=LABEL_SIGMA_SEC, label_width=LABEL_WIDTH, training=False, local_dir=CEED_LOCAL_DIR)
        test_ds = _ceed_test_ds_cache
        return (train_ds, valid_ds, test_ds)
    elif DATA_SOURCE == 'h5_three_channel':
        cache_key = _get_split_cache_key(False, ())
        if cache_key in _dataset_split_cache:
            train_idx, val_idx, test_idx = _dataset_split_cache[cache_key]
            print(f'[build_datasets] ✓ 使用缓存的三通道 H5 划分（train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}）')
        else:
            full_ds = ThreeChannelH5Dataset(root_dir=H5_THREE_CHANNEL_ROOT, crop_len=CROP_LEN, sampling_rate=SAMPLE_RATE, label_sigma_sec=LABEL_SIGMA_SEC, label_width=LABEL_WIDTH, training=True, arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT, filter_natural_only=H5_FILTER_NATURAL_ONLY, allow_earthquake_types=H5_ALLOW_TYPES, strict_check=H5_STRICT_CHECK, limit=H5_LIMIT)
            n_full = len(full_ds)
            if n_full < 2:
                raise ValueError(f'三通道 H5 样本数不足（{n_full}），无法划分')
            test_size = int(n_full * H5_TEST_RATIO) if H5_TEST_RATIO and H5_TEST_RATIO > 0 else 0
            remainder = n_full - test_size
            val_size = max(1, int(remainder * (1 - H5_TRAIN_VAL_RATIO)))
            train_size = remainder - val_size
            if train_size < 1:
                raise ValueError(f'三通道 H5 划分后训练集为空（n_full={n_full}, test={test_size}, val={val_size}）')
            g = torch.Generator().manual_seed(SEED)
            perm = torch.randperm(n_full, generator=g)
            test_idx = perm[n_full - test_size:].tolist() if test_size > 0 else []
            train_idx = perm[:train_size].tolist()
            val_idx = perm[train_size:train_size + val_size].tolist()
            _dataset_split_cache[cache_key] = (train_idx, val_idx, test_idx)
            print(f'[build_datasets] ✓ 三通道 H5 划分完成（先划出 test={len(test_idx)}，剩余 9:1 → train={train_size}, val={val_size}）')
        full_train_ds = ThreeChannelH5Dataset(root_dir=H5_THREE_CHANNEL_ROOT, crop_len=CROP_LEN, sampling_rate=SAMPLE_RATE, label_sigma_sec=LABEL_SIGMA_SEC, label_width=LABEL_WIDTH, training=True, arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT, filter_natural_only=H5_FILTER_NATURAL_ONLY, allow_earthquake_types=H5_ALLOW_TYPES, strict_check=H5_STRICT_CHECK, limit=H5_LIMIT)
        full_valid_ds = ThreeChannelH5Dataset(root_dir=H5_THREE_CHANNEL_ROOT, crop_len=CROP_LEN, sampling_rate=SAMPLE_RATE, label_sigma_sec=LABEL_SIGMA_SEC, label_width=LABEL_WIDTH, training=False, arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT, filter_natural_only=H5_FILTER_NATURAL_ONLY, allow_earthquake_types=H5_ALLOW_TYPES, strict_check=H5_STRICT_CHECK, limit=H5_LIMIT)
        train_idx_use = train_idx[:H5_LIMIT_TRAIN] if H5_LIMIT_TRAIN is not None else train_idx
        val_idx_use = val_idx[:H5_LIMIT_VAL] if H5_LIMIT_VAL is not None else val_idx
        test_idx_use = test_idx[:H5_LIMIT_TEST] if H5_LIMIT_TEST is not None and test_idx else test_idx
        train_ds = Subset(full_train_ds, train_idx_use)
        valid_ds = Subset(full_valid_ds, val_idx_use)
        if test_idx_use:
            full_test_ds = ThreeChannelH5Dataset(root_dir=H5_THREE_CHANNEL_ROOT, crop_len=CROP_LEN, sampling_rate=SAMPLE_RATE, label_sigma_sec=LABEL_SIGMA_SEC, label_width=LABEL_WIDTH, training=False, arrival_relative_to_segment=H5_ARRIVAL_RELATIVE_TO_SEGMENT, filter_natural_only=H5_FILTER_NATURAL_ONLY, allow_earthquake_types=H5_ALLOW_TYPES, strict_check=H5_STRICT_CHECK, limit=H5_LIMIT)
            test_ds = Subset(full_test_ds, test_idx_use)
        else:
            test_ds = None
    else:
        raise ValueError("DATA_SOURCE must be 'npz', 'ceed', or 'h5_three_channel'")
    return (train_ds, valid_ds, test_ds)

@torch.inference_mode()
def eval_loss(model, loader, device, threshold_balance_weight: float=0.1, class_weights: torch.Tensor=None, epoch: int=0, total_epochs: int=0) -> float:
    model.eval()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [valid]', leave=False) if epoch > 0 else loader
    for x, y, _ in pbar:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        temporal_att = getattr(model, 'last_temporal_att', None)
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        loss = combined_loss(logits, y, threshold_balance_weight=threshold_balance_weight, tol_samples=DYN_TOL_SAMPLES, class_weights=class_weights, optimal_thr_mse_weight=0, sample_level_balance=True, use_ttversky_loss=bool(getattr(model, 'use_ttversky_loss', False)), tt_loss_weight=float(getattr(model, 'tt_loss_weight', 0.3)), temporal_att=temporal_att, tt_time_weight=float(getattr(model, 'tt_time_weight', 0.3)), tt_start_weight=float(getattr(model, 'tt_start_weight', 0.4)), tt_temporal_att_weight=float(getattr(model, 'tt_temporal_att_weight', 0.1)), tt_start_window=int(getattr(model, 'tt_start_window', 2)), tt_start_peak_threshold=float(getattr(model, 'tt_start_peak_threshold', 0.2)), tt_alpha_p=float(getattr(model, 'tt_alpha_p', 0.7)), tt_beta_p=float(getattr(model, 'tt_beta_p', 0.3)), tt_alpha_s=float(getattr(model, 'tt_alpha_s', 0.8)), tt_beta_s=float(getattr(model, 'tt_beta_s', 0.2)))
        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs
        if epoch > 0:
            pbar.set_postfix({'loss': f'{total / max(1, n):.4f}'})
    return total / max(1, n)

@torch.inference_mode()
def collect_conf_err(model, loader, device):
    model.eval()
    p_conf, p_err, p_has = ([], [], [])
    s_conf, s_err, s_has = ([], [], [])
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_np = y.cpu().numpy()
        B = probs.shape[0]
        for b in range(B):
            has_p = bool(y_np[b, 1].max() > 1e-06)
            p_idx = int(np.argmax(probs[b, 1]))
            p_c = float(probs[b, 1, p_idx])
            p_e = abs(p_idx - int(np.argmax(y_np[b, 1]))) if has_p else None
            p_conf.append(p_c)
            p_err.append(p_e)
            p_has.append(has_p)
            has_s = bool(y_np[b, 2].max() > 1e-06)
            s_idx = int(np.argmax(probs[b, 2]))
            s_c = float(probs[b, 2, s_idx])
            s_e = abs(s_idx - int(np.argmax(y_np[b, 2]))) if has_s else None
            s_conf.append(s_c)
            s_err.append(s_e)
            s_has.append(has_s)
    return ((p_conf, p_err, p_has), (s_conf, s_err, s_has))

def best_threshold(confs, errs, has_gts, tol: int, grid: list[float], debug: bool=False):
    best_thr, best_f1 = (0.5, -1.0)
    best_stats = None
    total_samples = len(confs)
    samples_with_gt = sum(has_gts)
    samples_without_gt = total_samples - samples_with_gt
    if debug and total_samples > 0:
        print(f'  总样本数: {total_samples}, 有标签: {samples_with_gt}, 无标签: {samples_without_gt}')
        if samples_with_gt > 0:
            valid_errs = [e for e, h in zip(errs, has_gts) if h and e is not None]
            if valid_errs:
                print(f'  有效误差统计: mean={np.mean(valid_errs):.1f}, min={np.min(valid_errs)}, max={np.max(valid_errs)}, <=tol({tol})的比例={np.mean(np.array(valid_errs) <= tol):.1%}')
            valid_confs = [c for c, h in zip(confs, has_gts) if h]
            if valid_confs:
                print(f'  有标签样本的置信度统计: mean={np.mean(valid_confs):.3f}, min={np.min(valid_confs):.3f}, max={np.max(valid_confs):.3f}')
    for thr in grid:
        tp = fp = fn = 0
        for c, e, h in zip(confs, errs, has_gts):
            if c >= thr:
                if h and e is not None and (e <= tol):
                    tp += 1
                else:
                    fp += 1
            elif h:
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_stats = (tp, fp, fn, prec, rec, f1)
    if best_f1 < 0:
        best_f1 = 0.0
    if debug and best_stats:
        tp, fp, fn, prec, rec, f1 = best_stats
        print(f'  最佳阈值={best_thr:.3f}: TP={tp}, FP={fp}, FN={fn}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}')
    return (best_thr, best_f1)

@torch.inference_mode()
def eval_detailed(model, loader, device, thr_p: float=None, thr_s: float=None, uncertainty_threshold_options: dict=None, tol: int=10, current_epoch: int | None=None, quiet: bool=False) -> Dict[str, float]:
    model.eval()
    time_correct = 0
    n_time = 0
    p_tp = p_fp = p_fn = p_tn = 0
    s_tp = s_fp = s_fn = s_tn = 0
    all_p_residual_signed: list[int] = []
    all_s_residual_signed: list[int] = []
    s_conf_list, s_thr_list = ([], [])
    opts = uncertainty_threshold_options or {}
    structural_opts = {'enabled': bool(opts.get('use_structural_postproc', True)), 'min_interval_same': int(opts.get('postproc_min_interval_same', 20)), 'min_interval_ps': int(opts.get('postproc_min_interval_ps', 30)), 'candidate_limit': int(opts.get('postproc_candidate_limit', 10)), 'cap_p': int(opts.get('postproc_cap_p', 1)), 'cap_s': int(opts.get('postproc_cap_s', 1)), 'enforce_p_before_s': bool(opts.get('postproc_enforce_p_before_s', False))}
    structural_stats = {'cap_p': 0, 'cap_s': 0, 'order': 0}
    candidate_lengths: dict[str, list[int]] = {'p': [], 's': []}
    zero_candidate_counts = {'p': 0, 's': 0}
    samples_processed = 0
    threshold_pass_counts = {'p': 0, 's': 0}
    uncertainty_stats = {'p': {'tp': [], 'fp': [], 'fn': []}, 's': {'tp': [], 'fp': [], 'fn': []}}
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        use_per_sample_threshold = False
        suggested_thr = None
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        pred_cls = probs.argmax(dim=1)
        true_cls = y.argmax(dim=1)
        time_correct += (pred_cls == true_cls).sum().item()
        n_time += pred_cls.numel()
        probs_np = probs.cpu().numpy()
        y_np = y.cpu().numpy()
        B = probs_np.shape[0]
        uncertainty_np = np.zeros((B, 2))
        if use_per_sample_threshold:
            suggested_thr_np = suggested_thr.cpu().numpy()
        else:
            suggested_thr_np = None
        for b in range(B):
            sample_thr_p = float(suggested_thr_np[b, 0]) if use_per_sample_threshold else thr_p if thr_p is not None else 0.5
            sample_thr_s = float(suggested_thr_np[b, 1]) if use_per_sample_threshold else thr_s if thr_s is not None else 0.5
            p_prob = probs_np[b, 1]
            s_prob = probs_np[b, 2]
            if use_per_sample_threshold:
                threshold_pass_counts['p'] += int(float(p_prob.max()) >= sample_thr_p)
                threshold_pass_counts['s'] += int(float(s_prob.max()) >= sample_thr_s)
                samples_processed += 1
            cur_unc_p = float(uncertainty_np[b, 0]) if use_per_sample_threshold else None
            cur_unc_s = float(uncertainty_np[b, 1]) if use_per_sample_threshold else None
            if structural_opts['enabled']:
                p_candidates = _select_phase_peaks(p_prob, sample_thr_p, structural_opts['min_interval_same'], structural_opts['candidate_limit'])
                p_candidates, removed_p_cap = apply_global_cap(p_candidates, p_prob, structural_opts['cap_p'])
                structural_stats['cap_p'] += removed_p_cap
            else:
                p_candidates = [int(np.argmax(p_prob))]
            candidate_lengths['p'].append(len(p_candidates))
            if len(p_candidates) == 0:
                zero_candidate_counts['p'] += 1
            p_pred = False
            p_idx = int(np.argmax(p_prob))
            p_conf = float(p_prob[p_idx])
            for idx in p_candidates:
                p_val = float(p_prob[idx])
                if p_val >= sample_thr_p:
                    p_pred = True
                    p_idx = int(idx)
                    p_conf = p_val
                    break
            if structural_opts['enabled'] and (not p_pred):
                p_idx = int(np.argmax(p_prob))
                p_conf = float(p_prob[p_idx])
            if structural_opts['enabled']:
                s_candidates = _select_phase_peaks(s_prob, sample_thr_s, structural_opts['min_interval_same'], structural_opts['candidate_limit'])
                s_candidates = _apply_ps_spacing(s_candidates, [p_idx] if p_pred else [], structural_opts['min_interval_ps'])
                if structural_opts['enforce_p_before_s']:
                    s_candidates, removed_order = enforce_p_before_s(p_candidates, s_candidates, structural_opts['min_interval_ps'], p_prob, s_prob)
                    structural_stats['order'] += removed_order
                s_candidates, removed_s_cap = apply_global_cap(s_candidates, s_prob, structural_opts['cap_s'])
                structural_stats['cap_s'] += removed_s_cap
            else:
                s_candidates = [int(np.argmax(s_prob))]
            candidate_lengths['s'].append(len(s_candidates))
            if len(s_candidates) == 0:
                zero_candidate_counts['s'] += 1
            s_pred = False
            s_idx = int(np.argmax(s_prob))
            s_conf = float(s_prob[s_idx])
            for idx in s_candidates:
                s_val = float(s_prob[idx])
                if s_val >= sample_thr_s:
                    s_pred = True
                    s_idx = int(idx)
                    s_conf = s_val
                    break
            if structural_opts['enabled'] and (not s_pred):
                s_idx = int(np.argmax(s_prob))
                s_conf = float(s_prob[s_idx])
            has_p = bool(y_np[b, 1].max() > 1e-06)
            if has_p:
                gt = int(np.argmax(y_np[b, 1]))
                err = abs(p_idx - gt)
                all_p_residual_signed.append(p_idx - gt)
                if p_pred and p_conf >= sample_thr_p:
                    if err <= tol:
                        p_tp += 1
                    else:
                        p_fp += 1
                else:
                    p_fn += 1
            elif p_pred and p_conf >= sample_thr_p:
                p_fp += 1
            else:
                p_tn += 1
            if use_per_sample_threshold and cur_unc_p is not None:
                if has_p:
                    if p_pred and p_conf >= sample_thr_p:
                        if err <= tol:
                            uncertainty_stats['p']['tp'].append(cur_unc_p)
                        else:
                            uncertainty_stats['p']['fp'].append(cur_unc_p)
                    else:
                        uncertainty_stats['p']['fn'].append(cur_unc_p)
                elif p_pred and p_conf >= sample_thr_p:
                    uncertainty_stats['p']['fp'].append(cur_unc_p)
            has_s = bool(y_np[b, 2].max() > 1e-06)
            if has_s:
                if use_per_sample_threshold and s_pred:
                    s_conf_list.append(s_conf)
                    s_thr_list.append(sample_thr_s)
                gt = int(np.argmax(y_np[b, 2]))
                err = abs(s_idx - gt)
                all_s_residual_signed.append(s_idx - gt)
                if s_pred and s_conf >= sample_thr_s:
                    if err <= tol:
                        s_tp += 1
                    else:
                        s_fp += 1
                else:
                    s_fn += 1
            elif s_pred and s_conf >= sample_thr_s:
                s_fp += 1
            else:
                s_tn += 1
            if use_per_sample_threshold and cur_unc_s is not None:
                if has_s:
                    if s_pred and s_conf >= sample_thr_s:
                        if err <= tol:
                            uncertainty_stats['s']['tp'].append(cur_unc_s)
                        else:
                            uncertainty_stats['s']['fp'].append(cur_unc_s)
                    else:
                        uncertainty_stats['s']['fn'].append(cur_unc_s)
                elif s_pred and s_conf >= sample_thr_s:
                    uncertainty_stats['s']['fp'].append(cur_unc_s)
    acc = time_correct / max(1, n_time)
    if not quiet and candidate_lengths['p']:

        def _print_len_stats(vals: list[int], tag: str):
            arr = np.array(vals, dtype=np.float32)
            q = np.quantile(arr, [0.1, 0.5, 0.9])
            print(f'[eval_detailed] {tag} candidate len: mean={arr.mean():.2f}, std={arr.std():.2f}, q10={q[0]:.1f}, q50={q[1]:.1f}, q90={q[2]:.1f}', flush=True)
        _print_len_stats(candidate_lengths['p'], 'P')
        _print_len_stats(candidate_lengths['s'], 'S')
        samples_total = len(candidate_lengths['p'])
        sample_denom = samples_total if samples_total else 1
        print(f'[eval_detailed] zero candidate rate: P={zero_candidate_counts['p'] / sample_denom:.1%}, S={zero_candidate_counts['s'] / sample_denom:.1%}', flush=True)
        pass_total = samples_processed if samples_processed else 1
        print(f'[eval_detailed] threshold pass counts (per sample tracked): P={threshold_pass_counts['p']}/{pass_total}, S={threshold_pass_counts['s']}/{pass_total}', flush=True)
        print(f'[eval_detailed] structural pruning: cap_p={structural_stats['cap_p']}, cap_s={structural_stats['cap_s']}, order={structural_stats['order']}', flush=True)
        for phase in ('p', 's'):
            stats = uncertainty_stats[phase]
            if stats['tp'] or stats['fp'] or stats['fn']:
                for tag in ('tp', 'fp', 'fn'):
                    arr = np.array(stats[tag]) if stats[tag] else np.array([0.0])
                    print(f'[eval_detailed] {phase.upper()} {tag} uncertainty: mean={arr.mean():.4f}, std={arr.std():.4f}', flush=True)
    if not quiet and use_per_sample_threshold and s_conf_list and (s_tp + s_fp == 0):
        s_conf_arr = np.array(s_conf_list)
        s_thr_arr = np.array(s_thr_list)
        above = np.sum(s_conf_arr >= s_thr_arr)
        print(f'[eval_detailed] S-F1=0 诊断（有S样本）: S预测概率 mean={s_conf_arr.mean():.4f}, min={s_conf_arr.min():.4f}, max={s_conf_arr.max():.4f} | 预测S阈值 mean={s_thr_arr.mean():.4f}, min={s_thr_arr.min():.4f}, max={s_thr_arr.max():.4f} | s_c>=thr_s 的样本数={above}/{len(s_conf_list)}', flush=True)

    def _prf(tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        return (p, r, f1)
    p_prec, p_rec, p_f1 = _prf(p_tp, p_fp, p_fn)
    s_prec, s_rec, s_f1 = _prf(s_tp, s_fp, s_fn)

    def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
        denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom <= 0:
            return 0.0
        return float((tp * tn - fp * fn) / math.sqrt(denom))
    tp_all = p_tp + s_tp
    fp_all = p_fp + s_fp
    fn_all = p_fn + s_fn
    tn_all = p_tn + s_tn
    mcc = _mcc(tp_all, fp_all, fn_all, tn_all)
    if all_p_residual_signed:
        p_res_arr = np.asarray(all_p_residual_signed, dtype=np.float32) / float(SAMPLE_RATE)
        p_res_mean = float(p_res_arr.mean())
        p_res_std = float(p_res_arr.std())
        p_res_mae = float(np.abs(p_res_arr).mean())
    else:
        p_res_mean = p_res_std = p_res_mae = None
    if all_s_residual_signed:
        s_res_arr = np.asarray(all_s_residual_signed, dtype=np.float32) / float(SAMPLE_RATE)
        s_res_mean = float(s_res_arr.mean())
        s_res_std = float(s_res_arr.std())
        s_res_mae = float(np.abs(s_res_arr).mean())
    else:
        s_res_mean = s_res_std = s_res_mae = None
    return dict(time_acc=float(acc), mcc=float(mcc), p_prec=float(p_prec), p_rec=float(p_rec), p_f1=float(p_f1), s_prec=float(s_prec), s_rec=float(s_rec), s_f1=float(s_f1), p_res_mean_sec=p_res_mean, p_res_std_sec=p_res_std, p_res_mae_sec=p_res_mae, s_res_mean_sec=s_res_mean, s_res_std_sec=s_res_std, s_res_mae_sec=s_res_mae, p_residuals_signed=all_p_residual_signed if all_p_residual_signed else None, s_residuals_signed=all_s_residual_signed if all_s_residual_signed else None)

def enable_dropout_only(m: nn.Module) -> None:
    for mod in m.modules():
        name = mod.__class__.__name__
        if 'Dropout' in name:
            mod.train()
        if 'BatchNorm' in name:
            mod.eval()

def mc_forward(model: nn.Module, x: torch.Tensor, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MC Dropout forward: run model T times with dropout enabled (BN frozen), return mean logits,
    per-sample scalar uncertainty, per-time MI_map_norm [B, L] for phase-wise local uncertainty,
    and per-time prob_var_map [B, L] as a variance-based diagnostic.
    """
    model.eval()
    enable_dropout_only(model)
    preds = []
    with torch.no_grad():
        for _ in range(T):
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.unsqueeze(0))
    preds = torch.cat(preds, dim=0)
    try:
        logits_std = preds.std(dim=0)
        logit_std_mean = float(logits_std.mean().item())
    except Exception:
        logits_std = None
        logit_std_mean = float('nan')
    probs = torch.softmax(preds, dim=2)
    try:
        probs_std = probs.std(dim=0)
        prob_std_mean = float(probs_std.mean().item())
        print(f'[mc_forward] logit_std_mean / prob_std_mean: {logit_std_mean:.4e} / {prob_std_mean:.4e}')
    except Exception:
        pass
    E_p = probs.mean(dim=0)
    mean_logits = preds.mean(dim=0)
    B, C, L = E_p.shape
    eps = 1e-08
    H_Ep = -(E_p * (E_p + eps).log()).sum(dim=1) / math.log(2.0)
    H_p_per = -(probs * (probs + eps).log()).sum(dim=2) / math.log(2.0)
    E_Hp = H_p_per.mean(dim=0)
    MI_map = (H_Ep - E_Hp).clamp(0.0, None)
    max_mi = math.log2(3.0)
    MI_map_norm = (MI_map / max_mi).clamp(0.0, 1.0)
    try:
        print(f'[mc_forward] MI_map_norm min/max/mean: {MI_map_norm.min().item():.4e} / {MI_map_norm.max().item():.4e} / {MI_map_norm.mean().item():.4e}')
    except Exception:
        pass
    try:
        prob_var = probs.var(dim=0, unbiased=False)
        prob_var_map = prob_var.mean(dim=1)
        print(f'[mc_forward] prob_var_map min/max/mean: {prob_var_map.min().item():.4e} / {prob_var_map.max().item():.4e} / {prob_var_map.mean().item():.4e}')
    except Exception:
        prob_var_map = MI_map_norm.new_zeros(MI_map_norm.shape)
    uncertainty = MI_map_norm.mean(dim=1)
    return (mean_logits, uncertainty, MI_map_norm, prob_var_map)

def eval_detailed_mc_selective(model: nn.Module, loader, device: torch.device, thr_p: float=0.5, thr_s: float=0.5, mc_T: int=20, drop_ratio: float=0.1, coverage_points: int=20, tol: int=10, structural_opts: dict | None=None, quiet: bool=False, eval_seed: int | None=None, use_two_level_candidate: bool=False, candidate_thr_p: float | None=None, candidate_thr_s: float | None=None, use_score_candidate_s: bool=False, score_lambda_s: float=0.6, score_tau_unc_s: float | None=None, use_unc_gating_s: bool=False, unc_gating_tau_s: float | None=None, unc_gating_k_s: float | None=None, unc_gating_base_s: float | None=None) -> Dict[str, Any]:
    """
    Evaluate with MC Dropout + fixed threshold + selective prediction.
    Returns full metrics (all samples) and selective metrics (after dropping top drop_ratio by uncertainty),
    plus risk_coverage data for plotting.
    """
    if eval_seed is not None:
        try:
            s = int(eval_seed)
            random.seed(s)
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
        except Exception:
            pass
    structural_opts = structural_opts or {'enabled': True, 'min_interval_same': 20, 'min_interval_ps': 30, 'candidate_limit': 10, 'cap_p': 1, 'cap_s': 1, 'enforce_p_before_s': False}
    if use_two_level_candidate:
        base_thr_p = thr_p if thr_p is not None else 0.5
        base_thr_s = thr_s if thr_s is not None else 0.5
        cand_thr_p = float(candidate_thr_p if candidate_thr_p is not None else 0.3)
        cand_thr_s = float(candidate_thr_s if candidate_thr_s is not None else 0.3)
        cand_thr_p = min(cand_thr_p, float(base_thr_p))
        cand_thr_s = min(cand_thr_s, float(base_thr_s))
    else:
        cand_thr_p = float(thr_p if thr_p is not None else 0.5)
        if use_score_candidate_s:
            base_thr_s = thr_s if thr_s is not None else 0.5
            cand_thr_s = float(candidate_thr_s if candidate_thr_s is not None else 0.3)
            cand_thr_s = min(cand_thr_s, float(base_thr_s))
        else:
            cand_thr_s = float(thr_s if thr_s is not None else 0.5)
    all_p_ok, all_p_err = ([], [])
    all_s_ok, all_s_err = ([], [])
    all_p_residual_signed = []
    all_s_residual_signed = []
    all_pred_p, all_pred_s = ([], [])
    all_uncertainty = []
    all_unc_p, all_unc_s = ([], [])
    all_p_conf, all_s_conf = ([], [])
    all_p_conf_single, all_s_conf_single = ([], [])
    all_has_p, all_has_s = ([], [])
    time_correct, n_time = (0, 0)
    UNC_WIN = 25

    def _local_unc(mi_row: np.ndarray, idx: int, win: int) -> float:
        if idx < 0 or idx >= len(mi_row):
            return float('nan')
        lo, hi = (max(0, idx - win), min(len(mi_row), idx + win + 1))
        seg = mi_row[lo:hi]
        return float(np.mean(seg)) if seg.size else float('nan')
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            single_logits = model(x)
            single_probs = torch.softmax(single_logits, dim=1)
        single_probs_np = single_probs.cpu().numpy()
        for b in range(single_probs_np.shape[0]):
            all_p_conf_single.append(float(single_probs_np[b, 1].max()))
            all_s_conf_single.append(float(single_probs_np[b, 2].max()))
        if was_training:
            model.train()
        mean_logits, uncertainty, mi_map, _ = mc_forward(model, x, mc_T, device)
        probs = torch.softmax(mean_logits, dim=1)
        if y.shape[-1] != probs.shape[-1]:
            y = _center_crop_time(y, probs.shape[-1])
        pred_cls = probs.argmax(dim=1)
        true_cls = y.argmax(dim=1)
        time_correct += (pred_cls == true_cls).sum().item()
        n_time += pred_cls.numel()
        probs_np = probs.cpu().numpy()
        y_np = y.cpu().numpy()
        unc_np = uncertainty.cpu().numpy()
        mi_map_np = mi_map.cpu().numpy()
        B = probs_np.shape[0]
        L = mi_map_np.shape[1]
        for b in range(B):
            p_prob = probs_np[b, 1]
            s_prob = probs_np[b, 2]
            sample_thr_p = thr_p
            sample_thr_s = thr_s
            if structural_opts['enabled']:
                p_cand_thr = cand_thr_p if use_two_level_candidate else sample_thr_p
                p_candidates = _select_phase_peaks(p_prob, p_cand_thr, structural_opts['min_interval_same'], structural_opts['candidate_limit'])
                p_candidates, _ = apply_global_cap(p_candidates, p_prob, structural_opts['cap_p'])
            else:
                p_candidates = [int(np.argmax(p_prob))]
            if use_two_level_candidate:
                viable_p = []
                for idx in p_candidates:
                    p_val = float(p_prob[idx])
                    if p_val >= sample_thr_p:
                        u = _local_unc(mi_map_np[b], int(idx), UNC_WIN)
                        viable_p.append((u, -p_val, int(idx)))
                if viable_p:
                    u_sel, negp_sel, best_idx = min(viable_p, key=lambda t: (t[0], t[1]))
                    p_pred = True
                    p_idx = int(best_idx)
                    p_conf = float(p_prob[p_idx])
                else:
                    p_pred = False
                    p_idx = int(np.argmax(p_prob))
                    p_conf = float(p_prob[p_idx])
            else:
                p_pred = False
                p_idx = int(np.argmax(p_prob))
                p_conf = float(p_prob[p_idx])
                for idx in p_candidates:
                    if float(p_prob[idx]) >= sample_thr_p:
                        p_pred = True
                        p_idx = int(idx)
                        p_conf = float(p_prob[idx])
                        break
            if structural_opts['enabled']:
                s_cand_thr = cand_thr_s if use_two_level_candidate or use_score_candidate_s else sample_thr_s
                s_candidates = _select_phase_peaks(s_prob, s_cand_thr, structural_opts['min_interval_same'], structural_opts['candidate_limit'])
                s_candidates = _apply_ps_spacing(s_candidates, [p_idx] if p_pred else [], structural_opts['min_interval_ps'])
                if structural_opts['enforce_p_before_s']:
                    s_candidates, _ = enforce_p_before_s(p_candidates, s_candidates, structural_opts['min_interval_ps'], p_prob, s_prob)
                s_candidates, _ = apply_global_cap(s_candidates, s_prob, structural_opts['cap_s'])
            else:
                s_candidates = [int(np.argmax(s_prob))]
            if use_two_level_candidate:
                viable_s = []
                for idx in s_candidates:
                    s_val = float(s_prob[idx])
                    if s_val >= sample_thr_s:
                        u = _local_unc(mi_map_np[b], int(idx), UNC_WIN)
                        viable_s.append((u, -s_val, int(idx)))
                if viable_s:
                    u_sel, negs_sel, best_idx_s = min(viable_s, key=lambda t: (t[0], t[1]))
                    s_pred = True
                    s_idx = int(best_idx_s)
                    s_conf = float(s_prob[s_idx])
                else:
                    s_pred = False
                    s_idx = int(np.argmax(s_prob))
                    s_conf = float(s_prob[s_idx])
            elif use_score_candidate_s:
                viable_s = []
                for idx in s_candidates:
                    s_val = float(s_prob[idx])
                    if s_val >= sample_thr_s:
                        u = _local_unc(mi_map_np[b], int(idx), UNC_WIN)
                        score = s_val - float(score_lambda_s) * u
                        viable_s.append((score, -u, -s_val, int(idx)))
                if viable_s:
                    score_sel, negu_sel, negp_sel, best_idx_s = max(viable_s, key=lambda t: (t[0], t[1], t[2]))
                    s_idx = int(best_idx_s)
                    s_conf = float(s_prob[s_idx])
                    u_sel = -negu_sel
                    if score_tau_unc_s is None or u_sel <= float(score_tau_unc_s):
                        s_pred = True
                    else:
                        s_pred = False
                else:
                    s_pred = False
                    s_idx = int(np.argmax(s_prob))
                    s_conf = float(s_prob[s_idx])
            else:
                s_pred = False
                s_idx = int(np.argmax(s_prob))
                s_conf = float(s_prob[s_idx])
                for idx in s_candidates:
                    if float(s_prob[idx]) >= sample_thr_s:
                        s_pred = True
                        s_idx = int(idx)
                        s_conf = float(s_prob[idx])
                        break
            has_p = bool(y_np[b, 1].max() > 1e-06)
            has_s = bool(y_np[b, 2].max() > 1e-06)
            pred_p = p_pred and p_conf >= sample_thr_p
            all_has_p.append(has_p)
            all_has_s.append(has_s)
            all_pred_p.append(pred_p)
            all_pred_s.append(False)
            if pred_p:
                p_center = p_idx
            elif has_p:
                p_center = int(np.argmax(p_prob))
            else:
                p_center = -1
            if s_pred:
                s_center = s_idx
            elif has_s:
                s_center = int(np.argmax(s_prob))
            else:
                s_center = -1
            unc_p = _local_unc(mi_map_np[b], p_center, UNC_WIN) if p_center >= 0 else float('nan')
            unc_s = _local_unc(mi_map_np[b], s_center, UNC_WIN) if s_center >= 0 else float('nan')
            all_unc_p.append(unc_p)
            all_unc_s.append(unc_s)
            u_comb = unc_np[b]
            if np.isfinite(unc_p) or np.isfinite(unc_s):
                u_comb = float(np.nanmax([unc_p, unc_s]))
            all_uncertainty.append(u_comb)
            all_p_conf.append(p_conf)
            all_s_conf.append(s_conf)
            thr_s_eff = sample_thr_s
            pred_s = s_pred
            if use_unc_gating_s and np.isfinite(unc_s):
                if unc_gating_k_s is not None and unc_gating_base_s is not None:
                    thr_s_eff = float(unc_gating_base_s + unc_gating_k_s * unc_s)
                if unc_gating_tau_s is not None and unc_s > float(unc_gating_tau_s):
                    pred_s = False
            pred_s = pred_s and s_conf >= thr_s_eff
            all_pred_s[-1] = pred_s
            if has_p:
                gt_p = int(np.argmax(y_np[b, 1]))
                err_p = abs(p_idx - gt_p)
                all_p_residual_signed.append(p_idx - gt_p)
                if p_pred and p_conf >= sample_thr_p:
                    all_p_ok.append(err_p <= tol)
                    all_p_err.append(err_p)
                else:
                    all_p_ok.append(False)
                    all_p_err.append(None)
            else:
                all_p_ok.append(None)
                all_p_err.append(None)
            if has_s:
                gt_s = int(np.argmax(y_np[b, 2]))
                err_s = abs(s_idx - gt_s)
                all_s_residual_signed.append(s_idx - gt_s)
                if pred_s:
                    all_s_ok.append(err_s <= tol)
                    all_s_err.append(err_s)
                else:
                    all_s_ok.append(False)
                    all_s_err.append(None)
            else:
                all_s_ok.append(None)
                all_s_err.append(None)
    n = len(all_uncertainty)
    error_type_p_list = []
    error_type_s_list = []
    error_flag_list = []
    for i in range(n):
        hp, pp, pok = (all_has_p[i], all_pred_p[i], all_p_ok[i])
        hs, ps, sok = (all_has_s[i], all_pred_s[i], all_s_ok[i])
        if not hp and pp:
            etp = 'FP'
        elif hp and (not pp):
            etp = 'FN'
        elif hp and pp and (pok is True):
            etp = 'TP'
        elif hp and pp and (pok is False):
            etp = 'FP'
        else:
            etp = 'TN'
        if not hs and ps:
            ets = 'FP'
        elif hs and (not ps):
            ets = 'FN'
        elif hs and ps and (sok is True):
            ets = 'TP'
        elif hs and ps and (sok is False):
            ets = 'FP'
        else:
            ets = 'TN'
        error_type_p_list.append(etp)
        error_type_s_list.append(ets)
        error_flag_list.append(1 if etp in ('FP', 'FN') or ets in ('FP', 'FN') else 0)

    def _metrics(has_p_list, pred_p_list, p_ok_list, has_s_list, pred_s_list, s_ok_list):
        p_tp = sum((1 for h, pred, ok in zip(has_p_list, pred_p_list, p_ok_list) if h and pred and (ok is True)))
        p_fp = sum((1 for h, pred in zip(has_p_list, pred_p_list) if not h and pred)) + sum((1 for h, pred, ok in zip(has_p_list, pred_p_list, p_ok_list) if h and pred and (ok is False)))
        p_fn = sum((1 for h, pred in zip(has_p_list, pred_p_list) if h and (not pred)))
        s_tp = sum((1 for h, pred, ok in zip(has_s_list, pred_s_list, s_ok_list) if h and pred and (ok is True)))
        s_fp = sum((1 for h, pred in zip(has_s_list, pred_s_list) if not h and pred)) + sum((1 for h, pred, ok in zip(has_s_list, pred_s_list, s_ok_list) if h and pred and (ok is False)))
        s_fn = sum((1 for h, pred in zip(has_s_list, pred_s_list) if h and (not pred)))

        def _prf(tp, fp, fn):
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
            return (p, r, f1)
        p_prec, p_rec, p_f1 = _prf(p_tp, p_fp, p_fn)
        s_prec, s_rec, s_f1 = _prf(s_tp, s_fp, s_fn)
        return dict(p_prec=p_prec, p_rec=p_rec, p_f1=p_f1, s_prec=s_prec, s_rec=s_rec, s_f1=s_f1, p_tp=p_tp, p_fp=p_fp, p_fn=p_fn, s_tp=s_tp, s_fp=s_fp, s_fn=s_fn)
    full_metrics = _metrics(all_has_p, all_pred_p, all_p_ok, all_has_s, all_pred_s, all_s_ok)
    unc_p_arr = np.array(all_unc_p, dtype=np.float64)
    unc_s_arr = np.array(all_unc_s, dtype=np.float64)
    has_p_arr = np.array(all_has_p, dtype=bool)
    has_s_arr = np.array(all_has_s, dtype=bool)
    pred_p_arr = np.array(all_pred_p, dtype=bool)
    pred_s_arr = np.array(all_pred_s, dtype=bool)
    try:
        print('[eval_detailed_mc_selective] unc_p std={:.4e}, unc_s std={:.4e}'.format(float(np.nanstd(unc_p_arr)), float(np.nanstd(unc_s_arr))))
    except Exception:
        pass

    def _selective_phase(phase: str):
        if phase == 'P':
            mask = (has_p_arr | pred_p_arr) & np.isfinite(unc_p_arr)
        else:
            mask = (has_s_arr | pred_s_arr) & np.isfinite(unc_s_arr)
        idx_all = np.where(mask)[0]
        if idx_all.size == 0:
            return (full_metrics['p_f1'] if phase == 'P' else full_metrics['s_f1'], full_metrics['p_prec'] if phase == 'P' else full_metrics['s_prec'], full_metrics['p_rec'] if phase == 'P' else full_metrics['s_rec'])
        unc_arr_ph = unc_p_arr if phase == 'P' else unc_s_arr
        sorted_idx = idx_all[np.argsort(unc_arr_ph[idx_all])]
        k_keep = max(1, int(len(idx_all) * (1.0 - drop_ratio)))
        keep_idx = sorted_idx[:k_keep]
        hp = [all_has_p[i] for i in keep_idx]
        pp = [all_pred_p[i] for i in keep_idx]
        pok = [all_p_ok[i] for i in keep_idx]
        hs = [all_has_s[i] for i in keep_idx]
        ps = [all_pred_s[i] for i in keep_idx]
        sok = [all_s_ok[i] for i in keep_idx]
        m = _metrics(hp, pp, pok, hs, ps, sok)
        return (m['p_f1'] if phase == 'P' else m['s_f1'], m['p_prec'] if phase == 'P' else m['s_prec'], m['p_rec'] if phase == 'P' else m['s_rec'])
    sel_p_f1, sel_p_prec, sel_p_rec = _selective_phase('P')
    sel_s_f1, sel_s_prec, sel_s_rec = _selective_phase('S')
    selective_metrics = dict(p_f1=sel_p_f1, p_prec=sel_p_prec, p_rec=sel_p_rec, s_f1=sel_s_f1, s_prec=sel_s_prec, s_rec=sel_s_rec)

    def _risk_coverage_phase(phase: str):
        if phase == 'P':
            mask = (has_p_arr | pred_p_arr) & np.isfinite(unc_p_arr)
        else:
            mask = (has_s_arr | pred_s_arr) & np.isfinite(unc_s_arr)
        idx_all = np.where(mask)[0]
        if idx_all.size == 0:
            return []
        unc_arr_ph = unc_p_arr if phase == 'P' else unc_s_arr
        sorted_idx = idx_all[np.argsort(unc_arr_ph[idx_all])]
        out = []
        for c in np.linspace(0.05, 1.0, coverage_points):
            k_keep = max(1, int(len(idx_all) * c))
            idx = sorted_idx[:k_keep]
            hp = [all_has_p[i] for i in idx]
            pp = [all_pred_p[i] for i in idx]
            pok = [all_p_ok[i] for i in idx]
            hs = [all_has_s[i] for i in idx]
            ps = [all_pred_s[i] for i in idx]
            sok = [all_s_ok[i] for i in idx]
            m = _metrics(hp, pp, pok, hs, ps, sok)
            risk = 1.0 - (m['p_f1'] if phase == 'P' else m['s_f1'])
            out.append((float(c), float(risk)))
        return out
    risk_coverage_p_unc = _risk_coverage_phase('P')
    risk_coverage_s_unc = _risk_coverage_phase('S')

    def _risk_coverage_phase_conf(phase: str):
        if phase == 'P':
            mask = (has_p_arr | pred_p_arr) & np.isfinite(unc_p_arr)
            conf_arr = np.array(all_p_conf, dtype=np.float64)
        else:
            mask = (has_s_arr | pred_s_arr) & np.isfinite(unc_s_arr)
            conf_arr = np.array(all_s_conf, dtype=np.float64)
        idx_all = np.where(mask)[0]
        if idx_all.size == 0:
            return []
        sorted_idx = idx_all[np.argsort(-conf_arr[idx_all])]
        out = []
        for c in np.linspace(0.05, 1.0, coverage_points):
            k_keep = max(1, int(len(idx_all) * c))
            idx = sorted_idx[:k_keep]
            hp = [all_has_p[i] for i in idx]
            pp = [all_pred_p[i] for i in idx]
            pok = [all_p_ok[i] for i in idx]
            hs = [all_has_s[i] for i in idx]
            ps = [all_pred_s[i] for i in idx]
            sok = [all_s_ok[i] for i in idx]
            m = _metrics(hp, pp, pok, hs, ps, sok)
            risk = 1.0 - (m['p_f1'] if phase == 'P' else m['s_f1'])
            out.append((float(c), float(risk)))
        return out
    rng_rc = np.random.default_rng(SEED)

    def _risk_coverage_phase_rand(phase: str):
        if phase == 'P':
            mask = (has_p_arr | pred_p_arr) & np.isfinite(unc_p_arr)
        else:
            mask = (has_s_arr | pred_s_arr) & np.isfinite(unc_s_arr)
        idx_all = np.where(mask)[0]
        if idx_all.size == 0:
            return []
        shuffled = rng_rc.permutation(idx_all)
        out = []
        for c in np.linspace(0.05, 1.0, coverage_points):
            k_keep = max(1, int(len(idx_all) * c))
            idx = shuffled[:k_keep]
            hp = [all_has_p[i] for i in idx]
            pp = [all_pred_p[i] for i in idx]
            pok = [all_p_ok[i] for i in idx]
            hs = [all_has_s[i] for i in idx]
            ps = [all_pred_s[i] for i in idx]
            sok = [all_s_ok[i] for i in idx]
            m = _metrics(hp, pp, pok, hs, ps, sok)
            risk = 1.0 - (m['p_f1'] if phase == 'P' else m['s_f1'])
            out.append((float(c), float(risk)))
        return out
    risk_coverage = {'P': risk_coverage_p_unc, 'S': risk_coverage_s_unc, 'P_conf': _risk_coverage_phase_conf('P'), 'S_conf': _risk_coverage_phase_conf('S'), 'P_rand': _risk_coverage_phase_rand('P'), 'S_rand': _risk_coverage_phase_rand('S')}
    if all_p_residual_signed:
        p_res_arr = np.asarray(all_p_residual_signed, dtype=np.float32) / float(SAMPLE_RATE)
        p_res_mean_sec = float(p_res_arr.mean())
        p_res_std_sec = float(p_res_arr.std())
        p_res_mae_sec = float(np.abs(p_res_arr).mean())
    else:
        p_res_mean_sec = p_res_std_sec = p_res_mae_sec = None
    if all_s_residual_signed:
        s_res_arr = np.asarray(all_s_residual_signed, dtype=np.float32) / float(SAMPLE_RATE)
        s_res_mean_sec = float(s_res_arr.mean())
        s_res_std_sec = float(s_res_arr.std())
        s_res_mae_sec = float(np.abs(s_res_arr).mean())
    else:
        s_res_mean_sec = s_res_std_sec = s_res_mae_sec = None

    def _fmt_res(v):
        return f'{float(v):.4f}' if v is not None else 'N/A'
    conf_single_p = np.array(all_p_conf_single, dtype=np.float64)
    conf_single_s = np.array(all_s_conf_single, dtype=np.float64)

    def _selective_phase_conf(phase: str):
        if phase == 'P':
            mask = (has_p_arr | pred_p_arr) & np.isfinite(unc_p_arr)
            conf_arr = conf_single_p
        else:
            mask = (has_s_arr | pred_s_arr) & np.isfinite(unc_s_arr)
            conf_arr = conf_single_s
        idx_all = np.where(mask)[0]
        if idx_all.size == 0:
            return (full_metrics['p_f1'] if phase == 'P' else full_metrics['s_f1'], full_metrics['p_prec'] if phase == 'P' else full_metrics['s_prec'], full_metrics['p_rec'] if phase == 'P' else full_metrics['s_rec'])
        sorted_idx = idx_all[np.argsort(-conf_arr[idx_all])]
        k_keep = max(1, int(len(idx_all) * (1.0 - drop_ratio)))
        keep_idx = sorted_idx[:k_keep]
        hp = [all_has_p[i] for i in keep_idx]
        pp = [all_pred_p[i] for i in keep_idx]
        pok = [all_p_ok[i] for i in keep_idx]
        hs = [all_has_s[i] for i in keep_idx]
        ps = [all_pred_s[i] for i in keep_idx]
        sok = [all_s_ok[i] for i in keep_idx]
        m = _metrics(hp, pp, pok, hs, ps, sok)
        return (m['p_f1'] if phase == 'P' else m['s_f1'], m['p_prec'] if phase == 'P' else m['s_prec'], m['p_rec'] if phase == 'P' else m['s_rec'])

    def _selective_phase_rand(phase: str):
        if phase == 'P':
            mask = (has_p_arr | pred_p_arr) & np.isfinite(unc_p_arr)
        else:
            mask = (has_s_arr | pred_s_arr) & np.isfinite(unc_s_arr)
        idx_all = np.where(mask)[0]
        if idx_all.size == 0:
            return (full_metrics['p_f1'] if phase == 'P' else full_metrics['s_f1'], full_metrics['p_prec'] if phase == 'P' else full_metrics['s_prec'], full_metrics['p_rec'] if phase == 'P' else full_metrics['s_rec'])
        shuffled = rng_rc.permutation(idx_all)
        k_keep = max(1, int(len(idx_all) * (1.0 - drop_ratio)))
        keep_idx = shuffled[:k_keep]
        hp = [all_has_p[i] for i in keep_idx]
        pp = [all_pred_p[i] for i in keep_idx]
        pok = [all_p_ok[i] for i in keep_idx]
        hs = [all_has_s[i] for i in keep_idx]
        ps = [all_pred_s[i] for i in keep_idx]
        sok = [all_s_ok[i] for i in keep_idx]
        m = _metrics(hp, pp, pok, hs, ps, sok)
        return (m['p_f1'] if phase == 'P' else m['s_f1'], m['p_prec'] if phase == 'P' else m['s_prec'], m['p_rec'] if phase == 'P' else m['s_rec'])
    conf_p_f1, conf_p_prec, conf_p_rec = _selective_phase_conf('P')
    conf_s_f1, conf_s_prec, conf_s_rec = _selective_phase_conf('S')
    rand_p_f1, rand_p_prec, rand_p_rec = _selective_phase_rand('P')
    rand_s_f1, rand_s_prec, rand_s_rec = _selective_phase_rand('S')
    if not quiet:
        print(f'[eval_detailed_mc_selective] MC T={mc_T}, drop_ratio={drop_ratio:.2%}', flush=True)
        print(f'[eval_detailed_mc_selective] Full      : P-Prec={full_metrics['p_prec']:.4f}, P-Rec={full_metrics['p_rec']:.4f}, P-F1={full_metrics['p_f1']:.4f}; S-Prec={full_metrics['s_prec']:.4f}, S-Rec={full_metrics['s_rec']:.4f}, S-F1={full_metrics['s_f1']:.4f} | p_res(mean/std/mae)s=({_fmt_res(p_res_mean_sec)}/{_fmt_res(p_res_std_sec)}/{_fmt_res(p_res_mae_sec)}), s_res=({_fmt_res(s_res_mean_sec)}/{_fmt_res(s_res_std_sec)}/{_fmt_res(s_res_mae_sec)})', flush=True)
        print(f'[eval_detailed_mc_selective] Selective: (drop top {drop_ratio:.0%} by uncertainty) P-Prec={selective_metrics['p_prec']:.4f}, P-Rec={selective_metrics['p_rec']:.4f}, P-F1={selective_metrics['p_f1']:.4f}; S-Prec={selective_metrics['s_prec']:.4f}, S-Rec={selective_metrics['s_rec']:.4f}, S-F1={selective_metrics['s_f1']:.4f} | p_res(mean/std/mae)s=({_fmt_res(p_res_mean_sec)}/{_fmt_res(p_res_std_sec)}/{_fmt_res(p_res_mae_sec)}), s_res=({_fmt_res(s_res_mean_sec)}/{_fmt_res(s_res_std_sec)}/{_fmt_res(s_res_mae_sec)})', flush=True)
        print(f'[eval_detailed_mc_selective] 1-peak  : (single-fwd peak conf) P-Prec={conf_p_prec:.4f}, P-Rec={conf_p_rec:.4f}, P-F1={conf_p_f1:.4f}; S-Prec={conf_s_prec:.4f}, S-Rec={conf_s_rec:.4f}, S-F1={conf_s_f1:.4f} | p_res(mean/std/mae)s=({_fmt_res(p_res_mean_sec)}/{_fmt_res(p_res_std_sec)}/{_fmt_res(p_res_mae_sec)}), s_res=({_fmt_res(s_res_mean_sec)}/{_fmt_res(s_res_std_sec)}/{_fmt_res(s_res_mae_sec)})', flush=True)
        print(f'[eval_detailed_mc_selective] Random  : P-Prec={rand_p_prec:.4f}, P-Rec={rand_p_rec:.4f}, P-F1={rand_p_f1:.4f}; S-Prec={rand_s_prec:.4f}, S-Rec={rand_s_rec:.4f}, S-F1={rand_s_f1:.4f} | p_res(mean/std/mae)s=({_fmt_res(p_res_mean_sec)}/{_fmt_res(p_res_std_sec)}/{_fmt_res(p_res_mae_sec)}), s_res=({_fmt_res(s_res_mean_sec)}/{_fmt_res(s_res_std_sec)}/{_fmt_res(s_res_mae_sec)})', flush=True)
    time_acc = time_correct / max(1, n_time)
    return dict(time_acc=time_acc, p_prec=full_metrics['p_prec'], p_rec=full_metrics['p_rec'], p_f1=full_metrics['p_f1'], s_prec=full_metrics['s_prec'], s_rec=full_metrics['s_rec'], s_f1=full_metrics['s_f1'], p_res_mean_sec=p_res_mean_sec, p_res_std_sec=p_res_std_sec, p_res_mae_sec=p_res_mae_sec, s_res_mean_sec=s_res_mean_sec, s_res_std_sec=s_res_std_sec, s_res_mae_sec=s_res_mae_sec, selective_p_f1=selective_metrics['p_f1'], selective_s_f1=selective_metrics['s_f1'], selective_p_prec=selective_metrics['p_prec'], selective_p_rec=selective_metrics['p_rec'], selective_s_prec=selective_metrics['s_prec'], selective_s_rec=selective_metrics['s_rec'], conf_p_f1=conf_p_f1, conf_s_f1=conf_s_f1, conf_p_prec=conf_p_prec, conf_p_rec=conf_p_rec, conf_s_prec=conf_s_prec, conf_s_rec=conf_s_rec, rand_p_f1=rand_p_f1, rand_s_f1=rand_s_f1, rand_p_prec=rand_p_prec, rand_p_rec=rand_p_rec, rand_s_prec=rand_s_prec, rand_s_rec=rand_s_rec, risk_coverage=risk_coverage, mc_uncertainty=all_uncertainty, mc_error_flag=error_flag_list, mc_error_type_p=error_type_p_list, mc_error_type_s=error_type_s_list, p_residuals_signed=all_p_residual_signed, s_residuals_signed=all_s_residual_signed)

@torch.inference_mode()
def compute_risk_coverage_confidence_non_mc(model: nn.Module, loader, device: torch.device, thr_p: float, thr_s: float, coverage_points: int=20, tol: int=DYN_TOL_SAMPLES) -> dict[str, list[tuple[float, float]]]:
    model.eval()
    all_has_p, all_has_s = ([], [])
    all_pred_p, all_pred_s = ([], [])
    all_p_ok, all_s_ok = ([], [])
    all_p_conf, all_s_conf = ([], [])
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_np = y.cpu().numpy()
        B = probs.shape[0]
        for b in range(B):
            p_prob = probs[b, 1]
            s_prob = probs[b, 2]
            has_p = bool(y_np[b, 1].max() > 1e-06)
            has_s = bool(y_np[b, 2].max() > 1e-06)
            p_idx = int(np.argmax(p_prob))
            s_idx = int(np.argmax(s_prob))
            p_conf = float(p_prob[p_idx])
            s_conf = float(s_prob[s_idx])
            all_has_p.append(has_p)
            all_has_s.append(has_s)
            all_p_conf.append(p_conf)
            all_s_conf.append(s_conf)
            if has_p:
                gt_p = int(np.argmax(y_np[b, 1]))
                err_p = abs(p_idx - gt_p)
                pred_p = p_conf >= thr_p
                all_pred_p.append(pred_p)
                if pred_p:
                    all_p_ok.append(err_p <= tol)
                else:
                    all_p_ok.append(False)
            else:
                pred_p = p_conf >= thr_p
                all_pred_p.append(pred_p)
                all_p_ok.append(None)
            if has_s:
                gt_s = int(np.argmax(y_np[b, 2]))
                err_s = abs(s_idx - gt_s)
                pred_s = s_conf >= thr_s
                all_pred_s.append(pred_s)
                if pred_s:
                    all_s_ok.append(err_s <= tol)
                else:
                    all_s_ok.append(False)
            else:
                pred_s = s_conf >= thr_s
                all_pred_s.append(pred_s)
                all_s_ok.append(None)
    has_p_arr = np.array(all_has_p, dtype=bool)
    has_s_arr = np.array(all_has_s, dtype=bool)
    pred_p_arr = np.array(all_pred_p, dtype=bool)
    pred_s_arr = np.array(all_pred_s, dtype=bool)
    p_conf_arr = np.array(all_p_conf, dtype=np.float64)
    s_conf_arr = np.array(all_s_conf, dtype=np.float64)

    def _metrics(has_p_list, pred_p_list, p_ok_list, has_s_list, pred_s_list, s_ok_list):
        p_tp = sum((1 for h, pred, ok in zip(has_p_list, pred_p_list, p_ok_list) if h and pred and (ok is True)))
        p_fp = sum((1 for h, pred in zip(has_p_list, pred_p_list) if not h and pred)) + sum((1 for h, pred, ok in zip(has_p_list, pred_p_list, p_ok_list) if h and pred and (ok is False)))
        p_fn = sum((1 for h, pred in zip(has_p_list, pred_p_list) if h and (not pred)))
        s_tp = sum((1 for h, pred, ok in zip(has_s_list, pred_s_list, s_ok_list) if h and pred and (ok is True)))
        s_fp = sum((1 for h, pred in zip(has_s_list, pred_s_list) if not h and pred)) + sum((1 for h, pred, ok in zip(has_s_list, pred_s_list, s_ok_list) if h and pred and (ok is False)))
        s_fn = sum((1 for h, pred in zip(has_s_list, pred_s_list) if h and (not pred)))

        def _prf(tp, fp, fn):
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
            return (p, r, f1)
        p_prec, p_rec, p_f1 = _prf(p_tp, p_fp, p_fn)
        s_prec, s_rec, s_f1 = _prf(s_tp, s_fp, s_fn)
        return dict(p_prec=p_prec, p_rec=p_rec, p_f1=p_f1, s_prec=s_prec, s_rec=s_rec, s_f1=s_f1)

    def _risk_coverage_phase_conf(phase: str) -> list[tuple[float, float]]:
        if phase == 'P':
            mask = has_p_arr | pred_p_arr
            conf_arr = p_conf_arr
        else:
            mask = has_s_arr | pred_s_arr
            conf_arr = s_conf_arr
        idx_all = np.where(mask)[0]
        if idx_all.size == 0:
            return []
        sorted_idx = idx_all[np.argsort(-conf_arr[idx_all])]
        out = []
        for c in np.linspace(0.05, 1.0, coverage_points):
            k_keep = max(1, int(len(idx_all) * c))
            idx = sorted_idx[:k_keep]
            hp = [all_has_p[i] for i in idx]
            pp = [all_pred_p[i] for i in idx]
            pok = [all_p_ok[i] for i in idx]
            hs = [all_has_s[i] for i in idx]
            ps = [all_pred_s[i] for i in idx]
            sok = [all_s_ok[i] for i in idx]
            m = _metrics(hp, pp, pok, hs, ps, sok)
            risk = 1.0 - (m['p_f1'] if phase == 'P' else m['s_f1'])
            out.append((float(c), float(risk)))
        return out
    return {'P_conf': _risk_coverage_phase_conf('P'), 'S_conf': _risk_coverage_phase_conf('S')}

@torch.inference_mode()
def compute_uncertainty_confidence_non_mc(model: nn.Module, loader, device: torch.device) -> list[float]:
    model.eval()
    unc_conf_list: list[float] = []
    for x, _y, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        B = probs.shape[0]
        for b in range(B):
            max_prob = float(probs[b].max())
            unc_conf_list.append(1.0 - max_prob)
    return unc_conf_list

def collect_pca_features_and_labels(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats: list[list[float]] = []
    labs: list[int] = []
    max_per_class = 1000
    counts = {0: 0, 1: 0, 2: 0}
    win = 50

    def _add_feat(feat: np.ndarray, label: int) -> None:
        if counts[label] >= max_per_class:
            return
        feats.append(feat.astype(np.float32).tolist())
        labs.append(label)
        counts[label] += 1
    with torch.no_grad():
        for x, y, _ in loader:
            if all((c >= max_per_class for c in counts.values())):
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if y.shape[-1] != logits.shape[-1]:
                y = _center_crop_time(y, logits.shape[-1])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_np = y.cpu().numpy()
            B, C, T = probs.shape
            for b in range(B):
                if all((c >= max_per_class for c in counts.values())):
                    break
                prob_ch = probs[b]
                y_b = y_np[b]
                has_p = bool(y_b[1].max() > 1e-06)
                has_s = bool(y_b[2].max() > 1e-06)

                def _window_feat(center: int) -> np.ndarray:
                    lo = max(0, center - win)
                    hi = min(T, center + win + 1)
                    w = prob_ch[:, lo:hi]
                    if w.shape[1] == 0:
                        return np.zeros(6, dtype=np.float32)
                    mean = w.mean(axis=1)
                    std = w.std(axis=1) + 1e-08
                    return np.concatenate([mean, std], axis=0)
                if has_p and counts[1] < max_per_class:
                    p_idx = int(np.argmax(y_b[1]))
                    feat_p = _window_feat(p_idx)
                    _add_feat(feat_p, 1)
                if has_s and counts[2] < max_per_class:
                    s_idx = int(np.argmax(y_b[2]))
                    feat_s = _window_feat(s_idx)
                    _add_feat(feat_s, 2)
                if counts[0] < max_per_class:
                    bg_mask = (y_b[1] < 1e-06) & (y_b[2] < 1e-06)
                    cand_idx = np.where(bg_mask)[0]
                    if cand_idx.size > 0:
                        if has_p:
                            p_idx = int(np.argmax(y_b[1]))
                            cand_idx = cand_idx[np.abs(cand_idx - p_idx) >= win]
                        if has_s and cand_idx.size > 0:
                            s_idx = int(np.argmax(y_b[2]))
                            cand_idx = cand_idx[np.abs(cand_idx - s_idx) >= win]
                    if cand_idx.size > 0:
                        center = int(np.random.default_rng(2024).choice(cand_idx))
                        feat_bg = _window_feat(center)
                        _add_feat(feat_bg, 0)
    if not feats:
        return (np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int64))
    return (np.asarray(feats, dtype=np.float32), np.asarray(labs, dtype=np.int64))

def collect_pr_snr_data(model, loader, device, tol_samples: int=DYN_TOL_SAMPLES) -> dict:
    model.eval()
    has_p, has_s = ([], [])
    max_prob_p, max_prob_s = ([], [])
    p_ok, s_ok = ([], [])
    snr_list = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if y.shape[-1] != logits.shape[-1]:
                y = _center_crop_time(y, logits.shape[-1])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_np = y.cpu().numpy()
            x_np = x.cpu().numpy()
            B = probs.shape[0]
            for b in range(B):
                prob_p = probs[b, 1]
                prob_s = probs[b, 2]
                mp_p = float(np.max(prob_p))
                mp_s = float(np.max(prob_s))
                p_idx = int(np.argmax(prob_p))
                s_idx = int(np.argmax(prob_s))
                hp = bool(y_np[b, 1].max() > 1e-06)
                hs = bool(y_np[b, 2].max() > 1e-06)
                p_gt = int(np.argmax(y_np[b, 1])) if hp else None
                s_gt = int(np.argmax(y_np[b, 2])) if hs else None
                po = hp and p_gt is not None and (abs(p_idx - p_gt) <= tol_samples)
                so = hs and s_gt is not None and (abs(s_idx - s_gt) <= tol_samples)
                snr_val = estimate_snr(x_np[b], p_idx=p_gt)
                has_p.append(hp)
                has_s.append(hs)
                max_prob_p.append(mp_p)
                max_prob_s.append(mp_s)
                p_ok.append(po)
                s_ok.append(so)
                snr_list.append(snr_val)
    return {'has_p': has_p, 'has_s': has_s, 'max_prob_p': max_prob_p, 'max_prob_s': max_prob_s, 'p_ok': p_ok, 's_ok': s_ok, 'snr': snr_list}

@torch.inference_mode()
def save_visuals(model, dataset, device, out_dir: str, n: int=4):
    os.makedirs(out_dir, exist_ok=True)
    idxs = list(range(len(dataset)))
    random.seed(SEED)
    random.shuffle(idxs)
    idxs = idxs[:n]
    for idx in idxs:
        x_t, y_t, name = dataset[idx]
        x = x_t.unsqueeze(0).to(device)
        y = y_t.unsqueeze(0).to(device)
        logits = model(x)
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        y_np = y.squeeze(0).cpu().numpy()
        x_np = x_t.cpu().numpy()
        p_idx = int(np.argmax(y_np[1])) if y_np[1].max() > 1e-06 else None
        snr = estimate_snr(x_np, p_idx=p_idx)
        plot_one_sample_visual(x_np, y_np, probs, name, snr, os.path.join(out_dir, f'vis_{idx}.png'))

@torch.inference_mode()
def save_representative_waveforms_2x2(model: nn.Module, valid_loader: DataLoader, device: torch.device, out_path: str, tol: int, thr_p: float=0.5, thr_s: float=0.5, max_scan_samples: int=2000, snr_high: float=15.0, snr_low: float=5.0, ch_missing_std_eps: float=1e-06, ch_missing_max_eps: float=1e-06, extra_loaders: list[DataLoader] | None=None, rep_npz_path: str | None=None) -> None:
    if plt is None:
        return
    model.eval()

    def _has_phase(y_np_b: np.ndarray, cls: int) -> bool:
        return bool(y_np_b[cls].max() > 1e-06)

    def _phase_gt_idx(y_np_b: np.ndarray, cls: int) -> int:
        return int(np.argmax(y_np_b[cls]))

    def _phase_pred_idx_conf(prob_b: np.ndarray, cls: int) -> tuple[int, float]:
        idx = int(np.argmax(prob_b[cls]))
        return (idx, float(prob_b[cls, idx]))

    def _channel_missing_mask(x_np_b: np.ndarray) -> np.ndarray:
        stds = np.std(x_np_b, axis=1)
        maxabs = np.max(np.abs(x_np_b), axis=1)
        return (stds <= ch_missing_std_eps) | (maxabs <= ch_missing_max_eps)
    chosen: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str, float]] = {}
    chosen_name: dict[str, str] = {}
    seen_names: set[str] = set()
    scanned = 0
    loaders: list[DataLoader] = [valid_loader]
    if extra_loaders:
        loaders.extend(extra_loaders)
    for loader in loaders:
        for x, y, names in loader:
            if scanned >= max_scan_samples:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if y.shape[-1] != logits.shape[-1]:
                y = _center_crop_time(y, logits.shape[-1])
            probs = torch.softmax(logits, dim=1)
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            B = x_np.shape[0]
            for b in range(B):
                if scanned >= max_scan_samples:
                    break
                scanned += 1
                name_b = str(names[b]) if hasattr(names, '__getitem__') else str(scanned)
                if name_b in seen_names:
                    continue
                has_p = _has_phase(y_np[b], 1)
                has_s = _has_phase(y_np[b], 2)
                if not (has_p and has_s):
                    continue
                p_gt = _phase_gt_idx(y_np[b], 1)
                snr_b = float(estimate_snr(x_np[b], p_idx=p_gt))
                p_pred_idx, p_conf = _phase_pred_idx_conf(probs_np[b], 1)
                s_pred_idx, s_conf = _phase_pred_idx_conf(probs_np[b], 2)
                p_pass = p_conf >= float(thr_p)
                s_pass = s_conf >= float(thr_s)
                p_ok = p_pass and abs(p_pred_idx - p_gt) <= int(tol)
                s_gt = _phase_gt_idx(y_np[b], 2)
                s_ok = s_pass and abs(s_pred_idx - s_gt) <= int(tol)
                miss_mask = _channel_missing_mask(x_np[b])
                has_missing = bool(np.any(miss_mask))
                if 'missing' not in chosen and has_missing and p_ok and s_ok:
                    chosen['missing'] = (x_np[b], y_np[b], probs_np[b], 'Channel missing', snr_b)
                    chosen_name['missing'] = name_b
                    seen_names.add(name_b)
                    continue
                if p_ok and s_ok:
                    if snr_b >= snr_high and 'normal' not in chosen:
                        chosen['normal'] = (x_np[b], y_np[b], probs_np[b], 'Normal SNR', snr_b)
                        chosen_name['normal'] = name_b
                        seen_names.add(name_b)
                        continue
                    if snr_low <= snr_b < snr_high and 'low' not in chosen:
                        chosen['low'] = (x_np[b], y_np[b], probs_np[b], 'Low SNR', snr_b)
                        chosen_name['low'] = name_b
                        seen_names.add(name_b)
                        continue
                    if snr_b < snr_low and 'very_low' not in chosen:
                        chosen['very_low'] = (x_np[b], y_np[b], probs_np[b], 'Very low SNR', snr_b)
                        chosen_name['very_low'] = name_b
                        seen_names.add(name_b)
                        continue
            if len(chosen) >= 4 and all((k in chosen for k in ('normal', 'low', 'very_low', 'missing'))):
                break
        if len(chosen) >= 4 and all((k in chosen for k in ('normal', 'low', 'very_low', 'missing'))):
            break
    if 'missing' not in chosen and 'normal' in chosen:
        x0, y0, _p0, _t0, _snr0 = chosen['normal']
        x_syn = np.array(x0, copy=True)
        if x_syn.shape[0] >= 3:
            x_syn[1, :] = 0.0
            x_syn[2, :] = 0.0
        elif x_syn.shape[0] >= 2:
            x_syn[1, :] = 0.0
        x_t = torch.from_numpy(x_syn).unsqueeze(0).to(device)
        logits = model(x_t)
        probs_syn = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        p_gt0 = int(np.argmax(y0[1])) if y0[1].max() > 1e-06 else None
        snr_syn = float(estimate_snr(x_syn, p_idx=p_gt0))
        chosen['missing'] = (x_syn, y0, probs_syn, 'Channel missing', snr_syn)
        chosen_name.setdefault('missing', 'synthetic_missing_from_' + chosen_name.get('normal', 'unknown'))

    def _fallback_pick(keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float] | None:
        for k in keys:
            if k in chosen:
                return chosen[k]
        return None
    s_normal = _fallback_pick(['normal', 'low', 'very_low', 'missing'])
    s_low = _fallback_pick(['low', 'normal', 'very_low', 'missing'])
    s_very = _fallback_pick(['very_low', 'low', 'normal', 'missing'])
    s_miss = _fallback_pick(['missing', 'normal', 'low', 'very_low'])
    if not (s_normal and s_low and s_very and s_miss):
        return
    samples = [s_normal, s_low, s_very, s_miss]
    plot_representative_waveforms_grid(samples, out_path=out_path)
    if rep_npz_path:
        try:
            exclude_names = set(chosen_name.values())
            chosen2: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str, float]] = {}
            seen_names2: set[str] = set(exclude_names)
            scanned2 = 0
            for x, y, names in valid_loader:
                if scanned2 >= max_scan_samples:
                    break
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                if y.shape[-1] != logits.shape[-1]:
                    y = _center_crop_time(y, logits.shape[-1])
                probs = torch.softmax(logits, dim=1)
                x_np = x.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                probs_np = probs.detach().cpu().numpy()
                B = x_np.shape[0]
                for b in range(B):
                    if scanned2 >= max_scan_samples:
                        break
                    scanned2 += 1
                    name_b = str(names[b]) if hasattr(names, '__getitem__') else f'sample_{scanned2}'
                    if name_b in seen_names2:
                        continue
                    has_p = _has_phase(y_np[b], 1)
                    has_s = _has_phase(y_np[b], 2)
                    if not (has_p and has_s):
                        continue
                    p_gt = _phase_gt_idx(y_np[b], 1)
                    snr_b = float(estimate_snr(x_np[b], p_idx=p_gt))
                    p_pred_idx, p_conf = _phase_pred_idx_conf(probs_np[b], 1)
                    s_pred_idx, s_conf = _phase_pred_idx_conf(probs_np[b], 2)
                    p_pass = p_conf >= float(thr_p)
                    s_pass = s_conf >= float(thr_s)
                    p_ok = p_pass and abs(p_pred_idx - p_gt) <= int(tol)
                    s_gt = _phase_gt_idx(y_np[b], 2)
                    s_ok = s_pass and abs(s_pred_idx - s_gt) <= int(tol)
                    miss_mask = _channel_missing_mask(x_np[b])
                    has_missing = bool(np.any(miss_mask))
                    if 'missing' not in chosen2 and has_missing and p_ok and s_ok:
                        chosen2['missing'] = (x_np[b], y_np[b], probs_np[b], 'Channel missing', snr_b)
                        seen_names2.add(name_b)
                        continue
                    if p_ok and s_ok:
                        if snr_b >= snr_high and 'normal' not in chosen2:
                            chosen2['normal'] = (x_np[b], y_np[b], probs_np[b], 'Normal SNR', snr_b)
                            seen_names2.add(name_b)
                            continue
                        if snr_low <= snr_b < snr_high and 'low' not in chosen2:
                            chosen2['low'] = (x_np[b], y_np[b], probs_np[b], 'Low SNR', snr_b)
                            seen_names2.add(name_b)
                            continue
                        if snr_b < snr_low and 'very_low' not in chosen2:
                            chosen2['very_low'] = (x_np[b], y_np[b], probs_np[b], 'Very low SNR', snr_b)
                            seen_names2.add(name_b)
                            continue
                if len(chosen2) >= 4 and all((k in chosen2 for k in ('normal', 'low', 'very_low', 'missing'))):
                    break
            if 'missing' not in chosen2 and 'normal' in chosen2:
                x0, y0, _p0, _t0, _snr0 = chosen2['normal']
                x_syn = np.array(x0, copy=True)
                if x_syn.shape[0] >= 3:
                    x_syn[1, :] = 0.0
                    x_syn[2, :] = 0.0
                elif x_syn.shape[0] >= 2:
                    x_syn[1, :] = 0.0
                x_t = torch.from_numpy(x_syn).unsqueeze(0).to(device)
                logits = model(x_t)
                probs_syn = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
                p_gt0 = int(np.argmax(y0[1])) if y0[1].max() > 1e-06 else None
                snr_syn = float(estimate_snr(x_syn, p_idx=p_gt0))
                chosen2['missing'] = (x_syn, y0, probs_syn, 'Channel missing', snr_syn)

            def _fallback_pick2(keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float] | None:
                for k in keys:
                    if k in chosen2:
                        return chosen2[k]
                return None
            s_normal2 = _fallback_pick2(['normal', 'low', 'very_low', 'missing'])
            s_low2 = _fallback_pick2(['low', 'normal', 'very_low', 'missing'])
            s_very2 = _fallback_pick2(['very_low', 'low', 'normal', 'missing'])
            s_miss2 = _fallback_pick2(['missing', 'normal', 'low', 'very_low'])
            if not (s_normal2 and s_low2 and s_very2 and s_miss2):
                print('[save_representative_waveforms_2x2] representative_compare.npz: 未能找到完整的第二批样本，跳过导出', flush=True)
                return
            samples2 = [s_normal2, s_low2, s_very2, s_miss2]
            x_list = np.stack([s[0] for s in samples2], axis=0)
            y_list = np.stack([s[1] for s in samples2], axis=0)
            snr_list = np.array([float(s[4]) for s in samples2], dtype=float)
            name_list = np.array(['normal', 'low', 'very_low', 'missing'], dtype=object)
            np.savez(rep_npz_path, x_list=x_list, y_list=y_list, snr_list=snr_list, names=name_list)
            print(f'[save_representative_waveforms_2x2] 代表性样本（第二批）已保存: {rep_npz_path}', flush=True)
        except Exception as e:
            print(f'[save_representative_waveforms_2x2] 保存 representative_compare.npz 失败: {e}', flush=True)

def _compute_error_label(pass_mask: bool, ok_mask: bool, has_label: bool) -> str:
    if not has_label:
        return 'FP' if pass_mask else 'TN'
    if pass_mask and ok_mask:
        return 'TP'
    if pass_mask and (not ok_mask):
        return 'FP'
    if not pass_mask:
        return 'FN'
    return 'FP'

@torch.inference_mode()
def collect_threshold_visualization_data(model, loader, device, n_samples: int=200, uncertainty_mode: str='entropy', uncertainty_threshold_options: dict | None=None, tol_samples: int=DYN_TOL_SAMPLES):
    if plt is None:
        return None
    thr_opts = _get_uncertainty_threshold_kwargs(uncertainty_threshold_options)
    time_win = thr_opts.get('time_window', 0)
    aggregate = thr_opts.get('aggregate', 'mean')
    threshold_kwargs = {k: v for k, v in thr_opts.items() if k not in ('time_window', 'aggregate')}
    opts = uncertainty_threshold_options or {}
    fixed_thr_p = float(opts.get('fixed_thr_p', 0.5))
    fixed_thr_s = float(opts.get('fixed_thr_s', 0.5))
    model.eval()
    all_data = {'thr_p': [], 'thr_s': [], 'uncertainty': [], 'snr': [], 'signal_strength': [], 'noise_std': [], 'max_prob_p': [], 'max_prob_s': [], 'pass_p': [], 'pass_s': [], 'p_error': [], 's_error': [], 'p_error_fixed': [], 's_error_fixed': [], 'p_idx': [], 's_idx': [], 'p_gt': [], 's_gt': []}
    count = 0
    opts = uncertainty_threshold_options or {}
    stable_dynamic = bool(opts.get('stable_dynamic', False))
    stable_state: dict | None = None
    for x, y, _ in loader:
        if count >= n_samples:
            break
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if y.shape[-1] != logits.shape[-1]:
            y = _center_crop_time(y, logits.shape[-1])
        probs = torch.softmax(logits, dim=1)
        uncertainty = compute_uncertainty_from_probs(probs, mode=uncertainty_mode, time_window=time_win, aggregate=aggregate)
        if uncertainty.dim() == 3:
            uncertainty = uncertainty.mean(dim=-1)
        if stable_dynamic:
            suggested_thr, stable_state = _stable_dynamic_threshold(uncertainty, stable_state, opts)
        else:
            suggested_thr = uncertainty_to_threshold(uncertainty, **threshold_kwargs)
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        probs_np = probs.cpu().numpy()
        uncertainty_np = uncertainty.cpu().numpy()
        suggested_thr_np = suggested_thr.cpu().numpy()
        B = x.shape[0]
        for b in range(B):
            if count >= n_samples:
                break
            prob_p = float(probs_np[b, 1, :].max())
            prob_s = float(probs_np[b, 2, :].max())
            p_idx = int(np.argmax(probs_np[b, 1, :]))
            s_idx = int(np.argmax(probs_np[b, 2, :]))
            has_p = bool(y_np[b, 1].max() > 1e-06)
            has_s = bool(y_np[b, 2].max() > 1e-06)
            p_gt = int(np.argmax(y_np[b, 1])) if has_p else None
            s_gt = int(np.argmax(y_np[b, 2])) if has_s else None
            p_ok = has_p and p_gt is not None and (abs(p_idx - p_gt) <= tol_samples)
            s_ok = has_s and s_gt is not None and (abs(s_idx - s_gt) <= tol_samples)
            thr_p_val = float(suggested_thr_np[b, 0])
            thr_s_val = float(suggested_thr_np[b, 1])
            pass_p = prob_p >= thr_p_val
            pass_s = prob_s >= thr_s_val
            pass_p_fixed = prob_p >= fixed_thr_p
            pass_s_fixed = prob_s >= fixed_thr_s
            p_err = _compute_error_label(pass_p, p_ok, has_p)
            s_err = _compute_error_label(pass_s, s_ok, has_s)
            p_err_fixed = _compute_error_label(pass_p_fixed, p_ok, has_p)
            s_err_fixed = _compute_error_label(pass_s_fixed, s_ok, has_s)
            all_data['thr_p'].append(thr_p_val)
            all_data['thr_s'].append(thr_s_val)
            all_data['uncertainty'].append(float(uncertainty_np[b].mean()))
            all_data['snr'].append(estimate_snr(x_np[b], p_idx=int(p_gt) if p_gt is not None else None))
            all_data['signal_strength'].append(prob_p)
            all_data['noise_std'].append(prob_s)
            all_data['max_prob_p'].append(prob_p)
            all_data['max_prob_s'].append(prob_s)
            all_data['pass_p'].append(pass_p)
            all_data['pass_s'].append(pass_s)
            all_data['p_idx'].append(p_idx)
            all_data['s_idx'].append(s_idx)
            all_data['p_gt'].append(p_gt)
            all_data['s_gt'].append(s_gt)
            all_data['p_error'].append(p_err)
            all_data['s_error'].append(s_err)
            all_data['p_error_fixed'].append(p_err_fixed)
            all_data['s_error_fixed'].append(s_err_fixed)
            count += 1
    if count == 0:
        print('Warning: No samples collected for threshold visualization')
        return None
    return all_data

def _metrics_vectorized(err_p: np.ndarray, err_s: np.ndarray, ph: np.ndarray, sh: np.ndarray, idx: np.ndarray) -> dict:
    tp = int((ph[idx] & (err_p[idx] == 'TP')).sum() + (sh[idx] & (err_s[idx] == 'TP')).sum())
    fp = int((err_p[idx] == 'FP').sum() + (err_s[idx] == 'FP').sum())
    fn = int((ph[idx] & (err_p[idx] == 'FN')).sum() + (sh[idx] & (err_s[idx] == 'FN')).sum())
    prec = tp / (tp + fp + 1e-08)
    rec = tp / (tp + fn + 1e-08)
    f1 = 2 * prec * rec / (prec + rec + 1e-08)
    return {'precision': prec, 'recall': rec, 'f1': f1, 'fp': fp, 'fn': fn}

def _error_rate_channel(errs: np.ndarray, has_label: np.ndarray) -> np.ndarray:
    out = np.zeros(len(errs), dtype=float)
    mask = ((errs == 'FP') | (errs == 'FN')) & has_label
    out[mask] = 1.0
    return out

def collect_dynamic_effect_data(threshold_data: dict, fixed_thr_p: float, fixed_thr_s: float, high_unc_percent: float=0.2, n_bins: int=10) -> dict | None:
    if not threshold_data or 'p_error_fixed' not in threshold_data:
        return None
    unc = np.array(threshold_data['uncertainty'])
    thr_p = np.array(threshold_data['thr_p'])
    thr_s = np.array(threshold_data['thr_s'])
    prob_p = np.array(threshold_data['max_prob_p'])
    prob_s = np.array(threshold_data['max_prob_s'])
    p_has = np.array([gt is not None for gt in threshold_data['p_gt']])
    s_has = np.array([gt is not None for gt in threshold_data['s_gt']])
    p_err = np.array(threshold_data['p_error'])
    s_err = np.array(threshold_data['s_error'])
    p_err_f = np.array(threshold_data['p_error_fixed'])
    s_err_f = np.array(threshold_data['s_error_fixed'])
    n = len(unc)
    denom = p_has.astype(float) + s_has.astype(float) + 1e-08
    fixed_err = (_error_rate_channel(p_err_f, p_has) + _error_rate_channel(s_err_f, s_has)) / denom
    dynamic_err = (_error_rate_channel(p_err, p_has) + _error_rate_channel(s_err, s_has)) / denom
    bins = np.linspace(unc.min(), min(unc.max() + 1e-09, 1.0), n_bins + 1)
    bin_centers = []
    fixed_error_rate = []
    dynamic_error_rate = []
    for i in range(n_bins):
        mask = (unc >= bins[i]) & (unc < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
        fixed_error_rate.append(float(fixed_err[mask].mean()))
        dynamic_error_rate.append(float(dynamic_err[mask].mean()))
    sorted_idx = np.argsort(unc)
    high_start = int((1.0 - high_unc_percent) * n)
    high_idx = sorted_idx[high_start:] if high_start < n else sorted_idx[-1:]
    pass_p_fixed = prob_p >= fixed_thr_p
    pass_s_fixed = prob_s >= fixed_thr_s
    pass_p_dyn = prob_p >= thr_p
    pass_s_dyn = prob_s >= thr_s
    changed = (pass_p_fixed != pass_p_dyn) | (pass_s_fixed != pass_s_dyn)
    high_mask = np.zeros(n, dtype=bool)
    high_mask[high_idx] = True
    fp_suppressed = int(((p_err_f == 'FP') & (p_err != 'FP') & high_mask).sum() + ((s_err_f == 'FP') & (s_err != 'FP') & high_mask).sum())
    fn_introduced = int(((p_err_f != 'FN') & (p_err == 'FN') & high_mask).sum() + ((s_err_f != 'FN') & (s_err == 'FN') & high_mask).sum())
    data = {'uncertainty_bins': np.array(bin_centers) if bin_centers else np.array([]), 'fixed_error_rate': np.array(fixed_error_rate) if fixed_error_rate else np.array([]), 'dynamic_error_rate': np.array(dynamic_error_rate) if dynamic_error_rate else np.array([]), 'high_unc_fixed_metrics': _metrics_vectorized(p_err_f, s_err_f, p_has, s_has, high_idx), 'high_unc_dynamic_metrics': _metrics_vectorized(p_err, s_err, p_has, s_has, high_idx), 'threshold_shift': (np.abs(thr_p - fixed_thr_p) + np.abs(thr_s - fixed_thr_s)) / 2.0, 'uncertainties': unc, 'decision_change_ratio': float(np.mean(changed)), 'decision_change_high_unc_ratio': float(changed[high_idx].mean()) if len(high_idx) > 0 else 0.0, 'high_unc_fp_suppressed': fp_suppressed, 'high_unc_fn_introduced': fn_introduced, 'high_unc_n': len(high_idx)}
    return data

def visualize_threshold_distribution(model, loader, device, out_dir: str, n_samples: int=200, use_uncertainty_threshold: bool=True, uncertainty_mode: str='entropy', uncertainty_threshold_options: dict=None, threshold_data: dict | None=None):
    if threshold_data is None:
        threshold_data = collect_threshold_visualization_data(model, loader, device, n_samples=n_samples, uncertainty_mode=uncertainty_mode, uncertainty_threshold_options=uncertainty_threshold_options)
    if not threshold_data:
        return None
    snr_array = np.array(threshold_data['snr'])
    thr_p_array = np.array(threshold_data['thr_p'])
    thr_s_array = np.array(threshold_data['thr_s'])
    print('\n=== Threshold Statistics by SNR Range ===')
    snr_bins = [0, 10, 15, 20, 25, 100]
    snr_labels = ['0-10', '10-15', '15-20', '20-25', '25+']
    for i in range(len(snr_bins) - 1):
        mask = (snr_array >= snr_bins[i]) & (snr_array < snr_bins[i + 1])
        if mask.sum() > 0:
            print(f'\nSNR {snr_labels[i]} dB (n={mask.sum()}): P thr mean={thr_p_array[mask].mean():.3f}, S thr mean={thr_s_array[mask].mean():.3f}')
    print(f'\nOverall (n={len(snr_array)}): P thr mean={thr_p_array.mean():.3f}, S thr mean={thr_s_array.mean():.3f}')
    return threshold_data

def plot_dynamic_threshold_pipeline(model, loader, device, out_dir: str, top_k_exceptions: int=5, uncertainty_mode: str='entropy', uncertainty_threshold_options: dict | None=None, tol_samples: int=DYN_TOL_SAMPLES):
    if plt is None:
        return
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    thr_opts = _get_uncertainty_threshold_kwargs(uncertainty_threshold_options)
    time_win = thr_opts.get('time_window', 0)
    aggregate = thr_opts.get('aggregate', 'mean')
    threshold_kwargs = {k: v for k, v in thr_opts.items() if k not in ('time_window', 'aggregate')}
    thresholds_p, thresholds_s = ([], [])
    snrs, uncertainties_p, uncertainties_s = ([], [], [])
    errors_p, errors_s = ([], [])
    deviations_p, deviations_s = ([], [])
    sample_indices = []
    waveforms_list, probs_p_list, probs_s_list, labels_p_list, labels_s_list = ([], [], [], [], [])
    thresholds_p_list, thresholds_s_list = ([], [])
    exception_indices = []
    with torch.no_grad():
        opts = uncertainty_threshold_options or {}
        stable_dynamic = bool(opts.get('stable_dynamic', False))
        stable_state: dict | None = None
        batch_offset = 0
        for batch_idx, (x, y, _) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            if y.shape[-1] != x.shape[-1]:
                y = _center_crop_time(y, x.shape[-1])
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            fusion_alpha = float((uncertainty_threshold_options or {}).get('fusion_alpha', 0.5))
            use_phase_channels = bool((uncertainty_threshold_options or {}).get('use_phase_channels', True))
            uncertainty = compute_uncertainty_from_probs(probs, mode=uncertainty_mode, time_window=time_win, aggregate=aggregate, fusion_alpha=fusion_alpha, use_phase_channels=use_phase_channels)
            if uncertainty.dim() == 3:
                uncertainty = uncertainty.mean(dim=-1)
            if stable_dynamic:
                suggested_thr, stable_state = _stable_dynamic_threshold(uncertainty, stable_state, opts)
            else:
                suggested_thr = uncertainty_to_threshold(uncertainty, **threshold_kwargs)
            thr_p_batch = suggested_thr[:, 0].cpu().numpy()
            thr_s_batch = suggested_thr[:, 1].cpu().numpy()
            unc_batch = uncertainty.cpu().numpy()
            B = x.shape[0]
            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()
            probs_np = probs.cpu().numpy()
            for b in range(B):
                prob_p = float(probs_np[b, 1, :].max())
                prob_s = float(probs_np[b, 2, :].max())
                p_idx = int(np.argmax(probs_np[b, 1, :]))
                s_idx = int(np.argmax(probs_np[b, 2, :]))
                has_p = bool(y_np[b, 1].max() > 1e-06)
                has_s = bool(y_np[b, 2].max() > 1e-06)
                p_gt = int(np.argmax(y_np[b, 1])) if has_p else None
                s_gt = int(np.argmax(y_np[b, 2])) if has_s else None
                p_ok = has_p and p_gt is not None and (abs(p_idx - p_gt) <= tol_samples)
                s_ok = has_s and s_gt is not None and (abs(s_idx - s_gt) <= tol_samples)
                pass_p = prob_p >= thr_p_batch[b]
                pass_s = prob_s >= thr_s_batch[b]
                p_err = _compute_error_label(pass_p, p_ok, has_p)
                s_err = _compute_error_label(pass_s, s_ok, has_s)
                snr_val = estimate_snr(x_np[b], p_idx=p_gt)
                thresholds_p.append(thr_p_batch[b])
                thresholds_s.append(thr_s_batch[b])
                snrs.append(snr_val)
                uncertainties_p.append(float(unc_batch[b, 0]))
                uncertainties_s.append(float(unc_batch[b, 1]))
                errors_p.append(p_err)
                errors_s.append(s_err)
                sample_idx = batch_offset + b
                sample_indices.append(sample_idx)
                if p_err in ('FP', 'FN') or s_err in ('FP', 'FN'):
                    waveforms_list.append(x_np[b].copy())
                    probs_p_list.append(probs_np[b, 1, :].copy())
                    probs_s_list.append(probs_np[b, 2, :].copy())
                    labels_p_list.append(y_np[b, 1, :].copy())
                    labels_s_list.append(y_np[b, 2, :].copy())
                    thresholds_p_list.append(thr_p_batch[b])
                    thresholds_s_list.append(thr_s_batch[b])
                    exception_indices.append(len(sample_indices) - 1)
            batch_offset += B
    if len(thresholds_p) == 0:
        print('[plot_dynamic_threshold_pipeline] 没有收集到数据，跳过', flush=True)
        return
    thresholds_p = np.array(thresholds_p)
    thresholds_s = np.array(thresholds_s)
    snrs = np.array(snrs)
    uncertainties_p = np.array(uncertainties_p)
    uncertainties_s = np.array(uncertainties_s)
    errors_p = np.array(errors_p)
    errors_s = np.array(errors_s)
    sample_indices = np.array(sample_indices)
    median_thr_p = np.median(thresholds_p)
    median_thr_s = np.median(thresholds_s)
    deviations_p = np.abs(thresholds_p - median_thr_p)
    deviations_s = np.abs(thresholds_s - median_thr_s)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax0 = axes[0]
    tp_mask_p = errors_p == 'TP'
    fp_mask_p = errors_p == 'FP'
    fn_mask_p = errors_p == 'FN'
    if tp_mask_p.any():
        ax0.scatter(uncertainties_p[tp_mask_p], thresholds_p[tp_mask_p], s=deviations_p[tp_mask_p] * 100 + 10, c='gray', alpha=0.3, label='P TP')
    if fp_mask_p.any():
        ax0.scatter(uncertainties_p[fp_mask_p], thresholds_p[fp_mask_p], s=deviations_p[fp_mask_p] * 100 + 10, c='red', alpha=0.8, label='P FP')
    if fn_mask_p.any():
        ax0.scatter(uncertainties_p[fn_mask_p], thresholds_p[fn_mask_p], s=deviations_p[fn_mask_p] * 100 + 10, c='orange', alpha=0.8, label='P FN')
    tp_mask_s = errors_s == 'TP'
    fp_mask_s = errors_s == 'FP'
    fn_mask_s = errors_s == 'FN'
    if tp_mask_s.any():
        ax0.scatter(uncertainties_s[tp_mask_s], thresholds_s[tp_mask_s], s=deviations_s[tp_mask_s] * 100 + 10, c='gray', alpha=0.3, marker='x', label='S TP')
    if fp_mask_s.any():
        ax0.scatter(uncertainties_s[fp_mask_s], thresholds_s[fp_mask_s], s=deviations_s[fp_mask_s] * 100 + 10, c='red', alpha=0.8, marker='x', label='S FP')
    if fn_mask_s.any():
        ax0.scatter(uncertainties_s[fn_mask_s], thresholds_s[fn_mask_s], s=deviations_s[fn_mask_s] * 100 + 10, c='orange', alpha=0.8, marker='x', label='S FN')
    ax0.set_xlabel('Uncertainty')
    ax0.set_ylabel('Dynamic Threshold')
    ax0.set_title('Threshold vs Uncertainty')
    ax0.legend(loc='upper right', fontsize=9)
    ax0.grid(True, alpha=0.3)
    for wave_type, thr, err, dev, unc_arr, idx_arr in zip(['P', 'S'], [thresholds_p, thresholds_s], [errors_p, errors_s], [deviations_p, deviations_s], [uncertainties_p, uncertainties_s], [sample_indices, sample_indices]):
        for etype in ['FP', 'FN']:
            mask = err == etype
            if mask.sum() > 0:
                top_k = min(top_k_exceptions, mask.sum())
                top_idx = np.argsort(dev[mask])[-top_k:]
                for i in np.where(mask)[0][top_idx]:
                    ax0.annotate(f'{wave_type}{idx_arr[i]}', (unc_arr[i], thr[i]), color='red' if etype == 'FP' else 'orange', fontsize=7, alpha=0.8)
    ax1 = axes[1]
    if tp_mask_p.any():
        ax1.scatter(snrs[tp_mask_p], thresholds_p[tp_mask_p], s=deviations_p[tp_mask_p] * 100 + 10, c='gray', alpha=0.3, label='P TP')
    if fp_mask_p.any():
        ax1.scatter(snrs[fp_mask_p], thresholds_p[fp_mask_p], s=deviations_p[fp_mask_p] * 100 + 10, c='red', alpha=0.8, label='P FP')
    if fn_mask_p.any():
        ax1.scatter(snrs[fn_mask_p], thresholds_p[fn_mask_p], s=deviations_p[fn_mask_p] * 100 + 10, c='orange', alpha=0.8, label='P FN')
    if tp_mask_s.any():
        ax1.scatter(snrs[tp_mask_s], thresholds_s[tp_mask_s], s=deviations_s[tp_mask_s] * 100 + 10, c='gray', alpha=0.3, marker='x', label='S TP')
    if fp_mask_s.any():
        ax1.scatter(snrs[fp_mask_s], thresholds_s[fp_mask_s], s=deviations_s[fp_mask_s] * 100 + 10, c='red', alpha=0.8, marker='x', label='S FP')
    if fn_mask_s.any():
        ax1.scatter(snrs[fn_mask_s], thresholds_s[fn_mask_s], s=deviations_s[fn_mask_s] * 100 + 10, c='orange', alpha=0.8, marker='x', label='S FN')
    ax1.set_xlabel('SNR')
    ax1.set_ylabel('Dynamic Threshold')
    ax1.set_title('Threshold vs SNR (highlight FP/FN)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    for wave_type, thr, err, dev, idx_arr in zip(['P', 'S'], [thresholds_p, thresholds_s], [errors_p, errors_s], [deviations_p, deviations_s], [sample_indices, sample_indices]):
        for etype in ['FP', 'FN']:
            mask = err == etype
            if mask.sum() > 0:
                top_k = min(top_k_exceptions, mask.sum())
                top_idx = np.argsort(dev[mask])[-top_k:]
                for i in np.where(mask)[0][top_idx]:
                    ax1.annotate(f'{wave_type}{idx_arr[i]}', (snrs[i], thr[i]), color='red' if etype == 'FP' else 'orange', fontsize=7, alpha=0.8)
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'dynamic_threshold_pipeline.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'[INFO] 论文级动态阈值图已生成：{out_path}', flush=True)
    if len(waveforms_list) == 0:
        print(f'[INFO] 没有异常样本（FP/FN），跳过异常样本波形图', flush=True)
    else:
        waveforms_arr = np.array(waveforms_list)
        probs_p_arr = np.array(probs_p_list)
        probs_s_arr = np.array(probs_s_list)
        labels_p_arr = np.array(labels_p_list)
        labels_s_arr = np.array(labels_s_list)
        thresholds_p_arr = np.array(thresholds_p_list)
        thresholds_s_arr = np.array(thresholds_s_list)
        exception_indices_arr = np.array(exception_indices)
        for wave_type, err_arr_all, dev_arr_all, probs_arr, labels_arr, thresholds_arr in zip(['P', 'S'], [errors_p, errors_s], [deviations_p, deviations_s], [probs_p_arr, probs_s_arr], [labels_p_arr, labels_s_arr], [thresholds_p_arr, thresholds_s_arr]):
            for etype in ['FP', 'FN']:
                mask_all = err_arr_all == etype
                if mask_all.sum() == 0:
                    continue
                exception_mask = np.isin(exception_indices_arr, np.where(mask_all)[0])
                if exception_mask.sum() == 0:
                    continue
                exception_orig_indices = exception_indices_arr[exception_mask]
                dev_exceptions = dev_arr_all[exception_orig_indices]
                top_k = min(top_k_exceptions, len(dev_exceptions))
                top_local_indices = np.argsort(dev_exceptions)[-top_k:]
                selected_exception_indices = np.where(exception_mask)[0][top_local_indices]
                for rank, exc_idx in enumerate(selected_exception_indices):
                    orig_idx = sample_indices[exception_indices_arr[exc_idx]]
                    waveform = waveforms_arr[exc_idx]
                    prob_curve = probs_arr[exc_idx]
                    label_curve = labels_arr[exc_idx]
                    threshold_val = thresholds_arr[exc_idx]
                    deviation_val = dev_arr_all[exception_indices_arr[exc_idx]]
                    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
                    T_prob = len(prob_curve)
                    if waveform.ndim == 2:
                        waveform_plot = waveform[0, :] if waveform.shape[0] > 0 else waveform.mean(axis=0)
                    else:
                        waveform_plot = np.asarray(waveform).flatten()
                    T = min(len(waveform_plot), T_prob)
                    t = np.arange(T)
                    waveform_plot = waveform_plot[:T]
                    prob_curve = prob_curve[:T]
                    label_curve = label_curve[:T]
                    color_wave = 'black'
                    ax1.set_xlabel('Time (samples)')
                    ax1.set_ylabel('Waveform Amplitude', color=color_wave)
                    ax1.plot(t, waveform_plot, color=color_wave, alpha=0.6, linewidth=0.8, label='Waveform')
                    ax1.tick_params(axis='y', labelcolor=color_wave)
                    ax1.grid(True, alpha=0.3)
                    ax2 = ax1.twinx()
                    color_prob = 'blue'
                    color_thr = 'red'
                    ax2.set_ylabel('Probability / Threshold', color=color_prob)
                    ax2.plot(t, prob_curve, color=color_prob, alpha=0.8, linewidth=1.5, label=f'{wave_type} Prediction Prob')
                    ax2.axhline(y=threshold_val, color=color_thr, linestyle='--', linewidth=2, label=f'Dynamic Threshold ({threshold_val:.3f})')
                    ax2.tick_params(axis='y', labelcolor=color_prob)
                    ax2.set_ylim(0, 1.0)
                    if label_curve.max() > 1e-06:
                        gt_idx = int(np.argmax(label_curve))
                        ax1.axvline(x=gt_idx, color='green', linestyle=':', alpha=0.7, linewidth=1.5, label=f'{wave_type} Ground Truth')
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
                    ax1.set_title(f'{wave_type}-wave {etype} Sample #{orig_idx} (Deviation={deviation_val:.4f}, Rank={rank + 1}/{top_k})')
                    plt.tight_layout()
                    sample_out_path = os.path.join(out_dir, f'{wave_type}_{etype}_sample_{orig_idx}_rank{rank + 1}.png')
                    fig.savefig(sample_out_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
        print(f'[INFO] 异常样本波形图已生成（每类波 top-{top_k_exceptions} FP/FN）', flush=True)

def run_case(case: Dict[str, Any]):
    case_seed = case.get('seed', SEED)
    seed_everything(case_seed, deterministic=True)
    eval_only = bool(case.get('eval_only', False))
    print(f'[{case['name']}] 初始化设备...', flush=True)
    print(f'[{case['name']}] 使用随机种子: {case_seed}', flush=True)
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device = torch.device('cuda')
        try:
            print(f'[{case['name']}] 使用设备: {device} ({torch.cuda.get_device_name(0)})', flush=True)
        except Exception:
            print(f'[{case['name']}] 使用设备: {device}', flush=True)
    else:
        device = torch.device('cpu')
        print(f'[{case['name']}] 使用设备: {device}（未检测到可用 CUDA，若需 GPU 请检查 PyTorch 是否为 GPU 版及驱动）', flush=True)
    print(f'[{case['name']}] 加载数据集...', flush=True)
    train_ds, valid_ds, test_ds = build_datasets(case)
    print(f'[{case['name']}] 数据集加载完成: 训练集={len(train_ds)}, 验证集={len(valid_ds)}' + (f', 测试集={len(test_ds)}' if test_ds is not None else ''), flush=True)
    g_train = torch_generator(case_seed)
    g_valid = torch_generator(case_seed + 1)
    batch_size = case.get('batch_size', BATCH_SIZE)
    if batch_size != BATCH_SIZE:
        print(f'[{case['name']}] 使用 batch_size={batch_size}（默认 {BATCH_SIZE}）', flush=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False, worker_init_fn=seed_worker, generator=g_train)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False, worker_init_fn=seed_worker, generator=g_valid)
    run_name = f'{case['name']}_seed{case_seed}'
    log_split_info(run_name, len(train_ds), len(valid_ds), seed_value=case_seed)
    case_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(case_dir, exist_ok=True)
    _drop_rate = case.get('drop_rate', case.get('dropout', 0.0))
    _pool_size = case.get('pool_size', 4)
    model_cfg = dict(in_ch=3, n_class=3, depths=case.get('depths', 5), filters_root=case.get('filters_root', 8), kernels=case.get('kernels', (7,)), pool_size=_pool_size, drop_rate=_drop_rate, use_cbam=case.get('use_cbam', False), use_separable=case.get('use_separable', False), fusion_mode=case.get('fusion_mode', 'concat'), fusion_gate_hidden=case.get('fusion_gate_hidden', 16), fusion_residual_scale=case.get('fusion_residual_scale', 0.3), fusion_use_maxpool=case.get('fusion_use_maxpool', True), softgate_scope=case.get('softgate_scope', 'all'), use_temporal_bifpn_asff=case.get('use_temporal_bifpn_asff', False), cbam_modulate_softgate=case.get('cbam_modulate_softgate', True), cbam_softgate_strength=float(case.get('cbam_softgate_strength', 1.0)))
    print(f'[{case['name']}] 构建模型...', flush=True)
    model_cls = MODEL_CLASS_MAP.get(case.get('model_class', 'PhaseNetUNet'), PhaseNetUNet)
    try:
        _sig = inspect.signature(model_cls.__init__)
        _accepted = set(_sig.parameters.keys()) - {'self'}
        _filtered_model_cfg = {k: v for k, v in model_cfg.items() if k in _accepted}
        _dropped = sorted(set(model_cfg.keys()) - set(_filtered_model_cfg.keys()))
        if _dropped:
            print(f'[run_case] 跳过当前 PhaseNetUNet 不支持的参数: {_dropped}', flush=True)
    except Exception:
        _filtered_model_cfg = model_cfg
    model = model_cls(**_filtered_model_cfg).to(device)
    model.use_ttversky_loss = bool(case.get('use_ttversky_loss', False))
    model.tt_loss_weight = float(case.get('tt_loss_weight', 0.3))
    model.tt_time_weight = float(case.get('tt_time_weight', 0.3))
    model.tt_start_weight = float(case.get('tt_start_weight', 0.4))
    model.tt_temporal_att_weight = float(case.get('tt_temporal_att_weight', 0.1))
    model.tt_start_window = int(case.get('tt_start_window', 2))
    model.tt_start_peak_threshold = float(case.get('tt_start_peak_threshold', 0.2))
    model.tt_alpha_p = float(case.get('tt_alpha_p', 0.7))
    model.tt_beta_p = float(case.get('tt_beta_p', 0.3))
    model.tt_alpha_s = float(case.get('tt_alpha_s', 0.8))
    model.tt_beta_s = float(case.get('tt_beta_s', 0.2))
    has_dropout = any(('Dropout' in m.__class__.__name__ for m in model.modules()))
    n_dropout = sum((1 for m in model.modules() if 'Dropout' in m.__class__.__name__))
    params = sum((p.numel() for p in model.parameters()))
    size_mb = params * 4.0 / 1024.0 ** 2
    print(f'[{case['name']}] 模型构建完成，参数量: {params / 1000000.0:.2f}M (~{params / 1000.0:.1f}k)，size≈{size_mb:.2f}MB', flush=True)
    print(f'[{case['name']}] has_dropout_modules: {has_dropout}, dropout_count: {n_dropout}', flush=True)
    total_epochs = EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if SCHEDULER == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, threshold=PLATEAU_THRESHOLD, min_lr=PLATEAU_MIN_LR)
    else:
        scheduler = None
    class_weights_tensor = None
    tr_hist, va_hist = ([], [])
    best_by_f1 = bool(case.get('best_model_by_f1', BEST_MODEL_BY_F1))
    best = -1.0 if best_by_f1 else float('inf')
    threshold_balance_weight = case.get('threshold_balance_weight', 0.03)
    best_state_dict = None
    best_epoch = 1
    if eval_only:
        best_path = os.path.join(case_dir, 'best_model.pt')
        if os.path.exists(best_path):
            print(f'[{case['name']}] eval_only=True，加载已有最优模型: {best_path}', flush=True)
            state = torch.load(best_path, map_location=device)
            model.load_state_dict(state)
            tr_hist = [0.0]
            va_hist = [0.0]
            best = 0.0
        else:
            print(f'[{case['name']}] eval_only=True 但未找到 {best_path}，将执行正常训练流程。', flush=True)
            eval_only = False
    if not eval_only:
        enable_early_stop = DATA_SOURCE == 'h5_three_channel'
        no_improve_epochs = 0
        for ep in range(1, total_epochs + 1):
            current_threshold_weight = threshold_balance_weight
            model.train()
            total_tr = 0.0
            n_tr = 0
            pbar = tqdm(train_loader, desc=f'Epoch {ep}/{total_epochs} [train]', leave=False)
            for x, y, _ in pbar:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                if y.shape[-1] != logits.shape[-1]:
                    y = _center_crop_time(y, logits.shape[-1])
                temporal_att = getattr(model, 'last_temporal_att', None)
                loss = combined_loss(logits, y, threshold_balance_weight=current_threshold_weight, tol_samples=DYN_TOL_SAMPLES, class_weights=class_weights_tensor, optimal_thr_mse_weight=case.get('optimal_thr_mse_weight', 0.15), sample_level_balance=case.get('sample_level_balance', True), thr_reg_weight=case.get('thr_reg_weight', 0.008), use_phasewise_loss=case.get('use_phasewise_loss', False), phasewise_loss_weight=case.get('phasewise_loss_weight', 0.0), p_tversky_alpha=case.get('p_tversky_alpha', 0.7), p_tversky_beta=case.get('p_tversky_beta', 0.3), s_tversky_alpha=case.get('s_tversky_alpha', 0.3), s_tversky_beta=case.get('s_tversky_beta', 0.7), use_ttversky_loss=case.get('use_ttversky_loss', False), tt_loss_weight=case.get('tt_loss_weight', 0.3), temporal_att=temporal_att, tt_time_weight=case.get('tt_time_weight', 0.3), tt_start_weight=case.get('tt_start_weight', 0.4), tt_temporal_att_weight=case.get('tt_temporal_att_weight', 0.1), tt_start_window=case.get('tt_start_window', 2), tt_start_peak_threshold=case.get('tt_start_peak_threshold', 0.2), tt_alpha_p=case.get('tt_alpha_p', 0.7), tt_beta_p=case.get('tt_beta_p', 0.3), tt_alpha_s=case.get('tt_alpha_s', 0.8), tt_beta_s=case.get('tt_beta_s', 0.2))
                loss.backward()
                opt.step()
                bs = x.size(0)
                total_tr += float(loss.item()) * bs
                n_tr += bs
                pbar.set_postfix({'loss': f'{total_tr / max(1, n_tr):.4f}', 'lr': f'{opt.param_groups[0]['lr']:.2e}'})
            tr_loss = total_tr / max(1, n_tr)
            va_loss = eval_loss(model, valid_loader, device, threshold_balance_weight=current_threshold_weight, class_weights=class_weights_tensor, epoch=ep, total_epochs=total_epochs)
            if scheduler is not None:
                scheduler.step(va_loss)
            cur_lr = opt.param_groups[0].get('lr', LR)
            tr_hist.append(tr_loss)
            va_hist.append(va_loss)
            improved = False
            if best_by_f1:
                _fixed_p = float(case.get('fixed_thr_p', 0.5))
                _fixed_s = float(case.get('fixed_thr_s', 0.5))
                eval_metrics = eval_detailed(model, valid_loader, device, thr_p=_fixed_p, thr_s=_fixed_s, uncertainty_threshold_options=case, current_epoch=ep, tol=DYN_TOL_SAMPLES)
                weights = case.get('best_metric_weights', BEST_METRIC_WEIGHTS)
                w_p, w_s = (float(weights[0]), float(weights[1]))
                selection = w_p * eval_metrics['p_f1'] + w_s * eval_metrics['s_f1']
                if selection > best:
                    best = selection
                    best_epoch = ep
                    best_state_dict = copy.deepcopy(model.state_dict())
                    improved = True
            elif va_loss < best:
                best = va_loss
                best_epoch = ep
                best_state_dict = copy.deepcopy(model.state_dict())
                improved = True
            if enable_early_stop:
                if improved:
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
            best_label = 'best_f1' if best_by_f1 else 'best_val'
            print(f'[{case['name']}] epoch {ep:02d}/{total_epochs} train={tr_loss:.4f} valid={va_loss:.4f} {best_label}={best:.4f} lr={cur_lr:.2e}' + (f' no_improve={no_improve_epochs}/{H5_EARLY_STOP_PATIENCE}' if enable_early_stop else ''), flush=True)
            if enable_early_stop and no_improve_epochs >= H5_EARLY_STOP_PATIENCE:
                print(f'[{case['name']}] EarlyStopping(H5) 触发：连续 {H5_EARLY_STOP_PATIENCE} 个 epoch 未提升，提前结束训练。', flush=True)
                break
    if best_state_dict is not None and (not eval_only):
        model.load_state_dict(best_state_dict)
        criterion = 'best_f1' if best_by_f1 else 'best_val_loss'
        print(f'[{case['name']}] 使用验证集最优模型进行评估（best_epoch={best_epoch}, {criterion}={best:.4f}）', flush=True)
        best_path = os.path.join(case_dir, 'best_model.pt')
        torch.save(best_state_dict, best_path)
        print(f'[{case['name']}] 最优模型已保存: {best_path}', flush=True)
    use_mc_dropout_selective = case.get('use_mc_dropout_selective', False)
    fixed_thr_p = float(case.get('fixed_thr_p', 0.5))
    fixed_thr_s = float(case.get('fixed_thr_s', 0.5))
    if use_mc_dropout_selective:
        print(f'[{case['name']}] Using MC Dropout + Selective Prediction (fixed threshold P={fixed_thr_p:.3f}, S={fixed_thr_s:.3f})', flush=True)
        thr_p = fixed_thr_p
        thr_s = fixed_thr_s
        f1_p = f1_s = float('nan')
    else:
        print(f'[{case['name']}] 使用固定阈值 P={fixed_thr_p:.3f} S={fixed_thr_s:.3f}（基线模式）...', flush=True)
        thr_p = fixed_thr_p
        thr_s = fixed_thr_s
        print(f'[{case['name']}] 检查预测概率分布...', flush=True)
        model.eval()
        all_p_probs, all_s_probs = ([], [])
        with torch.inference_mode():
            for x, y, _ in valid_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                if y.shape[-1] != logits.shape[-1]:
                    y = _center_crop_time(y, logits.shape[-1])
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                y_np = y.cpu().numpy()
                B = probs.shape[0]
                for b in range(B):
                    if y_np[b, 1].max() > 1e-06:
                        p_max_prob = float(np.max(probs[b, 1]))
                        all_p_probs.append(p_max_prob)
                    if y_np[b, 2].max() > 1e-06:
                        s_max_prob = float(np.max(probs[b, 2]))
                        all_s_probs.append(s_max_prob)
        if all_p_probs:
            print(f'[{case['name']}] P波预测概率统计: mean={np.mean(all_p_probs):.3f}, std={np.std(all_p_probs):.3f}, min={np.min(all_p_probs):.3f}, max={np.max(all_p_probs):.3f}, 超过0.5的比例={np.mean(np.array(all_p_probs) >= 0.5):.1%}', flush=True)
        if all_s_probs:
            print(f'[{case['name']}] S波预测概率统计: mean={np.mean(all_s_probs):.3f}, std={np.std(all_s_probs):.3f}, min={np.min(all_s_probs):.3f}, max={np.max(all_s_probs):.3f}, 超过0.5的比例={np.mean(np.array(all_s_probs) >= 0.5):.1%}', flush=True)
        (p_conf, p_err, p_has), (s_conf, s_err, s_has) = collect_conf_err(model, valid_loader, device)
        _, f1_p = best_threshold(p_conf, p_err, p_has, DYN_TOL_SAMPLES, [thr_p], debug=False)
        _, f1_s = best_threshold(s_conf, s_err, s_has, DYN_TOL_SAMPLES, [thr_s], debug=False)
        print(f'[{case['name']}] 固定阈值: P={thr_p:.3f} (F1={f1_p:.4f}), S={thr_s:.3f} (F1={f1_s:.4f})', flush=True)
    metrics_unc = None
    metrics_score = None
    metrics_gate = None
    metrics_full = eval_detailed(model, valid_loader, device, thr_p=thr_p, thr_s=thr_s, uncertainty_threshold_options=case, current_epoch=total_epochs, tol=DYN_TOL_SAMPLES)
    metrics_mc: Dict[str, Any] | None = None
    if use_mc_dropout_selective:
        mc_T = int(case.get('mc_dropout_n_samples', 20))
        drop_ratio = float(case.get('mc_selective_drop_ratio', 0.1))
        coverage_points = int(case.get('mc_risk_coverage_points', 20))
        eval_seed = int(case.get('mc_eval_seed', SEED))
        metrics_mc = eval_detailed_mc_selective(model, valid_loader, device, thr_p=thr_p, thr_s=thr_s, mc_T=mc_T, drop_ratio=drop_ratio, coverage_points=coverage_points, tol=DYN_TOL_SAMPLES, quiet=False, eval_seed=eval_seed)
        if case.get('mc_use_unc_candidate', False):
            cand_thr_p = float(case.get('mc_candidate_thr_p', 0.3))
            cand_thr_s = float(case.get('mc_candidate_thr_s', 0.3))
            metrics_unc = eval_detailed_mc_selective(model, valid_loader, device, thr_p=thr_p, thr_s=thr_s, mc_T=mc_T, drop_ratio=drop_ratio, coverage_points=coverage_points, tol=DYN_TOL_SAMPLES, structural_opts=None, quiet=True, eval_seed=eval_seed, use_two_level_candidate=True, candidate_thr_p=cand_thr_p, candidate_thr_s=cand_thr_s)
            try:
                print(f'[{case['name']}] 两级阈值+unc 候选 对照 (Full 指标):', flush=True)
                print(f'  baseline : P-Prec={metrics['p_prec']:.4f}, P-Rec={metrics['p_rec']:.4f}, P-F1={metrics['p_f1']:.4f}; S-Prec={metrics['s_prec']:.4f}, S-Rec={metrics['s_rec']:.4f}, S-F1={metrics['s_f1']:.4f}', flush=True)
                print(f'  unc-cand : P-Prec={metrics_unc['p_prec']:.4f}, P-Rec={metrics_unc['p_rec']:.4f}, P-F1={metrics_unc['p_f1']:.4f}; S-Prec={metrics_unc['s_prec']:.4f}, S-Rec={metrics_unc['s_rec']:.4f}, S-F1={metrics_unc['s_f1']:.4f}', flush=True)
            except Exception:
                pass
        if case.get('mc_use_score_candidate_s', False):
            score_cand_thr_s = float(case.get('mc_score_candidate_thr_s', 0.3))
            score_lambda_s = float(case.get('mc_score_lambda_s', 0.6))
            score_tau_unc_s = case.get('mc_score_tau_unc_s', None)
            metrics_score = eval_detailed_mc_selective(model, valid_loader, device, thr_p=thr_p, thr_s=thr_s, mc_T=mc_T, drop_ratio=drop_ratio, coverage_points=coverage_points, tol=DYN_TOL_SAMPLES, structural_opts=None, quiet=True, eval_seed=eval_seed, use_two_level_candidate=False, candidate_thr_s=score_cand_thr_s, use_score_candidate_s=True, score_lambda_s=score_lambda_s, score_tau_unc_s=score_tau_unc_s)
            try:
                print(f'[{case['name']}] S-score 候选 对照 (Full 指标): (lambda_s={score_lambda_s:.2f}, cand_thr_s={score_cand_thr_s:.2f}, tau_unc_s={score_tau_unc_s})', flush=True)
                print(f'  baseline : S-Prec={metrics['s_prec']:.4f}, S-Rec={metrics['s_rec']:.4f}, S-F1={metrics['s_f1']:.4f}', flush=True)
                print(f'  score-cand: S-Prec={metrics_score['s_prec']:.4f}, S-Rec={metrics_score['s_rec']:.4f}, S-F1={metrics_score['s_f1']:.4f}', flush=True)
            except Exception:
                pass
        if case.get('mc_use_unc_gating_s', False):
            gate_tau_s = case.get('mc_unc_gating_tau_s', None)
            gate_k_s = case.get('mc_unc_gating_k_s', None)
            gate_base_s = case.get('mc_unc_gating_base_s', None)
            metrics_gate = eval_detailed_mc_selective(model, valid_loader, device, thr_p=thr_p, thr_s=thr_s, mc_T=mc_T, drop_ratio=drop_ratio, coverage_points=coverage_points, tol=DYN_TOL_SAMPLES, structural_opts=None, quiet=True, eval_seed=eval_seed, use_two_level_candidate=False, use_score_candidate_s=False, use_unc_gating_s=True, unc_gating_tau_s=gate_tau_s, unc_gating_k_s=gate_k_s, unc_gating_base_s=gate_base_s)
            try:
                print(f'[{case['name']}] S-unc gating 对照 (Full 指标): (tau_unc_s={gate_tau_s}, k_s={gate_k_s}, base_s={gate_base_s})', flush=True)
                print(f'  baseline : S-Prec={metrics['s_prec']:.4f}, S-Rec={metrics['s_rec']:.4f}, S-F1={metrics['s_f1']:.4f}', flush=True)
                print(f'  unc-gate : S-Prec={metrics_gate['s_prec']:.4f}, S-Rec={metrics_gate['s_rec']:.4f}, S-F1={metrics_gate['s_f1']:.4f}', flush=True)
            except Exception:
                pass
    if isinstance(metrics_mc, dict) and 'mc_uncertainty' in metrics_mc:
        try:
            metrics_mc['mc_uncertainty'] = [float(u) for u in metrics_mc['mc_uncertainty']]
        except Exception:
            pass
    if isinstance(metrics_unc, dict) and 'mc_uncertainty' in metrics_unc:
        try:
            metrics_unc['mc_uncertainty'] = [float(u) for u in metrics_unc['mc_uncertainty']]
        except Exception:
            pass
    test_loader_for_vis = None
    if RUN_TEST_EVAL and test_ds is not None:
        try:
            print(f'[{case['name']}] 在独立测试集上评估（不再调参，仅前向计算指标）...', flush=True)
            g_test = torch_generator(SEED + 2)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False, worker_init_fn=seed_worker, generator=g_test)
            test_metrics = eval_detailed(model, test_loader, device, thr_p=thr_p, thr_s=thr_s, uncertainty_threshold_options=case, current_epoch=total_epochs, tol=DYN_TOL_SAMPLES)

            def _fmt_res_t(key: str) -> str:
                v = test_metrics.get(key)
                return f'{float(v):.4f}' if v is not None else 'N/A'
            print(f'[{case['name']}] TEST 结果: time_acc={test_metrics.get('time_acc', float('nan')):.4f}, P-Prec={test_metrics.get('p_prec', float('nan')):.4f}, P-Rec={test_metrics.get('p_rec', float('nan')):.4f}, P-F1={test_metrics.get('p_f1', float('nan')):.4f}, S-Prec={test_metrics.get('s_prec', float('nan')):.4f}, S-Rec={test_metrics.get('s_rec', float('nan')):.4f}, S-F1={test_metrics.get('s_f1', float('nan')):.4f} | p_res(mean/std/mae)s=({_fmt_res_t('p_res_mean_sec')}/{_fmt_res_t('p_res_std_sec')}/{_fmt_res_t('p_res_mae_sec')}), s_res=({_fmt_res_t('s_res_mean_sec')}/{_fmt_res_t('s_res_std_sec')}/{_fmt_res_t('s_res_mae_sec')})', flush=True)
            test_loader_for_vis = test_loader
        except Exception as e:
            print(f'[{case['name']}] 在测试集上评估失败（{e}），仅保留验证集指标。', flush=True)
    case_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(case_dir, exist_ok=True)
    if use_mc_dropout_selective and isinstance(metrics_mc, dict) and (metrics_mc.get('risk_coverage') is not None):
        try:
            unc_conf_list = compute_uncertainty_confidence_non_mc(model, valid_loader, device)
            if len(unc_conf_list) == len(metrics_mc.get('mc_uncertainty', [])):
                metrics_mc['mc_uncertainty_conf'] = [float(u) for u in unc_conf_list]
        except Exception:
            pass
        try:
            rc_conf = compute_risk_coverage_confidence_non_mc(model, valid_loader, device, thr_p=thr_p, thr_s=thr_s, coverage_points=coverage_points, tol=DYN_TOL_SAMPLES)
            if isinstance(metrics_mc.get('risk_coverage'), dict) and isinstance(rc_conf, dict):
                metrics_mc['risk_coverage']['P_conf'] = rc_conf.get('P_conf', [])
                metrics_mc['risk_coverage']['S_conf'] = rc_conf.get('S_conf', [])
        except Exception:
            pass
        with open(os.path.join(case_dir, 'risk_coverage.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics_mc['risk_coverage'], f)
        ue_export = {'uncertainty': [float(u) for u in metrics_mc['mc_uncertainty']], 'uncertainty_mc': [float(u) for u in metrics_mc['mc_uncertainty']], 'uncertainty_conf': [float(u) for u in metrics_mc.get('mc_uncertainty_conf', [])], 'error_flag': metrics_mc['mc_error_flag'], 'error_type_p': metrics_mc['mc_error_type_p'], 'error_type_s': metrics_mc['mc_error_type_s'], 'unc_p': [float(u) for u in metrics_mc.get('unc_p_list', [])] if metrics_mc.get('unc_p_list') is not None else [], 'unc_s': [float(u) for u in metrics_mc.get('unc_s_list', [])] if metrics_mc.get('unc_s_list') is not None else []}
        with open(os.path.join(case_dir, 'uncertainty_error_data.json'), 'w', encoding='utf-8') as f:
            json.dump(ue_export, f)
        if plt is not None:
            try:
                overview_path = os.path.join(case_dir, 'uncertainty_overview_grid.png')
                plot_uncertainty_overview_grid(metrics_mc['risk_coverage'], ue_export, overview_path, num_bins=10)
                print(f'[{case['name']}] 已生成: {overview_path}', flush=True)
            except Exception as e:
                print(f'[{case['name']}] 生成 uncertainty_overview_grid.png 失败: {e}', flush=True)
        n_samples = len(metrics_mc['mc_uncertainty'])
        unc_arr = np.array(metrics_mc['mc_uncertainty'], dtype=np.float64)
        etp: list[str] = metrics_mc['mc_error_type_p']
        ets: list[str] = metrics_mc['mc_error_type_s']
        snr_arr: list[float] = [float('nan')] * n_samples
        batch_offset = 0
        try:
            with torch.inference_mode():
                for x, y, _names in valid_loader:
                    x_np = x.cpu().numpy()
                    y_np = y.cpu().numpy()
                    B = x_np.shape[0]
                    for b in range(B):
                        idx = batch_offset + b
                        if idx >= n_samples:
                            break
                        p_idx = int(np.argmax(y_np[b, 1])) if y_np[b, 1].max() > 1e-06 else None
                        snr_arr[idx] = float(estimate_snr(x_np[b], p_idx=p_idx))
                    batch_offset += B
        except Exception:
            snr_arr = [float('nan')] * n_samples

        def _snr_bucket(v: float) -> str:
            if not np.isfinite(v):
                return 'mid'
            if v < 5.0:
                return 'low'
            if v < 15.0:
                return 'mid'
            return 'high'

        def _select_typical_indices(target: str, max_count: int) -> list[int]:
            cand: list[int] = []
            for i in range(n_samples):
                if target == 'SFP' and ets[i] == 'FP':
                    cand.append(i)
                elif target == 'PFN' and etp[i] == 'FN':
                    cand.append(i)
            if not cand or max_count <= 0:
                return []
            buckets: dict[str, list[int]] = {'low': [], 'mid': [], 'high': []}
            for i in cand:
                b = _snr_bucket(snr_arr[i])
                if b not in buckets:
                    b = 'mid'
                buckets[b].append(i)
            for key in buckets:
                buckets[key].sort(key=lambda idx: unc_arr[idx], reverse=True)
            chosen: list[int] = []
            for key in ('low', 'mid', 'high'):
                if len(chosen) >= max_count:
                    break
                lst = buckets[key]
                if lst:
                    chosen.append(lst.pop(0))
            if len(chosen) < max_count:
                remaining = [i for i in cand if i not in chosen]
                remaining.sort(key=lambda idx: unc_arr[idx], reverse=True)
                for i in remaining:
                    if len(chosen) >= max_count:
                        break
                    chosen.append(i)
            return chosen
        sorted_idx_by_unc = np.argsort(unc_arr)[::-1]
        idx_to_group: dict[int, str] = {}
        idx_to_subtype: dict[int, str] = {}
        typical_sfp_max = 3
        for i in _select_typical_indices('SFP', typical_sfp_max):
            if i not in idx_to_group:
                idx_to_group[i] = 'typical'
                idx_to_subtype[i] = 'SFP'
        typical_pfn_max = 3
        for i in _select_typical_indices('PFN', typical_pfn_max):
            if i not in idx_to_group:
                idx_to_group[i] = 'typical'
                idx_to_subtype[i] = 'PFN'
        high_wrong_max = 3
        for i in sorted_idx_by_unc:
            if len([k for k, g in idx_to_group.items() if g == 'high_unc' and idx_to_subtype[k] == 'wrong']) >= high_wrong_max:
                break
            wrong = etp[i] in ('FP', 'FN') or ets[i] in ('FP', 'FN')
            if wrong and i not in idx_to_group:
                idx_to_group[i] = 'high_unc'
                idx_to_subtype[i] = 'wrong'
        high_correct_max = 3
        for i in sorted_idx_by_unc:
            if len([k for k, g in idx_to_group.items() if g == 'high_unc' and idx_to_subtype[k] == 'correct']) >= high_correct_max:
                break
            wrong = etp[i] in ('FP', 'FN') or ets[i] in ('FP', 'FN')
            if not wrong and i not in idx_to_group:
                idx_to_group[i] = 'high_unc'
                idx_to_subtype[i] = 'correct'
        failure_indices_set = set(idx_to_group.keys())
        fail_x, fail_y, fail_probs = ([], [], [])
        fail_names, fail_snr, fail_unc = ([], [], [])
        fail_etp, fail_ets, fail_group, fail_subtype = ([], [], [], [])
        batch_offset = 0
        model.eval()
        with torch.inference_mode():
            for x, y, names_batch in valid_loader:
                x = x.to(device)
                y = y.to(device)
                if y.shape[-1] != x.shape[-1]:
                    y = _center_crop_time(y, x.shape[-1])
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                x_np = x.cpu().numpy()
                y_np = y.cpu().numpy()
                probs_np = probs.cpu().numpy()
                B = x_np.shape[0]
                for b in range(B):
                    idx = batch_offset + b
                    if idx not in failure_indices_set:
                        continue
                    name_b = names_batch[b] if hasattr(names_batch, '__getitem__') else str(idx)
                    snr_b = float(snr_arr[idx]) if 0 <= idx < n_samples else float('nan')
                    fail_x.append(x_np[b])
                    fail_y.append(y_np[b])
                    fail_probs.append(probs_np[b])
                    fail_names.append(str(name_b))
                    fail_snr.append(snr_b)
                    fail_unc.append(metrics_mc['mc_uncertainty'][idx])
                    fail_etp.append(etp[idx])
                    fail_ets.append(ets[idx])
                    fail_group.append(idx_to_group.get(idx, 'unknown'))
                    fail_subtype.append(idx_to_subtype.get(idx, 'unknown'))
                batch_offset += B
        if fail_x:
            np.savez(os.path.join(case_dir, 'failure_samples.npz'), x_np=np.array(fail_x), y_np=np.array(fail_y), probs=np.array(fail_probs), snr=np.array(fail_snr), uncertainty=np.array(fail_unc), allow_pickle=True)
            with open(os.path.join(case_dir, 'failure_sample_meta.json'), 'w', encoding='utf-8') as f:
                json.dump({'names': fail_names, 'error_type_p': fail_etp, 'error_type_s': fail_ets, 'group': fail_group, 'subtype': fail_subtype}, f, ensure_ascii=False)
        if isinstance(metrics_mc, dict):
            p_res_mc = metrics_mc.get('p_residuals_signed')
            s_res_mc = metrics_mc.get('s_residuals_signed')
            if p_res_mc is not None and s_res_mc is not None:
                time_res_mc = {'p_residuals_signed': p_res_mc, 's_residuals_signed': s_res_mc}
                with open(os.path.join(case_dir, 'time_residuals.json'), 'w', encoding='utf-8') as f:
                    json.dump(time_res_mc, f)
                with open(os.path.join(case_dir, 'time_residuals_ours.json'), 'w', encoding='utf-8') as f:
                    json.dump(time_res_mc, f)
                if plt is not None:
                    plot_time_residual_distribution(p_res_mc, s_res_mc, os.path.join(case_dir, 'time_residual_distribution.png'), sample_rate=SAMPLE_RATE)
    if metrics_mc is None:
        p_res_full = metrics_full.get('p_residuals_signed')
        s_res_full = metrics_full.get('s_residuals_signed')
        if p_res_full is not None and s_res_full is not None:
            time_res_full = {'p_residuals_signed': p_res_full, 's_residuals_signed': s_res_full}
            with open(os.path.join(case_dir, 'time_residuals.json'), 'w', encoding='utf-8') as f:
                json.dump(time_res_full, f)
            with open(os.path.join(case_dir, 'time_residuals_ours.json'), 'w', encoding='utf-8') as f:
                json.dump(time_res_full, f)
            if plt is not None:
                plot_time_residual_distribution(p_res_full, s_res_full, os.path.join(case_dir, 'time_residual_distribution.png'), sample_rate=SAMPLE_RATE)
    if plt is not None and case.get('generate_visualizations', True):
        try:
            features, labels = collect_pca_features_and_labels(model, valid_loader, device)
            if len(features) >= 2:
                np.savez(os.path.join(case_dir, 'pca_features_labels.npz'), features=features, labels=labels, allow_pickle=True)
                np.savez(os.path.join(case_dir, 'pca_ours.npz'), features=features, labels=labels, allow_pickle=True)
                plot_pca_visualization(features, labels, os.path.join(case_dir, 'pca_visualization.png'), label_names=['BG window', 'P window', 'S window'])
        except Exception as e:
            print(f'[{case['name']}] PCA 可视化跳过（{e}）', flush=True)
    if plt is not None and case.get('generate_visualizations', True):
        try:
            pr_snr = collect_pr_snr_data(model, valid_loader, device, tol_samples=DYN_TOL_SAMPLES)
            if pr_snr['has_p']:
                with open(os.path.join(case_dir, 'pr_snr_data.json'), 'w', encoding='utf-8') as f:
                    json.dump(pr_snr, f)
                with open(os.path.join(case_dir, 'pr_snr_data_ours.json'), 'w', encoding='utf-8') as f:
                    json.dump(pr_snr, f)
                _write_comparison_format(case_dir)
                plot_pr_curve(pr_snr, os.path.join(case_dir, 'pr_curve.png'))
                plot_snr_stratified(pr_snr, os.path.join(case_dir, 'snr_stratified.png'), fixed_thr=0.5)
                plot_max_prob_histogram(pr_snr, os.path.join(case_dir, 'max_prob_histogram.png'), bins=15)
        except Exception as e:
            print(f'[{case['name']}] PR/SNR 图跳过（{e}）', flush=True)
    generate_visualizations = case.get('generate_visualizations', True)
    if plt is not None:
        plot_losses(tr_hist, va_hist, os.path.join(case_dir, 'loss_curve.png'))
    if plt is not None and generate_visualizations:
        print(f'[{case['name']}] 生成样本可视化...', flush=True)
        save_visuals(model, valid_ds, device, os.path.join(case_dir, 'figs'), n=N_VIS)
        try:
            rep_out = os.path.join(case_dir, 'representative_waveforms_2x2.png')
            rep_npz = None
            name_str = str(case.get('name', '')).lower()
            if name_str == 'full' or 'full' in name_str:
                rep_npz = os.path.join(case_dir, 'representative_compare.npz')
            extra_rep_loaders = [train_loader]
            if test_loader_for_vis is not None:
                extra_rep_loaders.append(test_loader_for_vis)
            save_representative_waveforms_2x2(model, valid_loader, device, out_path=rep_out, tol=DYN_TOL_SAMPLES, thr_p=float(case.get('fixed_thr_p', 0.5)), thr_s=float(case.get('fixed_thr_s', 0.5)), max_scan_samples=int(case.get('rep_waveform_max_scan', 2000)), extra_loaders=extra_rep_loaders, rep_npz_path=rep_npz)
        except Exception:
            pass
        print(f'[{case['name']}] 可视化图表生成完成（loss_curve.png + figs/vis_*.png）', flush=True)
    elif plt is None:
        print(f'[{case['name']}] 跳过可视化图表生成（matplotlib 不可用）', flush=True)
    elif not generate_visualizations:
        print(f'[{case['name']}] 已输出 loss_curve.png，跳过样本可视化（generate_visualizations=False）', flush=True)
    extra_unc_fields: Dict[str, Any] = {}
    if metrics_unc is not None:
        try:
            extra_unc_fields = dict(p_f1_uncand=float(metrics_unc.get('p_f1', 0.0)), s_f1_uncand=float(metrics_unc.get('s_f1', 0.0)), time_acc_uncand=float(metrics_unc.get('time_acc', 0.0)), p_prec_uncand=float(metrics_unc.get('p_prec', 0.0)), p_rec_uncand=float(metrics_unc.get('p_rec', 0.0)), s_prec_uncand=float(metrics_unc.get('s_prec', 0.0)), s_rec_uncand=float(metrics_unc.get('s_rec', 0.0)))
        except Exception:
            extra_unc_fields = {}
    extra_score_fields: Dict[str, Any] = {}
    if metrics_score is not None:
        try:
            extra_score_fields = dict(p_f1_score=float(metrics_score.get('p_f1', 0.0)), s_f1_score=float(metrics_score.get('s_f1', 0.0)), time_acc_score=float(metrics_score.get('time_acc', 0.0)), p_prec_score=float(metrics_score.get('p_prec', 0.0)), p_rec_score=float(metrics_score.get('p_rec', 0.0)), s_prec_score=float(metrics_score.get('s_prec', 0.0)), s_rec_score=float(metrics_score.get('s_rec', 0.0)))
        except Exception:
            extra_score_fields = {}
    extra_gate_fields: Dict[str, Any] = {}
    if metrics_gate is not None:
        try:
            extra_gate_fields = dict(p_f1_gate=float(metrics_gate.get('p_f1', 0.0)), s_f1_gate=float(metrics_gate.get('s_f1', 0.0)), time_acc_gate=float(metrics_gate.get('time_acc', 0.0)), p_prec_gate=float(metrics_gate.get('p_prec', 0.0)), p_rec_gate=float(metrics_gate.get('p_rec', 0.0)), s_prec_gate=float(metrics_gate.get('s_prec', 0.0)), s_rec_gate=float(metrics_gate.get('s_rec', 0.0)))
        except Exception:
            extra_gate_fields = {}
    metrics: Dict[str, Any]
    if metrics_mc is not None:
        base = dict(metrics_full)
        for k, v in metrics_mc.items():
            if k in {'time_acc', 'p_prec', 'p_rec', 'p_f1', 's_prec', 's_rec', 's_f1'}:
                continue
            base[k] = v
        metrics = base
    else:
        metrics = dict(metrics_full)
    row_name = case['name']
    if DATA_SOURCE == 'ceed' and (not row_name.endswith('_ceed')):
        row_name = f'{row_name}_ceed'
    row = dict(name=row_name, best_val=best, train_last=tr_hist[-1], valid_last=va_hist[-1], use_cbam=case['use_cbam'], kernels=list(case['kernels']), thr_p=thr_p, thr_s=thr_s, f1_p=float(metrics_full['p_f1']), f1_s=float(metrics_full['s_f1']), **metrics, **extra_unc_fields, **extra_score_fields, **extra_gate_fields)
    print(f'[{case['name']}] 训练和评估完成，最终结果: P-F1={metrics_full['p_f1']:.4f}, S-F1={metrics_full['s_f1']:.4f}', flush=True)
    return row

def log_split_info(case_name: str, train_size: int, valid_size: int, seed_value: int=SEED):
    info_dir = os.path.join(OUT_ROOT, case_name)
    os.makedirs(info_dir, exist_ok=True)
    info = {'case': case_name, 'seed': seed_value, 'timestamp': datetime.now().isoformat(), 'data_source': DATA_SOURCE, 'ceed_dataset': CEED_DATASET_NAME, 'ceed_train_split': CEED_TRAIN_SPLIT, 'ceed_valid_split': CEED_VALID_SPLIT, 'train_samples': train_size, 'valid_samples': valid_size}
    with open(os.path.join(info_dir, 'split_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

def append_csv(path: str, header: list[str], rows: list[Dict[str, Any]]):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            thr_p_str = f'{r['thr_p']:.2f}' if r.get('thr_p') is not None else 'N/A'
            thr_s_str = f'{r['thr_s']:.2f}' if r.get('thr_s') is not None else 'N/A'

            def _fmt_res(key: str) -> str:
                v = r.get(key, None)
                return f'{float(v):.4f}' if v is not None else 'N/A'
            w.writerow([r['name'], f'{r['best_val']:.6f}', f'{r['train_last']:.6f}', f'{r['valid_last']:.6f}', r['use_cbam'], r['kernels'], thr_p_str, thr_s_str, f'{r['time_acc']:.4f}', f'{r.get('mcc', 0.0):.4f}', f'{r['p_prec']:.4f}', f'{r['p_rec']:.4f}', f'{r['p_f1']:.4f}', f'{r['s_prec']:.4f}', f'{r['s_rec']:.4f}', f'{r['s_f1']:.4f}', _fmt_res('p_res_mean_sec'), _fmt_res('p_res_std_sec'), _fmt_res('p_res_mae_sec'), _fmt_res('s_res_mean_sec'), _fmt_res('s_res_std_sec'), _fmt_res('s_res_mae_sec')])
