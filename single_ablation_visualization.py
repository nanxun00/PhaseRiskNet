from __future__ import annotations
import json
import os
from typing import Optional, List, Tuple, Union, Dict
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:
    PCA = None
    StandardScaler = None

def plot_losses(tr, va, out_path: str) -> None:
    if plt is None:
        return
    plt.figure(figsize=(7, 4))
    xs = list(range(1, len(tr) + 1))
    plt.plot(xs, tr, label='train')
    plt.plot(xs, va, label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def estimate_snr(x: np.ndarray, p_idx: Optional[int]=None, s_idx: Optional[int]=None) -> float:
    if len(x.shape) == 2:
        if x.shape[0] < x.shape[1]:
            x = x.T
        x = x.mean(axis=1)
    if p_idx is not None and 0 <= p_idx < len(x):
        signal_window = x[max(0, p_idx - 50):min(len(x), p_idx + 50)]
        signal_power = np.var(signal_window)
    else:
        signal_power = np.max(np.abs(x)) ** 2
    noise_window = x[:min(500, len(x) // 3)]
    noise_power = np.var(noise_window)
    if noise_power < 1e-10:
        return 30.0
    return float(10 * np.log10(signal_power / noise_power))

def plot_one_sample_visual(x_np: np.ndarray, y_np: np.ndarray, probs: np.ndarray, name, snr: float, out_path: str) -> None:
    if plt is None:
        return
    T = probs.shape[-1]
    t = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax0, ax1 = (axes[0], axes[1])
    shift = 3.0
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for c in range(min(3, x_np.shape[0])):
        ax0.plot(t, x_np[c, :T] + c * shift, color=colors[c % 3], label=f'CH{c}')
    if y_np[1].max() > 1e-06:
        ax0.axvline(int(np.argmax(y_np[1])), color='g', linestyle='--', alpha=0.7, label='P_true')
    if y_np[2].max() > 1e-06:
        ax0.axvline(int(np.argmax(y_np[2])), color='b', linestyle='--', alpha=0.7, label='S_true')
    ax0.axvline(int(np.argmax(probs[1])), color='g', linestyle=':', alpha=0.9, label='P_pred')
    ax0.axvline(int(np.argmax(probs[2])), color='b', linestyle=':', alpha=0.9, label='S_pred')
    ax0.set_title(f'{name} (SNR≈{snr:.1f}dB)')
    ax0.legend(loc='upper right', fontsize=9)
    ax1.plot(t, probs[0], label='BG', alpha=0.7)
    ax1.plot(t, probs[1], label='P', linewidth=2)
    ax1.plot(t, probs[2], label='S', linewidth=2)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel('Probability')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_risk_coverage_curve(risk_coverage, out_path: str) -> None:
    if plt is None:
        return
    if isinstance(risk_coverage, dict):
        rc_p = risk_coverage.get('P') or []
        rc_s = risk_coverage.get('S') or []
        rc_p_conf = risk_coverage.get('P_conf') or []
        rc_s_conf = risk_coverage.get('S_conf') or []
        rc_p_rand = risk_coverage.get('P_rand') or []
        rc_s_rand = risk_coverage.get('S_rand') or []
        if not rc_p and (not rc_s) and (not rc_p_conf) and (not rc_s_conf):
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        if rc_p:
            cov_p = [float(r[0]) for r in rc_p]
            risk_p = [float(r[1]) for r in rc_p]
            order = np.argsort(cov_p)[::-1]
            cov_p = [cov_p[i] for i in order]
            risk_p = [risk_p[i] for i in order]
            ax.plot(cov_p, risk_p, marker='o', linestyle='-', label='P (MI)', color='tab:blue')
        if rc_s:
            cov_s = [float(r[0]) for r in rc_s]
            risk_s = [float(r[1]) for r in rc_s]
            order = np.argsort(cov_s)[::-1]
            cov_s = [cov_s[i] for i in order]
            risk_s = [risk_s[i] for i in order]
            ax.plot(cov_s, risk_s, marker='s', linestyle='-', label='S (MI)', color='tab:orange')
        if rc_p_conf:
            cov_p_c = [float(r[0]) for r in rc_p_conf]
            risk_p_c = [float(r[1]) for r in rc_p_conf]
            order = np.argsort(cov_p_c)[::-1]
            cov_p_c = [cov_p_c[i] for i in order]
            risk_p_c = [risk_p_c[i] for i in order]
            ax.plot(cov_p_c, risk_p_c, marker='o', linestyle='--', label='P (1−peak conf)', color='tab:blue', alpha=0.8)
        if rc_s_conf:
            cov_s_c = [float(r[0]) for r in rc_s_conf]
            risk_s_c = [float(r[1]) for r in rc_s_conf]
            order = np.argsort(cov_s_c)[::-1]
            cov_s_c = [cov_s_c[i] for i in order]
            risk_s_c = [risk_s_c[i] for i in order]
            ax.plot(cov_s_c, risk_s_c, marker='s', linestyle='--', label='S (1−peak conf)', color='tab:orange', alpha=0.8)
        if rc_p_rand:
            cov_p_c = [float(r[0]) for r in rc_p_rand]
            risk_p_c = [float(r[1]) for r in rc_p_rand]
            order = np.argsort(cov_p_c)[::-1]
            cov_p_c = [cov_p_c[i] for i in order]
            risk_p_c = [risk_p_c[i] for i in order]
            ax.plot(cov_p_c, risk_p_c, marker='o', linestyle='--', label='P (random)', color='tab:blue', alpha=0.7)
        if rc_s_rand:
            cov_s_c = [float(r[0]) for r in rc_s_rand]
            risk_s_c = [float(r[1]) for r in rc_s_rand]
            order = np.argsort(cov_s_c)[::-1]
            cov_s_c = [cov_s_c[i] for i in order]
            risk_s_c = [risk_s_c[i] for i in order]
            ax.plot(cov_s_c, risk_s_c, marker='s', linestyle='--', label='S (random)', color='tab:orange', alpha=0.7)
        ax.set_xlabel('Coverage')
        ax.set_ylabel('Risk (1−F1, lower is better)')
        ax.set_title('Risk–Coverage (phase-wise)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.15)

        def _pick_at_coverage(cov, risk, target: float) -> float | None:
            if not cov:
                return None
            idx = int(np.argmin(np.abs(np.array(cov) - target)))
            return float(risk[idx])
        if rc_p:
            r1 = _pick_at_coverage(cov_p, risk_p, 1.0)
            r9 = _pick_at_coverage(cov_p, risk_p, 0.9)
            if r1 is not None and r9 is not None:
                f1_full = 1.0 - r1
                f1_09 = 1.0 - r9
                ax.scatter([1.0, 0.9], [r1, r9], color='tab:blue', s=25, zorder=5)
                ax.annotate(f'P-F1: {f1_full:.3f} → {f1_09:.3f} (drop 10%)', xy=(0.9, r9), xytext=(-5, 0), textcoords='offset points', fontsize=8, color='tab:blue', ha='right')
        if rc_s:
            r1 = _pick_at_coverage(cov_s, risk_s, 1.0)
            r9 = _pick_at_coverage(cov_s, risk_s, 0.9)
            if r1 is not None and r9 is not None:
                f1_full = 1.0 - r1
                f1_09 = 1.0 - r9
                ax.scatter([1.0, 0.9], [r1, r9], color='tab:orange', s=25, zorder=5)
                ax.annotate(f'S-F1: {f1_full:.3f} → {f1_09:.3f} (drop 10%)', xy=(0.9, r9), xytext=(-5, 10), textcoords='offset points', fontsize=8, color='tab:orange', ha='right')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return
    if not risk_coverage:
        return
    coverage = [float(r[0]) for r in risk_coverage]
    risk_p = [float(r[1]) for r in risk_coverage]
    risk_s = [float(r[2]) for r in risk_coverage]
    order = np.argsort(coverage)[::-1]
    coverage = [coverage[i] for i in order]
    risk_p = [risk_p[i] for i in order]
    risk_s = [risk_s[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverage, risk_p, marker='o', label='P (MI)', color='tab:blue')
    ax.plot(coverage, risk_s, marker='s', label='S (MI)', color='tab:orange')
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Risk (1−F1, lower is better)')
    ax.set_title('Risk–Coverage (phase-wise)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.15)

    def _pick_at_coverage(cov, risk, target: float) -> float | None:
        if not cov:
            return None
        idx = int(np.argmin(np.abs(np.array(cov) - target)))
        return float(risk[idx])
    r1_p = _pick_at_coverage(coverage, risk_p, 1.0)
    r9_p = _pick_at_coverage(coverage, risk_p, 0.9)
    if r1_p is not None and r9_p is not None:
        ax.scatter([1.0, 0.9], [r1_p, r9_p], color='tab:blue', s=25, zorder=5)
        ax.annotate(f'P@1.0: {r1_p:.3f}', xy=(1.0, r1_p), xytext=(-5, 5), textcoords='offset points', fontsize=8, color='tab:blue', ha='right')
        ax.annotate(f'P@0.9: {r9_p:.3f}', xy=(0.9, r9_p), xytext=(-5, -20), textcoords='offset points', fontsize=8, color='tab:blue', ha='right')
    r1_s = _pick_at_coverage(coverage, risk_s, 1.0)
    r9_s = _pick_at_coverage(coverage, risk_s, 0.9)
    if r1_s is not None and r9_s is not None:
        ax.scatter([1.0, 0.9], [r1_s, r9_s], color='tab:orange', s=25, zorder=5)
        ax.annotate(f'S@1.0: {r1_s:.3f}', xy=(1.0, r1_s), xytext=(-5, 5), textcoords='offset points', fontsize=8, color='tab:orange', ha='right')
        ax.annotate(f'S@0.9: {r9_s:.3f}', xy=(0.9, r9_s), xytext=(-5, 10), textcoords='offset points', fontsize=8, color='tab:orange', ha='right')

    def _annotate_at_coverage(ax_main, cov, risk, label: str, color: str) -> None:
        if not cov:
            return
        target = 0.9
        idx = int(np.argmin(np.abs(np.array(cov) - target)))
        x0, y0 = (cov[idx], risk[idx])
        ax_main.scatter([x0], [y0], color=color, s=25, zorder=5)
        ax_main.annotate(f'{label} @0.9: {y0:.3f}', xy=(x0, y0), xytext=(5, 5), textcoords='offset points', fontsize=8, color=color)
    _annotate_at_coverage(ax, coverage, risk_p, 'P', 'tab:blue')
    _annotate_at_coverage(ax, coverage, risk_s, 'S', 'tab:orange')
    if inset_axes is not None:
        all_risks = risk_p + risk_s
        if all_risks:
            max_risk = float(max(all_risks))
            if max_risk > 0:
                y_max_zoom = min(max_risk * 1.2, 1.0)
                axins = inset_axes(ax, width='45%', height='45%', loc='upper left')
                axins.plot(coverage, risk_p, marker='o', color='tab:blue')
                axins.plot(coverage, risk_s, marker='s', color='tab:orange')
                axins.set_xlim(0.4, 1.0)
                axins.set_ylim(0.0, y_max_zoom)
                axins.grid(True, alpha=0.2)
                axins.tick_params(labelsize=8)
                axins.set_title('Zoomed (low risk)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_uncertainty_error_rate(uncertainty_mc: List[float], error_flag_mc: List[int], out_path: str, num_bins: int=10, uncertainty_conf: Optional[List[float]]=None, error_flag_conf: Optional[List[int]]=None) -> None:
    if plt is None or not uncertainty_mc:
        return
    unc_mc = np.array(uncertainty_mc, dtype=float)
    err_mc = np.array(error_flag_mc, dtype=float)
    if uncertainty_conf is not None and error_flag_conf is not None and (len(uncertainty_conf) == len(error_flag_conf)):
        unc_conf = np.array(uncertainty_conf, dtype=float)
        err_conf = np.array(error_flag_conf, dtype=float)
    else:
        unc_conf = None
        err_conf = None

    def _equal_count_bins(unc: np.ndarray, err: np.ndarray, k: int) -> Tuple[List[float], List[float], List[int]]:
        mask_valid = np.isfinite(unc) & np.isfinite(err)
        unc_valid = unc[mask_valid]
        err_valid = err[mask_valid]
        n_total = int(unc_valid.size)
        if n_total == 0:
            return ([], [], [])
        k_eff = min(k, n_total)
        order = np.argsort(unc_valid)
        err_sorted = err_valid[order]
        x_list: List[float] = []
        err_list: List[float] = []
        n_list: List[int] = []
        for i in range(k_eff):
            start = int(round(i * n_total / k_eff))
            end = int(round((i + 1) * n_total / k_eff))
            if start >= end:
                continue
            seg = err_sorted[start:end]
            if seg.size == 0:
                continue
            x_list.append(float(len(x_list) + 1))
            err_list.append(float(seg.mean()))
            n_list.append(int(seg.size))
        return (x_list, err_list, n_list)
    x_pos_mc, err_rate_mc, n_mc_list = _equal_count_bins(unc_mc, err_mc, num_bins)
    if unc_conf is not None and err_conf is not None:
        x_pos_conf, err_rate_conf, n_conf_list = _equal_count_bins(unc_conf, err_conf, num_bins)
    else:
        x_pos_conf, err_rate_conf, n_conf_list = ([], [], [])
    if not x_pos_mc and (not x_pos_conf):
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    if x_pos_mc:
        ax.plot(x_pos_mc, err_rate_mc, marker='o', color='tab:blue', label='MI (MC Dropout)')
    if x_pos_conf:
        ax.plot(x_pos_conf, err_rate_conf, marker='s', color='tab:orange', linestyle='--', label='1−peak conf (single-pass)')
    all_x = sorted({*x_pos_mc, *x_pos_conf})
    xticks = all_x
    xticklabels = [f'Q{int(x)}' for x in all_x]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Quantiles (Q1=lowest score, Q10=highest score)')
    ax.set_ylabel('Error rate')
    ax.set_title('Error rate vs uncertainty quantiles')
    ax.grid(True, alpha=0.3)
    if x_pos_mc and n_mc_list:
        x1, y1, n1 = (x_pos_mc[0], err_rate_mc[0], n_mc_list[0])
        if np.isfinite(y1) and n1 > 0:
            ax.annotate(f'n={n1}', xy=(x1, y1), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='tab:blue')
        xk, yk, nk = (x_pos_mc[-1], err_rate_mc[-1], n_mc_list[-1])
        if np.isfinite(yk) and nk > 0:
            ax.annotate(f'n={nk}', xy=(xk, yk), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='tab:blue')
    y_vals = [v for v in err_rate_mc if np.isfinite(v)]
    if unc_conf is not None:
        y_vals += [v for v in err_rate_conf if np.isfinite(v)]
    if y_vals:
        y_min = 0.0
        y_max = max(y_vals)
        ax.set_ylim(y_min, min(1.0, y_max * 1.2 + 0.001))
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_uncertainty_overview_grid(risk_coverage: dict, ue: dict, out_path: str, num_bins: int=10) -> None:
    if plt is None:
        return
    if not isinstance(risk_coverage, dict):
        return
    rc_p = risk_coverage.get('P') or []
    rc_s = risk_coverage.get('S') or []
    rc_p_conf = risk_coverage.get('P_conf') or []
    rc_s_conf = risk_coverage.get('S_conf') or []
    rc_p_rand = risk_coverage.get('P_rand') or []
    rc_s_rand = risk_coverage.get('S_rand') or []
    unc_mc = ue.get('uncertainty_mc') or ue.get('uncertainty', [])
    err_flag_mc = ue.get('error_flag', [])
    unc_conf = ue.get('uncertainty_conf')
    err_flag_conf = ue.get('error_flag_conf', err_flag_mc)
    if not unc_mc or not err_flag_mc:
        return
    unc_mc_arr = np.array(unc_mc, dtype=float)
    err_mc_arr = np.array(err_flag_mc, dtype=float)
    unc_conf_arr = np.array(unc_conf, dtype=float) if unc_conf is not None and len(unc_conf) == len(err_flag_conf) else None
    err_conf_arr = np.array(err_flag_conf, dtype=float) if unc_conf_arr is not None else None

    def _equal_count_bins(unc: np.ndarray, err: np.ndarray, k: int) -> Tuple[List[float], List[float], List[int]]:
        mask_valid = np.isfinite(unc) & np.isfinite(err)
        unc_valid = unc[mask_valid]
        err_valid = err[mask_valid]
        n_total = int(unc_valid.size)
        if n_total == 0:
            return ([], [], [])
        k_eff = min(k, n_total)
        order = np.argsort(unc_valid)
        err_sorted = err_valid[order]
        x_list: List[float] = []
        err_list: List[float] = []
        n_list: List[int] = []
        for i in range(k_eff):
            start = int(round(i * n_total / k_eff))
            end = int(round((i + 1) * n_total / k_eff))
            if start >= end:
                continue
            seg = err_sorted[start:end]
            if seg.size == 0:
                continue
            x_list.append(float(len(x_list) + 1))
            err_list.append(float(seg.mean()))
            n_list.append(int(seg.size))
        return (x_list, err_list, n_list)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax_a = axes[0, 0]
    if rc_p or rc_s or rc_p_conf or rc_s_conf:
        if rc_p:
            cov_p = [float(r[0]) for r in rc_p]
            risk_p = [float(r[1]) for r in rc_p]
            order = np.argsort(cov_p)[::-1]
            cov_p = [cov_p[i] for i in order]
            risk_p = [risk_p[i] for i in order]
            ax_a.plot(cov_p, risk_p, marker='o', linestyle='-', color='tab:blue', label='P (MI)')
        if rc_s:
            cov_s = [float(r[0]) for r in rc_s]
            risk_s = [float(r[1]) for r in rc_s]
            order = np.argsort(cov_s)[::-1]
            cov_s = [cov_s[i] for i in order]
            risk_s = [risk_s[i] for i in order]
            ax_a.plot(cov_s, risk_s, marker='s', linestyle='-', color='tab:orange', label='S (MI)')
        if rc_p_conf:
            cov_p_c = [float(r[0]) for r in rc_p_conf]
            risk_p_c = [float(r[1]) for r in rc_p_conf]
            order = np.argsort(cov_p_c)[::-1]
            cov_p_c = [cov_p_c[i] for i in order]
            risk_p_c = [risk_p_c[i] for i in order]
            ax_a.plot(cov_p_c, risk_p_c, marker='o', linestyle='--', color='tab:blue', alpha=0.85, label='P (1−peak conf)')
        if rc_s_conf:
            cov_s_c = [float(r[0]) for r in rc_s_conf]
            risk_s_c = [float(r[1]) for r in rc_s_conf]
            order = np.argsort(cov_s_c)[::-1]
            cov_s_c = [cov_s_c[i] for i in order]
            risk_s_c = [risk_s_c[i] for i in order]
            ax_a.plot(cov_s_c, risk_s_c, marker='s', linestyle='--', color='tab:orange', alpha=0.85, label='S (1−peak conf)')
        ax_a.set_xlabel('Coverage')
        ax_a.set_ylabel('Risk (1−F1)')
        ax_a.set_title('Risk–Coverage (MI vs 1−peak conf)')
        ax_a.grid(True, alpha=0.3)
        ax_a.set_xlim(0, 1)
        ax_a.legend(fontsize=8, handlelength=4.0)
    ax_b = axes[0, 1]
    x_mc, err_mc_bins, _ = _equal_count_bins(unc_mc_arr, err_mc_arr, num_bins)
    x_conf, err_conf_bins, _ = ([], [], [])
    if unc_conf_arr is not None and err_conf_arr is not None:
        x_conf, err_conf_bins, _ = _equal_count_bins(unc_conf_arr, err_conf_arr, num_bins)
    if x_mc:
        ax_b.plot(x_mc, err_mc_bins, marker='o', color='tab:blue', label='MI (MC Dropout)')
    if x_conf:
        ax_b.plot(x_conf, err_conf_bins, marker='s', linestyle='--', color='tab:orange', label='1−peak conf')
    all_x = sorted({*x_mc, *x_conf})
    if all_x:
        ax_b.set_xticks(all_x)
        ax_b.set_xticklabels([f'Q{int(x)}' for x in all_x])
    ax_b.set_xlabel('Quantiles')
    ax_b.set_ylabel('Error rate')
    ax_b.set_title('Error rate distribution across uncertainty quantiles (MI vs 1−peak conf)')
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(fontsize=8, loc='upper left', handlelength=4.0)
    ax_c = axes[1, 0]
    if rc_p or rc_s or rc_p_rand or rc_s_rand:
        if rc_p:
            cov_p = [float(r[0]) for r in rc_p]
            risk_p = [float(r[1]) for r in rc_p]
            order = np.argsort(cov_p)[::-1]
            cov_p = [cov_p[i] for i in order]
            risk_p = [risk_p[i] for i in order]
            ax_c.plot(cov_p, risk_p, marker='o', linestyle='-', color='tab:blue', label='P (MI)')
        if rc_s:
            cov_s = [float(r[0]) for r in rc_s]
            risk_s = [float(r[1]) for r in rc_s]
            order = np.argsort(cov_s)[::-1]
            cov_s = [cov_s[i] for i in order]
            risk_s = [risk_s[i] for i in order]
            ax_c.plot(cov_s, risk_s, marker='s', linestyle='-', color='tab:orange', label='S (MI)')
        if rc_p_rand:
            cov_pr = [float(r[0]) for r in rc_p_rand]
            risk_pr = [float(r[1]) for r in rc_p_rand]
            order = np.argsort(cov_pr)[::-1]
            cov_pr = [cov_pr[i] for i in order]
            risk_pr = [risk_pr[i] for i in order]
            ax_c.plot(cov_pr, risk_pr, marker='o', linestyle='--', color='tab:gray', alpha=0.8, label='P (Random)')
        if rc_s_rand:
            cov_sr = [float(r[0]) for r in rc_s_rand]
            risk_sr = [float(r[1]) for r in rc_s_rand]
            order = np.argsort(cov_sr)[::-1]
            cov_sr = [cov_sr[i] for i in order]
            risk_sr = [risk_sr[i] for i in order]
            ax_c.plot(cov_sr, risk_sr, marker='s', linestyle='--', color='tab:gray', alpha=0.8, label='S (Random)')
        ax_c.set_xlabel('Coverage')
        ax_c.set_ylabel('Risk (1−F1)')
        ax_c.set_title('Risk–Coverage (MI vs Random)')
        ax_c.grid(True, alpha=0.3)
        ax_c.set_xlim(0, 1)
        ax_c.legend(fontsize=8, handlelength=4.0)
    ax_d = axes[1, 1]
    x_mc2, err_mc2, _ = _equal_count_bins(unc_mc_arr, err_mc_arr, num_bins)
    rng = np.random.default_rng(0)
    unc_rand = rng.random(size=err_mc_arr.shape[0])
    x_rand, err_rand_bins, _ = _equal_count_bins(unc_rand, err_mc_arr, num_bins)
    if x_mc2:
        ax_d.plot(x_mc2, err_mc2, marker='o', color='tab:blue', label='MI (MC Dropout)')
    if x_rand:
        ax_d.plot(x_rand, err_rand_bins, marker='^', linestyle='--', color='tab:gray', label='Random')
    all_x2 = sorted({*x_mc2, *x_rand})
    if all_x2:
        ax_d.set_xticks(all_x2)
        ax_d.set_xticklabels([f'Q{int(x)}' for x in all_x2])
    ax_d.set_xlabel('Quantiles')
    ax_d.set_ylabel('Error rate')
    ax_d.set_title('Error rate distribution across uncertainty quantiles (MI vs Random)')
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=8, loc='upper left', handlelength=4.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_failure_sample_one(x_np: np.ndarray, y_np: np.ndarray, probs: np.ndarray, name, snr: float, uncertainty: float, error_type_p: str, error_type_s: str, group: str, subtype: str, out_path: str) -> None:
    if plt is None:
        return
    T = probs.shape[-1]
    t = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax0, ax1 = (axes[0], axes[1])
    shift = 3.0
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for c in range(min(3, x_np.shape[0])):
        ax0.plot(t, x_np[c, :T] + c * shift, color=colors[c % 3], label=f'CH{c}')
    if y_np[1].max() > 1e-06:
        ax0.axvline(int(np.argmax(y_np[1])), color='g', linestyle='--', alpha=0.7, label='P_true')
    if y_np[2].max() > 1e-06:
        ax0.axvline(int(np.argmax(y_np[2])), color='b', linestyle='--', alpha=0.7, label='S_true')
    ax0.axvline(int(np.argmax(probs[1])), color='g', linestyle=':', alpha=0.9, label='P_pred')
    ax0.axvline(int(np.argmax(probs[2])), color='b', linestyle=':', alpha=0.9, label='S_pred')
    if group == 'typical':
        title = f'{name} | SNR={snr:.1f} dB | P:{error_type_p} S:{error_type_s}'
    else:
        title = f'{name} | SNR={snr:.1f} dB | Unc={uncertainty:.4f} | P:{error_type_p} S:{error_type_s}'
    ax0.set_title(title)
    ax0.legend(loc='upper right', fontsize=9)
    ax1.plot(t, probs[0], label='BG', alpha=0.7)
    ax1.plot(t, probs[1], label='P', linewidth=2)
    ax1.plot(t, probs[2], label='S', linewidth=2)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel('Time (samples)')
    ax1.set_ylabel('Probability')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_failure_samples_from_npz(npz_path: str, meta_path: str, out_dir: str) -> None:
    if plt is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    x_np = data['x_np']
    y_np = data['y_np']
    probs = data['probs']
    snr = data['snr']
    uncertainty = data['uncertainty']
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    names = meta['names']
    etp = meta['error_type_p']
    ets = meta['error_type_s']
    groups = meta.get('group', ['typical'] * len(names))
    subtypes = meta.get('subtype', ['unknown'] * len(names))
    n = len(names)
    if n == 0:
        return
    if n != len(x_np) or n != len(etp) or n != len(ets):
        raise ValueError(f'Inconsistent length between failure_samples and meta: npz samples={len(x_np)}, names={n}, etp={len(etp)}, ets={len(ets)}')
    if len(groups) != n or len(subtypes) != n:
        raise ValueError('The length of group / subtype in failure_sample_meta.json is inconsistent with the number of samples.')
    cnt_typ_sfp = cnt_typ_pfn = 0
    cnt_high_wrong = cnt_high_correct = 0
    for i in range(n):
        g = groups[i]
        st = subtypes[i]
        if g == 'typical' and st == 'SFP':
            cnt_typ_sfp += 1
            fname = f'paper_fail_typical_SFP_{cnt_typ_sfp:02d}.png'
        elif g == 'typical' and st == 'PFN':
            cnt_typ_pfn += 1
            fname = f'paper_fail_typical_PFN_{cnt_typ_pfn:02d}.png'
        elif g == 'high_unc' and st == 'wrong':
            cnt_high_wrong += 1
            fname = f'paper_unc_high_wrong_{cnt_high_wrong:02d}.png'
        elif g == 'high_unc' and st == 'correct':
            cnt_high_correct += 1
            fname = f'paper_unc_high_correct_{cnt_high_correct:02d}.png'
        else:
            fname = f'failure_sample_{i:02d}_P{etp[i]}_S{ets[i]}.png'
        out_path = os.path.join(out_dir, fname)
        plot_failure_sample_one(x_np[i], y_np[i], probs[i], names[i], float(snr[i]), float(uncertainty[i]), etp[i], ets[i], g, st, out_path)

def plot_representative_waveforms_grid(samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]], out_path: str) -> None:
    if plt is None or len(samples) != 4:
        return
    from matplotlib import gridspec
    from matplotlib import rcParams
    rcParams['font.size'] = 10
    from matplotlib.lines import Line2D
    sample_rate = 100.0
    channel_names = ['Z', 'N', 'E']
    letters = ['(a)', '(b)', '(c)', '(d)']
    amp_ylim = (-0.5, 0.5)
    prob_ylim = (0.0, 1.0)
    wave_stride = 5
    fig = plt.figure(figsize=(20, 12))
    outer = gridspec.GridSpec(2, 2, wspace=0.15, hspace=0.25)
    for idx, (x_np, y_np, probs, title, snr) in enumerate(samples):
        sub_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[idx], height_ratios=[1, 1, 1, 1], hspace=0.15)
        T = int(probs.shape[-1]) if probs is not None else int(x_np.shape[-1])
        t_sec = np.arange(T) / float(sample_rate)
        p_true_idx = int(np.argmax(y_np[1])) if y_np[1].max() > 1e-06 else None
        s_true_idx = int(np.argmax(y_np[2])) if y_np[2].max() > 1e-06 else None
        p_true_sec = p_true_idx / float(sample_rate) if p_true_idx is not None else None
        s_true_sec = s_true_idx / float(sample_rate) if s_true_idx is not None else None
        if p_true_sec is not None:
            xlim = (p_true_sec - 10.0, p_true_sec + 10.0)
        elif s_true_sec is not None:
            xlim = (s_true_sec - 10.0, s_true_sec + 10.0)
        else:
            xlim = (0.0, 30.0)
        p_pred_idx = None
        s_pred_idx = None
        if probs is not None and getattr(probs, 'ndim', 0) >= 2:
            if probs.shape[0] > 1 and np.max(probs[1]) > 1e-06:
                p_pred_idx = int(np.argmax(probs[1]))
            if probs.shape[0] > 2 and np.max(probs[2]) > 1e-06:
                s_pred_idx = int(np.argmax(probs[2]))
        p_pred_sec = p_pred_idx / float(sample_rate) if p_pred_idx is not None else None
        s_pred_sec = s_pred_idx / float(sample_rate) if s_pred_idx is not None else None
        ax_z = fig.add_subplot(sub_gs[0])
        ax_n = fig.add_subplot(sub_gs[1], sharex=ax_z, sharey=ax_z)
        ax_e = fig.add_subplot(sub_gs[2], sharex=ax_z, sharey=ax_z)
        ax_prob = fig.add_subplot(sub_gs[3], sharex=ax_z)
        axes_wave = [ax_z, ax_n, ax_e]
        for ch in range(3):
            ax = axes_wave[ch]
            if ch < x_np.shape[0]:
                ax.plot(t_sec[::wave_stride], x_np[ch, :T:wave_stride], color='tab:blue', linewidth=0.5)
                if p_true_sec is not None:
                    ax.axvline(p_true_sec, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                if s_true_sec is not None:
                    ax.axvline(s_true_sec, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.set_ylim(amp_ylim)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.text(0.02, 0.92, channel_names[ch], transform=ax.transAxes, fontsize=10, va='top')
            ax.set_yticks([-0.5, 0.0, 0.5])
            ax.minorticks_off()
            ax.tick_params(axis='x', labelbottom=False)
            ax.margins(x=0.01)
            if ch == 0:
                ax.set_title(f'{letters[idx]} {title} (SNR≈{snr:.1f} dB)', fontsize=10)
            ax.set_xlim(xlim)
        if probs is not None and getattr(probs, 'ndim', 0) >= 2 and (probs.shape[0] > 1):
            ax_prob.plot(t_sec, probs[1, :T], color='green', linestyle='-', linewidth=1.2, label='P (pred)')
        if probs is not None and getattr(probs, 'ndim', 0) >= 2 and (probs.shape[0] > 2):
            ax_prob.plot(t_sec, probs[2, :T], color='orange', linestyle='-', linewidth=1.2, label='S (pred)')
        ax_prob.set_ylim(prob_ylim)
        ax_prob.set_ylabel('Probability', fontsize=10)
        ax_prob.minorticks_off()
        ax_prob.margins(x=0.01)
        ax_prob.set_xticks(np.arange(xlim[0], xlim[1] + 1e-06, 5.0))
        ax_prob.set_xlim(xlim)
        if idx >= 2:
            ax_prob.set_xlabel('Time (s)', fontsize=10)
        ax_prob.legend(fontsize=9, loc='upper right', frameon=False)
        for ax_wave in [ax_z, ax_n, ax_e]:
            ax_wave.get_shared_x_axes().join(ax_wave, ax_prob)
        plt.setp([ax_z, ax_n, ax_e, ax_prob], xlim=xlim)
        plt.setp([ax_z, ax_n, ax_e, ax_prob], xticks=np.arange(xlim[0], xlim[1] + 1e-06, 5.0))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_waveform_with_multi_model_probs(x_np: np.ndarray, y_np: np.ndarray, probs_dict: Dict[str, np.ndarray], title: str, snr: float, out_path: str) -> None:
    if plt is None or not probs_dict:
        return
    from matplotlib import gridspec
    T = x_np.shape[-1]
    t = np.arange(T)
    methods = list(probs_dict.keys())
    n_methods = len(methods)
    fig = plt.figure(figsize=(14, 3 + 2 * n_methods))
    outer = gridspec.GridSpec(n_methods + 1, 1, hspace=0.15)
    ax_wave = fig.add_subplot(outer[0])
    shift = 3.0
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    amp = float(np.max(np.abs(x_np[:, :T]))) if x_np.size else 1.0
    ylim = min(max(amp * 1.2, 0.5), 2.0)
    ymin, ymax = (-ylim, ylim)
    for c in range(min(3, x_np.shape[0])):
        ax_wave.plot(t, x_np[c, :T] + c * shift, color=colors[c % 3], label=f'CH{c}')
    ax_wave.set_title(f'{title} (SNR≈{snr:.1f} dB)', fontsize=11)
    ax_wave.grid(True, alpha=0.3)
    ax_wave.set_ylim(ymin, ymax)
    try:
        ax_wave.set_yticks(np.arange(ymin, ymax + 1e-06, 0.5))
    except Exception:
        pass
    ax_wave.legend(loc='upper right', fontsize=8)
    for i, name in enumerate(methods, start=1):
        probs = probs_dict[name]
        ax_prob = fig.add_subplot(outer[i], sharex=ax_wave)
        ax_prob.plot(t, probs[1], label='P_prob', color='g', linewidth=1.5)
        ax_prob.plot(t, probs[2], label='S_prob', color='b', linewidth=1.5)
        ax_prob.set_ylim(0.0, 1.0)
        ax_prob.grid(True, alpha=0.3)
        ax_prob.set_ylabel(name, fontsize=8)
        if i == n_methods:
            ax_prob.set_xlabel('Time (samples)')
        else:
            plt.setp(ax_prob.get_xticklabels(), visible=False)
        if i == 1:
            ax_prob.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_time_residual_distribution(residuals_p: List[float], residuals_s: List[float], out_path: str, sample_rate: float=100.0, bins: int=50, xlim_sec: Optional[float]=None) -> None:
    if plt is None:
        return
    rp_sec = np.array(residuals_p, dtype=float) / sample_rate
    rs_sec = np.array(residuals_s, dtype=float) / sample_rate
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    xlim = xlim_sec
    if xlim is None:
        all_r = np.concatenate([rp_sec, rs_sec]) if rp_sec.size and rs_sec.size else rp_sec if rp_sec.size else rs_sec
        xlim = max(0.1, np.abs(all_r).max() * 1.1) if all_r.size else 0.5
    for ax, data, title in [(axes[0], rp_sec, 'P-wave time residual'), (axes[1], rs_sec, 'S-wave time residual')]:
        if data.size:
            ax.hist(data, bins=bins, color='tab:blue', alpha=0.7, edgecolor='black', density=True)
            ax.axvline(0, color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('Residual (s)')
            abs_data = np.abs(data)
            med_abs = float(np.median(abs_data))
            mae = float(abs_data.mean())
            rmse = float(np.sqrt((data ** 2).mean()))
            p95 = float(np.percentile(abs_data, 95))
            prop_01 = float((abs_data <= 0.1).mean()) * 100.0
            prop_02 = float((abs_data <= 0.2).mean()) * 100.0
            stats_text = f'Median |Δt| = {med_abs:.3f} s\nMAE / RMSE = {mae:.3f} / {rmse:.3f} s\n95% |Δt| = {p95:.3f} s\n|Δt|≤0.1s: {prop_01:.1f}%  |Δt|≤0.2s: {prop_02:.1f}%'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.set_xlim(-xlim, xlim)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Time residual distribution (predicted − ground truth)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_pca_visualization(features: np.ndarray, labels: np.ndarray, out_path: str, label_names: Optional[List[str]]=None, n_components: int=2, max_samples: int=5000) -> None:
    if plt is None:
        return
    if PCA is None:
        print('PCA visualization skipped (sklearn not installed, from sklearn.decomposition import PCA failed)')
        return
    if features.size == 0 or len(features) < 2:
        return
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(X) > max_samples:
        rng = np.random.default_rng(2024)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = (X[idx], y[idx])
    if 'StandardScaler' in globals() and StandardScaler is not None:
        X = StandardScaler().fit_transform(X)
    else:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-08
        X = (X - mean) / std
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
    X2 = pca.fit_transform(X)
    if X2.shape[1] < 2:
        return
    names = label_names or ['BG window', 'P window', 'S window']
    uniq = np.unique(y)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['tab:gray', 'tab:blue', 'tab:orange']
    for i, u in enumerate(uniq):
        mask = y == u
        ax.scatter(X2[mask, 0], X2[mask, 1], label=names[u] if u < len(names) else str(u), alpha=0.3, s=10, c=colors[u % len(colors)])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    try:
        var_ratio = pca.explained_variance_ratio_
        pc1_var = float(var_ratio[0]) * 100.0
        pc2_var = float(var_ratio[1]) * 100.0 if var_ratio.size > 1 else 0.0
        ax.set_title(f'PCA of prediction windows (PC1 {pc1_var:.1f}%, PC2 {pc2_var:.1f}%)')
    except Exception:
        ax.set_title('PCA of prediction windows (BG / P / S)')
    x_vals = X2[:, 0]
    y_vals = X2[:, 1]
    try:
        x_min, x_max = np.percentile(x_vals, [1, 99])
        y_min, y_max = np.percentile(y_vals, [1, 99])
        dx = max(0.001, (x_max - x_min) * 0.05)
        dy = max(0.001, (y_max - y_min) * 0.05)
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)
    except Exception:
        pass
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_max_prob_histogram(pr_data: dict, out_path: str, bins: int=15) -> None:
    if plt is None:
        return
    max_p = np.asarray(pr_data['max_prob_p'], dtype=float)
    max_s = np.asarray(pr_data['max_prob_s'], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in [(axes[0], max_p, 'max_prob_p'), (axes[1], max_s, 'max_prob_s')]:
        ax.hist(data, bins=bins, color='tab:blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('max probability')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Distribution of max probability (P/S channels)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _pr_curve_from_scores(has_label: np.ndarray, correct: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    has_label = np.asarray(has_label, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    precisions = []
    recalls = []
    f1s = []
    for t in thresholds:
        pred = scores >= t
        tp = np.sum(pred & has_label & correct)
        fp = np.sum(pred & (~has_label | ~correct))
        fn = np.sum(has_label & (~pred | ~correct))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return (np.array(precisions), np.array(recalls), np.array(f1s))

def plot_pr_curve(pr_data: dict, out_path: str, n_thresholds: int=80) -> None:
    if plt is None:
        return
    has_p = np.asarray(pr_data['has_p'], dtype=bool)
    has_s = np.asarray(pr_data['has_s'], dtype=bool)
    max_p = np.asarray(pr_data['max_prob_p'], dtype=float)
    max_s = np.asarray(pr_data['max_prob_s'], dtype=float)
    p_ok = np.asarray(pr_data['p_ok'], dtype=bool)
    s_ok = np.asarray(pr_data['s_ok'], dtype=bool)
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    prec_p, rec_p, f1_p = _pr_curve_from_scores(has_p, p_ok, max_p, thresholds)
    prec_s, rec_s, f1_s = _pr_curve_from_scores(has_s, s_ok, max_s, thresholds)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, thresh, prec, rec, f1, title in [(axes[0], thresholds, prec_p, rec_p, f1_p, 'P-wave'), (axes[1], thresholds, prec_s, rec_s, f1_s, 'S-wave')]:
        ax.plot(thresh, prec, label='Precision', color='tab:blue')
        ax.plot(thresh, rec, label='Recall', color='tab:orange')
        ax.plot(thresh, f1, label='F1', color='tab:green', linestyle='--')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'Precision–Recall vs threshold ({title})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
    fig.suptitle('Precision–Recall vs threshold (imbalanced-class friendly)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_snr_stratified(snr_data: dict, out_path: str, snr_bins: Optional[List[float]]=None, fixed_thr: float=0.5, snr_reliable_only: bool=True) -> None:
    if plt is None:
        return
    has_p = np.asarray(snr_data['has_p'], dtype=bool)
    has_s = np.asarray(snr_data['has_s'], dtype=bool)
    max_p = np.asarray(snr_data['max_prob_p'], dtype=float)
    max_s = np.asarray(snr_data['max_prob_s'], dtype=float)
    p_ok = np.asarray(snr_data['p_ok'], dtype=bool)
    s_ok = np.asarray(snr_data['s_ok'], dtype=bool)
    snr = np.asarray(snr_data['snr'], dtype=float)
    if snr_reliable_only:
        reliable = has_p
        if reliable.sum() == 0:
            return
    else:
        reliable = np.ones(len(snr), dtype=bool)
    if snr_bins is None:
        snr_bins = [0, 8, 12, 16, 20, 25, 100]
    snr_bins = np.asarray(snr_bins)
    n_bins = len(snr_bins) - 1
    pred_p = max_p >= fixed_thr
    pred_s = max_s >= fixed_thr
    prec_bins_p, rec_bins_p, f1_bins_p = ([], [], [])
    prec_bins_s, rec_bins_s, f1_bins_s = ([], [], [])
    bin_labels = []
    for i in range(n_bins):
        lo, hi = (snr_bins[i], snr_bins[i + 1])
        mask = (snr >= lo) & (snr < hi) & reliable
        if mask.sum() == 0:
            prec_bins_p.append(np.nan)
            rec_bins_p.append(np.nan)
            f1_bins_p.append(np.nan)
            prec_bins_s.append(np.nan)
            rec_bins_s.append(np.nan)
            f1_bins_s.append(np.nan)
        else:
            tp_p = np.sum(pred_p[mask] & has_p[mask] & p_ok[mask])
            fp_p = np.sum(pred_p[mask] & (~has_p[mask] | ~p_ok[mask]))
            fn_p = np.sum(has_p[mask] & (~pred_p[mask] | ~p_ok[mask]))
            prec_p = tp_p / (tp_p + fp_p + 1e-12)
            rec_p = tp_p / (tp_p + fn_p + 1e-12)
            f1_p = 2 * prec_p * rec_p / (prec_p + rec_p + 1e-12)
            tp_s = np.sum(pred_s[mask] & has_s[mask] & s_ok[mask])
            fp_s = np.sum(pred_s[mask] & (~has_s[mask] | ~s_ok[mask]))
            fn_s = np.sum(has_s[mask] & (~pred_s[mask] | ~s_ok[mask]))
            prec_s = tp_s / (tp_s + fp_s + 1e-12)
            rec_s = tp_s / (tp_s + fn_s + 1e-12)
            f1_s = 2 * prec_s * rec_s / (prec_s + rec_s + 1e-12)
            prec_bins_p.append(prec_p)
            rec_bins_p.append(rec_p)
            f1_bins_p.append(f1_p)
            prec_bins_s.append(prec_s)
            rec_bins_s.append(rec_s)
            f1_bins_s.append(f1_s)
        bin_labels.append(f'{lo:.0f}-{hi:.0f}')
    x = np.arange(n_bins)
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * w, prec_bins_p, width=w, label='P Precision', color='tab:blue', alpha=0.8)
    ax.bar(x - 0.5 * w, rec_bins_p, width=w, label='P Recall', color='tab:blue', alpha=0.4)
    ax.bar(x + 0.5 * w, prec_bins_s, width=w, label='S Precision', color='tab:orange', alpha=0.8)
    ax.bar(x + 1.5 * w, rec_bins_s, width=w, label='S Recall', color='tab:orange', alpha=0.4)
    ax.plot(x, f1_bins_p, 'o-', color='tab:blue', linewidth=2, label='P F1')
    ax.plot(x, f1_bins_s, 's-', color='tab:orange', linewidth=2, label='S F1')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Score')
    ax.set_title(f'Performance by SNR (threshold={fixed_thr})' + (' [has_p only]' if snr_reliable_only else ''))
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
_COMPARE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
_COMPARE_LINESTYLES = ['-', '--', '-.', ':']

def plot_pr_curve_compare(methods: List[Tuple[str, dict]], out_path: str, n_thresholds: int=80) -> None:
    if plt is None or not methods:
        return
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ch_name in [(axes[0], 'P'), (axes[1], 'S')]:
        has_key = 'has_p' if ch_name == 'P' else 'has_s'
        ok_key = 'p_ok' if ch_name == 'P' else 's_ok'
        score_key = 'max_prob_p' if ch_name == 'P' else 'max_prob_s'
        for idx, (name, data) in enumerate(methods):
            has_l = np.asarray(data[has_key], dtype=bool)
            ok = np.asarray(data[ok_key], dtype=bool)
            sc = np.asarray(data[score_key], dtype=float)
            prec, rec, _ = _pr_curve_from_scores(has_l, ok, sc, thresholds)
            c = _COMPARE_COLORS[idx % len(_COMPARE_COLORS)]
            ls = _COMPARE_LINESTYLES[idx % len(_COMPARE_LINESTYLES)]
            ax.plot(thresholds, prec, color=c, linestyle=ls, label=f'{name} Prec')
            ax.plot(thresholds, rec, color=c, linestyle=ls, alpha=0.7, label=f'{name} Rec')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{ch_name}-wave Precision & Recall vs threshold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
    fig.suptitle('Precision–Recall vs threshold (method comparison)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_snr_stratified_compare(methods: List[Tuple[str, dict]], out_path: str, snr_bins: Optional[List[float]]=None, fixed_thr: float=0.5, snr_reliable_only: bool=True) -> None:
    if plt is None or not methods:
        return
    if snr_bins is None:
        snr_bins = [0, 8, 12, 16, 20, 25, 100]
    snr_bins = np.asarray(snr_bins)
    n_bins = len(snr_bins) - 1
    bin_labels = [f'{snr_bins[i]:.0f}-{snr_bins[i + 1]:.0f}' for i in range(n_bins)]
    x = np.arange(n_bins)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (name, data), idx in zip(methods, range(len(methods))):
        has_p = np.asarray(data['has_p'], dtype=bool)
        has_s = np.asarray(data['has_s'], dtype=bool)
        max_p = np.asarray(data['max_prob_p'], dtype=float)
        max_s = np.asarray(data['max_prob_s'], dtype=float)
        p_ok = np.asarray(data['p_ok'], dtype=bool)
        s_ok = np.asarray(data['s_ok'], dtype=bool)
        snr = np.asarray(data['snr'], dtype=float)
        reliable = has_p if snr_reliable_only else np.ones(len(snr), dtype=bool)
        pred_p = max_p >= fixed_thr
        pred_s = max_s >= fixed_thr
        f1_p_bins, f1_s_bins = ([], [])
        for i in range(n_bins):
            mask = (snr >= snr_bins[i]) & (snr < snr_bins[i + 1]) & reliable
            if mask.sum() == 0:
                f1_p_bins.append(np.nan)
                f1_s_bins.append(np.nan)
            else:
                tp_p = np.sum(pred_p[mask] & has_p[mask] & p_ok[mask])
                fp_p = np.sum(pred_p[mask] & (~has_p[mask] | ~p_ok[mask]))
                fn_p = np.sum(has_p[mask] & (~pred_p[mask] | ~p_ok[mask]))
                prec_p = tp_p / (tp_p + fp_p + 1e-12)
                rec_p = tp_p / (tp_p + fn_p + 1e-12)
                f1_p_bins.append(2 * prec_p * rec_p / (prec_p + rec_p + 1e-12))
                tp_s = np.sum(pred_s[mask] & has_s[mask] & s_ok[mask])
                fp_s = np.sum(pred_s[mask] & (~has_s[mask] | ~s_ok[mask]))
                fn_s = np.sum(has_s[mask] & (~pred_s[mask] | ~s_ok[mask]))
                prec_s = tp_s / (tp_s + fp_s + 1e-12)
                rec_s = tp_s / (tp_s + fn_s + 1e-12)
                f1_s_bins.append(2 * prec_s * rec_s / (prec_s + rec_s + 1e-12))
        c = _COMPARE_COLORS[idx % len(_COMPARE_COLORS)]
        ls = _COMPARE_LINESTYLES[idx % len(_COMPARE_LINESTYLES)]
        axes[0].plot(x, f1_p_bins, 'o-', color=c, linestyle=ls, linewidth=2, label=name)
        axes[1].plot(x, f1_s_bins, 's-', color=c, linestyle=ls, linewidth=2, label=name)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('F1')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_title('P-wave F1 by SNR')
    axes[1].set_title('S-wave F1 by SNR')
    fig.suptitle(f'Performance by SNR (threshold={fixed_thr})' + (' [has_p only]' if snr_reliable_only else '') + ' (method comparison)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_time_residual_distribution_compare(methods: List[Tuple[str, List[float], List[float]]], out_path: str, sample_rate: float=100.0, bins: int=50, xlim_sec: Optional[float]=None) -> None:
    if plt is None or not methods:
        return
    all_r = []
    for _, rp, rs in methods:
        all_r.extend(np.array(rp, dtype=float) / sample_rate)
        all_r.extend(np.array(rs, dtype=float) / sample_rate)
    xlim = xlim_sec
    if xlim is None and all_r:
        xlim = max(0.1, np.abs(np.array(all_r)).max() * 1.1)
    elif xlim is None:
        xlim = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (name, rp, rs), idx in zip(methods, range(len(methods))):
        rp_sec = np.array(rp, dtype=float) / sample_rate
        rs_sec = np.array(rs, dtype=float) / sample_rate
        c = _COMPARE_COLORS[idx % len(_COMPARE_COLORS)]
        if rp_sec.size:
            axes[0].hist(rp_sec, bins=bins, alpha=0.5, density=True, label=name, color=c, edgecolor='none')
        if rs_sec.size:
            axes[1].hist(rs_sec, bins=bins, alpha=0.5, density=True, label=name, color=c, edgecolor='none')
    for ax, title in [(axes[0], 'P-wave time residual'), (axes[1], 'S-wave time residual')]:
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Residual (s)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.set_xlim(-xlim, xlim)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle('Time residual distribution (predicted − GT, method comparison)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_time_residual_grid(methods: List[Tuple[str, List[float], List[float]]], out_path: str, sample_rate: float=100.0, bins: int=40, xlim_sec: float=0.3) -> None:
    if plt is None or not methods:
        return
    n_methods = len(methods)
    fig, axes = plt.subplots(2, n_methods, figsize=(3.2 * n_methods, 5.5))
    if n_methods == 1:
        axes_p = np.array([axes[0]])
        axes_s = np.array([axes[1]])
    else:
        axes_p = axes[0]
        axes_s = axes[1]
    xlim = float(xlim_sec)
    for idx, (name, rp, rs) in enumerate(methods):
        ax_p = axes_p[idx]
        ax_s = axes_s[idx]
        rp_sec = np.asarray(rp, dtype=float) / float(sample_rate)
        rs_sec = np.asarray(rs, dtype=float) / float(sample_rate)
        if rp_sec.size:
            mu_p = float(rp_sec.mean())
            sigma_p = float(rp_sec.std())
            ax_p.hist(rp_sec, bins=bins, color='tab:blue', alpha=0.8, edgecolor='black')
            ax_p.axvline(0.0, color='black', linestyle='--', linewidth=1)
            ax_p.text(0.03, 0.95, f'$\\mu$ = {mu_p:.3f}\n$\\sigma$ = {sigma_p:.3f}', transform=ax_p.transAxes, ha='left', va='top', fontsize=9)
        ax_p.set_xlim(-xlim, xlim)
        ax_p.set_ylabel('Frequency')
        if idx == 0:
            ax_p.set_title('P residuals', fontsize=10)
        else:
            ax_p.set_title('')
        ax_p.grid(True, alpha=0.3)
        if rs_sec.size:
            mu_s = float(rs_sec.mean())
            sigma_s = float(rs_sec.std())
            ax_s.hist(rs_sec, bins=bins, color='tab:orange', alpha=0.8, edgecolor='black')
            ax_s.axvline(0.0, color='black', linestyle='--', linewidth=1)
            ax_s.text(0.03, 0.95, f'$\\mu$ = {mu_s:.3f}\n$\\sigma$ = {sigma_s:.3f}', transform=ax_s.transAxes, ha='left', va='top', fontsize=9)
        ax_s.set_xlim(-xlim, xlim)
        ax_s.set_xlabel('Residual (s)')
        ax_s.set_ylabel('Frequency')
        ax_s.set_title(name, fontsize=10)
        ax_s.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_pca_visualization_compare(methods: List[Tuple[str, np.ndarray, np.ndarray]], out_path: str, label_names: Optional[List[str]]=None, n_components: int=2, max_samples_per_method: int=3000) -> None:
    if plt is None:
        return
    if PCA is None:
        print('PCA 对比图跳过（sklearn 未安装，from sklearn.decomposition import PCA 失败）')
        return
    if not methods:
        return
    names = label_names or ['No phase', 'P only', 'S only', 'P+S']
    n_methods = len(methods)
    n_col = min(2, n_methods)
    n_row = (n_methods + n_col - 1) // n_col
    fig, axes = plt.subplots(n_row, n_col, figsize=(6 * n_col, 5 * n_row))
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    colors = ['tab:gray', 'tab:blue', 'tab:orange', 'tab:green']
    for idx, (method_name, features, labels) in enumerate(methods):
        ax = axes[idx]
        X = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if len(X) > max_samples_per_method:
            rng = np.random.default_rng(2024 + idx)
            ii = rng.choice(len(X), max_samples_per_method, replace=False)
            X, y = (X[ii], y[ii])
        if len(X) < 2:
            ax.set_title(method_name)
            continue
        pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0] - 1))
        X2 = pca.fit_transform(X)
        if X2.shape[1] < 2:
            ax.set_title(method_name)
            continue
        for u in np.unique(y):
            mask = y == u
            ax.scatter(X2[mask, 0], X2[mask, 1], label=names[u] if u < len(names) else str(u), alpha=0.5, s=8, c=colors[u % len(colors)])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(method_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(len(methods), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('PCA of prediction features (method comparison)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
