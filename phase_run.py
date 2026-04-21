import os
import argparse
from typing import Dict
from phase_core import run_case, append_csv, METRICS_CSV, METRICS_HEADER, SEED
import phase_core as core
from utils.repro import seed_everything

def print_separator(title: str=''):
    print('\n' + '=' * 80)
    if title:
        print(f'  {title}')
        print('=' * 80)
    print()

ABLATION_CONFIGS: Dict[str, Dict] = {'phasenet_full_big': {'name': 'phasenet_full_big', 'model_class': 'PhaseNetUNet', 'depths': 5, 'filters_root': 8, 'kernels': (3, 7, 15), 'pool_size': 4, 'drop_rate': 0.1, 'use_cbam': True, 'use_separable': False, 'fusion_mode': 'softgate_residual', 'fusion_gate_hidden': 16, 'fusion_residual_scale': 0.3, 'fusion_use_maxpool': True, 'softgate_scope': 'deep', 'use_phasewise_loss': False, 'use_ttversky_loss': True, 'tt_loss_weight': 0.5, 'tt_time_weight': 0.3, 'tt_start_weight': 0.4, 'tt_temporal_att_weight': 0.1, 'tt_start_window': 2, 'tt_start_peak_threshold': 0.2, 'tt_alpha_p': 0.7, 'tt_beta_p': 0.3, 'tt_alpha_s': 0.8, 'tt_beta_s': 0.2, 'fixed_thr_p': 0.5, 'fixed_thr_s': 0.5, 'use_mc_dropout_selective': True, 'mc_dropout_n_samples': 20, 'mc_selective_drop_ratio': 0.1, 'mc_risk_coverage_points': 20, 'use_temporal_bifpn_asff': True, 'eval_only': False}, 'phasenet_full_small': {'name': 'phasenet_full_small', 'model_class': 'PhaseNetUNet', 'depths': 5, 'filters_root': 8, 'kernels': (3, 7, 15), 'pool_size': 4, 'drop_rate': 0.1, 'use_cbam': True, 'use_separable': True, 'fusion_mode': 'softgate_residual', 'fusion_gate_hidden': 16, 'fusion_residual_scale': 0.3, 'fusion_use_maxpool': True, 'softgate_scope': 'deep', 'use_phasewise_loss': False, 'use_ttversky_loss': True, 'tt_loss_weight': 0.5, 'tt_time_weight': 0.3, 'tt_start_weight': 0.4, 'tt_temporal_att_weight': 0.1, 'tt_start_window': 2, 'tt_start_peak_threshold': 0.2, 'tt_alpha_p': 0.7, 'tt_beta_p': 0.3, 'tt_alpha_s': 0.8, 'tt_beta_s': 0.2, 'fixed_thr_p': 0.5, 'fixed_thr_s': 0.5, 'use_mc_dropout_selective': True, 'mc_dropout_n_samples': 20, 'mc_selective_drop_ratio': 0.1, 'mc_risk_coverage_points': 20, 'use_temporal_bifpn_asff': True, 'eval_only': False}}


def run_test(config: Dict, quick_mode: bool=False) -> Dict:
    print_separator(f'Test: {config['name']}')
    print(f'[{config['name']}] Preparing data and model...', flush=True)
    if quick_mode:
        print('⚠️  Quick mode: small dataset + few epochs', flush=True)
        _ep = core.EPOCHS
        core.EPOCHS = 5
        try:
            result = run_case(config)
        finally:
            core.EPOCHS = _ep
    else:
        result = run_case(config)
    print(f'[{config['name']}] Completed.', flush=True)
    return result

def main():
    print('=' * 80, flush=True)
    print('adaptive/phase_run.py started (PhaseNetUNet backbone)', flush=True)
    print(f'Working directory: {os.getcwd()}', flush=True)
    print('=' * 80, flush=True)
    parser = argparse.ArgumentParser(description='PhaseNetUNet backbone ablation test')
    parser.add_argument('--quick', action='store_true', help='Quick test (small data, few epochs)')
    parser.add_argument('--gpu', type=str, default=None, help='GPU ID, such as 0 or 0,1,2,3 (set CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--include-ablation', action='store_true', help='Whether to run ablation configurations (excluding baseline)')
    parser.add_argument('--ablation-keys', type=str, default=None, help='Run only specified keys, comma-separated')
    parser.add_argument('--skip-baseline', action='store_true', help='Run only ablation configurations, skip baseline')
    parser.add_argument('--seed', type=int, default=SEED, help='Global random seed')
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f'[phase] Specified visible GPU: {args.gpu}', flush=True)
    core.SEED = args.seed
    print(f'[phase] Using random seed: {args.seed}', flush=True)
    seed_everything(args.seed, deterministic=True)
    results: Dict[str, Dict] = {}
    if args.skip_baseline:
        print('Step 1: --skip-baseline specified, skipping PhaseNet Baseline.', flush=True)
    elif 'phasenet_baseline' not in ABLATION_CONFIGS:
        print('Step 1: phasenet_baseline configuration not found, automatically skipping baseline.', flush=True)
    else:
        print_separator('Step 1: PhaseNet Baseline')
        results['baseline'] = run_test(ABLATION_CONFIGS['phasenet_baseline'], quick_mode=args.quick)
        append_csv(METRICS_CSV, METRICS_HEADER, [results['baseline']])
    test_configs: Dict[str, Dict] = {}
    if args.include_ablation:
        print_separator('Step 2: Ablation Experiments (PhaseNet Extended)')
        if args.ablation_keys:
            keys = [k.strip() for k in args.ablation_keys.split(',') if k.strip()]
            for k in keys:
                if k in ABLATION_CONFIGS and k != 'phasenet_baseline':
                    test_configs[k] = ABLATION_CONFIGS[k]
                else:
                    print(f"Warning: Unknown or skipped key '{k}'")
            print(f'Will run specified {len(test_configs)} ablation configurations.')
        else:
            test_configs = {k: v for k, v in ABLATION_CONFIGS.items() if k != 'phasenet_baseline'}
            print(f'Will run all {len(test_configs)} ablation configurations (default when --ablation-keys is not specified).')
    else:
        print('Step 2: --include-ablation not specified, skipping ablation.')
    for key, cfg in test_configs.items():
        print(f'\nTest: {key}')
        results[key] = run_test(cfg, quick_mode=args.quick)
        append_csv(METRICS_CSV, METRICS_HEADER, [results[key]])
    print_separator('Testing complete')


if __name__ == '__main__':
    main()
