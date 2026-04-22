from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None
from data import normalize_zero_mean_unit_std, make_gaussian_window, make_gaussian_window_sigma

class CEEDDataset(Dataset):

    def __init__(self, dataset_name: str='CEED.py', split: str='train', limit: int | None=10000, waveform_key: str | None='data', p_key: str | None=None, s_key: str | None=None, sampling_rate: float=100.0, crop_len: int=3000, label_sigma_sec: float | None=0.1, label_width: int=51, training: bool=True, local_dir: str | None=None, use_waveform_augmentation: bool=False, aug_noise_snr_db_min: float=3.0, aug_noise_snr_db_max: float=20.0, aug_amplitude_min: float=0.5, aug_amplitude_max: float=2.0):
        if load_dataset is None:
            raise ImportError('datasets is not installed, please pip install datasets')
        self.crop_len = int(crop_len)
        self.training = bool(training)
        self.sampling_rate = float(sampling_rate)
        self.waveform_key = waveform_key
        self.p_key = p_key
        self.s_key = s_key
        if local_dir:
            import os
            from glob import glob
            h5_files = sorted(glob(os.path.join(local_dir, '*.h5')))
            if h5_files:
                print(f'Using local dataset: {local_dir}')
                print(f"Found {len(h5_files)} H5 files: {[os.path.basename(f) for f in h5_files[:3]]}{('...' if len(h5_files) > 3 else '')}")
            else:
                print(f'Warning: No .h5 files found in local directory {local_dir}')
                print(f'Will attempt to download from network (if available)')
        ds = None
        errors = []
        base_kwargs = {'name': 'event', 'trust_remote_code': True}
        if local_dir:
            base_kwargs['data_dir'] = local_dir
            base_kwargs['download_mode'] = 'reuse_cache_if_exists'
        try:
            if local_dir:
                print(f'Loading dataset from local path: {local_dir}')
                print("(Note: 'Downloading and preparing' is the default message from datasets library, using local data)")
            import os
            old_env = os.environ.get('HF_DATASETS_VERBOSITY', None)
            if local_dir:
                os.environ['HF_DATASETS_VERBOSITY'] = 'error'
            try:
                ds = load_dataset(dataset_name, split=split, **base_kwargs)
                if ds is not None:
                    try:
                        _ = len(ds)
                    except (NotImplementedError, OSError) as access_err:
                        if 'LocalFileSystem' in str(access_err):
                            print('Warning: Dataset loaded but encountered LocalFileSystem error on access, retrying...')
                            raise access_err
            except OSError as os_err:
                if 'Cannot find data file' in str(os_err) or '.arrow' in str(os_err):
                    if local_dir:
                        import os
                        import shutil
                        cache_dir = os.path.expanduser('~/.cache/huggingface/datasets/ceed/')
                        print('=' * 60)
                        print('Detected missing or incomplete cache files')
                        print('=' * 60)
                        print(f'Cache directory: {cache_dir}')
                        if os.path.exists(cache_dir):
                            try:
                                cache_size = sum((os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(cache_dir) for filename in filenames)) / 1024 ** 3
                                print(f'Cache directory size: {cache_size:.2f} GB')
                                arrow_files = []
                                for root, dirs, files in os.walk(cache_dir):
                                    for f in files:
                                        if f.endswith('.arrow'):
                                            arrow_files.append(os.path.join(root, f))
                                if arrow_files:
                                    print(f'Found {len(arrow_files)} .arrow files (may be incomplete)')
                                    print(f"Example: {(arrow_files[0] if arrow_files else 'N/A')}")
                                else:
                                    print('No .arrow files found')
                            except Exception as diag_err:
                                print(f'Unable to diagnose cache directory: {diag_err}')
                        else:
                            print('Cache directory does not exist')
                        print('\nPossible causes:')
                        print('  1. Generation process was interrupted (insufficient memory, disk space, process killed, etc.)')
                        print('  2. Concurrent access from multiple processes corrupted cache files')
                        print('  3. Insufficient disk space caused write failure')
                        print('  4. Permission issues caused write failure')
                        print('\nSolution:')
                        print('  Attempting to force regenerate cache...')
                        print('=' * 60)
                        base_kwargs_retry = base_kwargs.copy()
                        base_kwargs_retry['download_mode'] = 'force_redownload'
                        try:
                            ds = load_dataset(dataset_name_to_use, split=split, **base_kwargs_retry)
                            print('✓ Cache regeneration successful!')
                        except Exception as retry_err:
                            retry_err_str = str(retry_err)
                            if 'Cannot find data file' in retry_err_str or '.arrow' in retry_err_str or 'NNNNN' in retry_err_str:
                                print('\nDetected incomplete cache files (filename contains NNNNN or file missing)')
                                print('Cleaning damaged cache and regenerating...')
                                try:
                                    import shutil
                                    if os.path.exists(cache_dir):
                                        shutil.rmtree(cache_dir)
                                        print(f'✓ Cleaned damaged cache: {cache_dir}')
                                    base_kwargs_final = base_kwargs.copy()
                                    base_kwargs_final['download_mode'] = 'force_redownload'
                                    ds = load_dataset(dataset_name_to_use, split=split, **base_kwargs_final)
                                    print('✓ Cache regeneration successful!')
                                except Exception as final_err:
                                    print('\nStill fails after cleaning cache, please check:')
                                    print('  1. Is disk space sufficient (at least 10 GB needed)')
                                    print('  2. Do you have write permission')
                                    print('  3. Is the disk full')
                                    raise RuntimeError(f'Unable to load dataset. Still fails even after cleaning cache.\\nPlease check system resources (disk space, permissions, etc.).\\nCache directory: {cache_dir}\\nOriginal error: {os_err}\\nRetry error: {retry_err}\\nFinal error: {final_err}') from final_err
                            else:
                                print('\nForce regeneration failed (cache file issue)')
                                print(f'Error: {retry_err}')
                                raise RuntimeError(f'Unable to load dataset.\\nOriginal error: {os_err}\\nRetry error: {retry_err}') from retry_err
                    else:
                        raise
                else:
                    raise
            finally:
                if old_env is None:
                    os.environ.pop('HF_DATASETS_VERBOSITY', None)
                else:
                    os.environ['HF_DATASETS_VERBOSITY'] = old_env
            if local_dir:
                print(f'✓ Successfully loaded local dataset with {len(ds)} samples')
        except NotImplementedError as e:
            import os
            old_env = os.environ.get('HF_DATASETS_VERBOSITY', None)
            if old_env is None:
                os.environ.pop('HF_DATASETS_VERBOSITY', None)
            else:
                os.environ['HF_DATASETS_VERBOSITY'] = old_env
            if 'LocalFileSystem' in str(e):
                cache_dir = os.path.expanduser('~/.cache/huggingface/datasets/ceed/')
                if local_dir:
                    import glob
                    arrow_pattern = os.path.join(cache_dir, '**', '*.arrow')
                    existing_arrows = glob.glob(arrow_pattern, recursive=True)
                    if existing_arrows:
                        print(f'Detected LocalFileSystem error, but found {len(existing_arrows)} existing cache files')
                        print('Attempting to load dataset directly (without regeneration)...')
                        try:
                            base_kwargs_direct = {k: v for k, v in base_kwargs.items() if k != 'download_mode'}
                            if not is_new_version and 'trust_remote_code' not in base_kwargs_direct:
                                base_kwargs_direct['trust_remote_code'] = True
                            ds = load_dataset(dataset_name_to_use, split=split, **base_kwargs_direct)
                            print('✓ Successfully loaded using existing cache!')
                        except Exception as direct_err:
                            print(f'Direct load failed: {direct_err}')
                            print('Cache may be incomplete or corrupted, needs regeneration...')
                            try:
                                base_kwargs_retry = base_kwargs.copy()
                                base_kwargs_retry['download_mode'] = 'force_redownload'
                                ds = load_dataset(dataset_name_to_use, split=split, **base_kwargs_retry)
                                print('✓ Cache re-generation successful！')
                            except Exception as e2:
                                raise RuntimeError(f'Unable to load local dataset, cache may be corrupted.\\nPlease manually clean cache (no root permission needed):\\n  rm -rf {cache_dir}\\nThen run again.\\nLocal data path: {local_dir}\\nOriginal error: {e}\\nDirect load error: {direct_err}\\nRegeneration error: {e2}') from e2
                    else:
                        print(f'Detected LocalFileSystem cache error, attempting to force regenerate cache...')
                        print(f'(If it fails, you may need to manually clean cache: rm -rf {cache_dir})')
                        try:
                            base_kwargs_retry = base_kwargs.copy()
                            base_kwargs_retry['download_mode'] = 'force_redownload'
                            ds = load_dataset(dataset_name_to_use, split=split, **base_kwargs_retry)
                            print('Successfully loaded local data!')
                        except Exception as e2:
                            raise RuntimeError(f'Failed to load the local dataset. The cache might be corrupted.\nPlease manually clear the cache (no root privileges required):\n  rm -rf {cache_dir}\nThen run again.\nLocal data path: {local_dir}\nOriginal error: {e}\nRetry error: {e2}') from e2
                else:
                    if os.path.exists(cache_dir):
                        print(f'Detected cache error, automatically cleaning cache directory: {cache_dir}')
                        try:
                            shutil.rmtree(cache_dir)
                            print('Cache cleaned, redownloading dataset...')
                        except Exception as cleanup_error:
                            print(f'Warning: Unable to automatically clean cache: {cleanup_error}')
                            print(f'Please manually execute: rm -rf {cache_dir}')
                    try:
                        ds = load_dataset(dataset_name_to_use, split=split, download_mode='force_redownload', **base_kwargs)
                        print('Dataset redownloaded successfully!')
                    except Exception as e2:
                        raise RuntimeError(f'Still unable to load dataset after cleaning cache.\\nOriginal error: {e}\\nRedownload error: {e2}\\nPlease check network connection or manually clean cache:\\n  rm -rf ~/.cache/huggingface/datasets/ceed/') from e2
            else:
                raise RuntimeError(f'Dataset loading error: {e}\\nPlease try cleaning cache: rm -rf ~/.cache/huggingface/datasets/ceed/') from e
        except (TypeError, ValueError) as e:
            errors.append(f'Method 1 failed: {type(e).__name__}: {e}')
            try:
                base_kwargs_with_trust = base_kwargs.copy()
                if not is_new_version:
                    base_kwargs_with_trust['trust_remote_code'] = True
                ds = load_dataset(dataset_name_to_use, split=split, **base_kwargs_with_trust)
            except (TypeError, ValueError) as e2:
                errors.append(f'Method 2 failed: {type(e2).__name__}: {e2}')
                try:
                    print('Attempting to force redownload dataset...')
                    base_kwargs_force = base_kwargs.copy()
                    base_kwargs_force['download_mode'] = 'force_redownload'
                    if not is_new_version:
                        base_kwargs_force['trust_remote_code'] = True
                    ds = load_dataset(dataset_name_to_use, split=split, **base_kwargs_force)
                except Exception as e3:
                    errors.append(f'Method 3 failed: {type(e3).__name__}: {e3}')
                    raise RuntimeError(f'Unable to load dataset, all attempts failed.\\nError details:\\n' + '\\n'.join((f'  - {err}' for err in errors)) + f"\\nSuggested solutions:\\n  1. Clean cache: rm -rf ~/.cache/huggingface/datasets/ceed/\\n  2. Check local data path: CEED_LOCAL_DIR = '{local_dir or 'None'}'\\n  3. If using local data, ensure the path is correct and contains .h5 files") from e3
            except NotImplementedError as e2:
                if 'LocalFileSystem' in str(e2):
                    import os
                    import shutil
                    cache_dir = os.path.expanduser('~/.cache/huggingface/datasets/ceed/')
                    if os.path.exists(cache_dir):
                        print(f'Detected cache error, automatically cleaning cache directory: {cache_dir}')
                        try:
                            shutil.rmtree(cache_dir)
                            print('Cache cleaned, redownloading dataset...')
                        except Exception as cleanup_error:
                            print(f'Warning: Unable to automatically clean cache: {cleanup_error}')
                    try:
                        base_kwargs_retry = base_kwargs.copy()
                        base_kwargs_retry['download_mode'] = 'force_redownload'
                        if not is_new_version:
                            base_kwargs_retry['trust_remote_code'] = True
                        ds = load_dataset(dataset_name_to_use, split=split, **base_kwargs_retry)
                        print('Dataset redownloaded successfully!')
                    except Exception as e3:
                        raise RuntimeError(f'Still unable to load dataset after cleaning cache.\\nPlease manually clean cache: rm -rf ~/.cache/huggingface/datasets/ceed/\\nOriginal error: {e2}') from e3
                else:
                    raise RuntimeError(f'Dataset loading error: {e2}') from e2
        if ds is None:
            raise RuntimeError(f'Unable to load dataset.\\nError details:\\n' + '\\n'.join((f'  - {err}' for err in errors)))
        if limit is not None and len(ds) > limit:
            ds = ds.select(range(limit))
        self.ds = ds
        self.use_waveform_augmentation = bool(use_waveform_augmentation)
        self.aug_noise_snr_db_min = float(aug_noise_snr_db_min)
        self.aug_noise_snr_db_max = float(aug_noise_snr_db_max)
        self.aug_amplitude_min = float(aug_amplitude_min)
        self.aug_amplitude_max = float(aug_amplitude_max)
        if label_sigma_sec is not None:
            sigma_samples = float(label_sigma_sec) * self.sampling_rate
            self.gauss = make_gaussian_window_sigma(sigma_samples=sigma_samples, cover=3.0)
        else:
            self.gauss = make_gaussian_window(label_width)

    def __len__(self):
        return len(self.ds)

    def _get_waveform(self, rec: dict) -> np.ndarray:
        if self.waveform_key is not None and self.waveform_key in rec:
            arr = np.asarray(rec[self.waveform_key])
        elif 'data' in rec:
            arr = np.asarray(rec['data'])
        else:
            arr = None
            for _, v in rec.items():
                try:
                    a = np.asarray(v)
                except Exception:
                    continue
                if a.ndim >= 2 and a.size > 0:
                    arr = a
                    break
            if arr is None:
                raise ValueError('No suitable waveform field found, please manually specify waveform_key')
        return self._to_tc_strict(arr)

    @staticmethod
    def _to_tc_strict(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 3:
            s0, s1, s2 = a.shape
            if s1 == 3:
                chan_dim = 1
            elif s2 == 3:
                chan_dim = 2
            elif s0 == 3:
                chan_dim = 0
            else:
                chan_dim = int(np.argmin([s0, s1, s2]))
            dims = [s0, s1, s2]
            time_candidates = [(i, l) for i, l in enumerate(dims) if i != chan_dim]
            time_dim = max(time_candidates, key=lambda x: x[1])[0]
            sample_dim = [0, 1, 2]
            sample_dim.remove(chan_dim)
            sample_dim.remove(time_dim)
            sample_dim = sample_dim[0]
            a = np.take(a, indices=0, axis=sample_dim)
            if a.shape[0] == 3 and a.shape[1] != 3:
                a = a.T
            elif a.shape[1] == 3 and a.shape[0] != 3:
                pass
            elif a.shape[0] >= a.shape[1]:
                pass
            else:
                a = a.T
        elif a.ndim == 2:
            if a.shape[0] == 3 and a.shape[1] != 3:
                a = a.T
            elif a.shape[1] == 3 and a.shape[0] != 3:
                pass
            elif a.shape[0] >= a.shape[1]:
                pass
            else:
                a = a.T
        else:
            raise ValueError(f'Unsupported waveform dimension: {a.shape}')
        return a.astype(np.float32)

    def _to_text(self, x) -> str:
        if isinstance(x, (bytes, np.bytes_)):
            try:
                return x.decode('utf-8', errors='ignore')
            except Exception:
                return str(x)
        return str(x)

    def _get_pick_index(self, rec: dict, key: str) -> int | None:
        if key not in rec:
            return None
        v = rec[key]
        try:
            if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                first_element = v[0]
                while isinstance(first_element, (list, np.ndarray)) and len(first_element) > 0:
                    first_element = first_element[0]
                return int(float(first_element))
            else:
                return int(float(v))
        except Exception:
            return None

    def _extract_phase_picks_from_lists(self, rec: dict) -> tuple[int | None, int | None]:
        itp = None
        its = None
        if 'phase_index' in rec and 'phase_type' in rec:
            phase_indices = rec['phase_index']
            phase_types = rec['phase_type']
            if isinstance(phase_indices, (list, np.ndarray)):
                processed_indices = []
                for idx in phase_indices:
                    if isinstance(idx, (list, np.ndarray)) and len(idx) > 0:
                        processed_indices.append(idx[0])
                    else:
                        processed_indices.append(idx)
                phase_indices = processed_indices
            if isinstance(phase_types, (list, np.ndarray)):
                processed_types = []
                for phase_type in phase_types:
                    if isinstance(phase_type, (list, np.ndarray)) and len(phase_type) > 0:
                        processed_types.append(phase_type[0])
                    else:
                        processed_types.append(phase_type)
                phase_types = processed_types
            for idx, phase_type in zip(phase_indices, phase_types):
                try:
                    idx_val = int(float(idx))
                    phase_str = self._to_text(phase_type).upper().strip()
                    if phase_str == 'P' and itp is None:
                        itp = idx_val
                    elif phase_str == 'S' and its is None:
                        its = idx_val
                    if itp is not None and its is not None:
                        break
                except Exception:
                    continue
        return (itp, its)

    def _extract_phase_picks_from_lists_robust(self, rec: dict) -> tuple[int | None, int | None]:
        itp = None
        its = None
        if 'phase_index' not in rec or 'phase_type' not in rec:
            return (itp, its)
        pidx = rec['phase_index']
        ptyp = rec['phase_type']

        def _ensure_2d(obj):
            if isinstance(obj, (list, np.ndarray)):
                if len(obj) > 0 and isinstance(obj[0], (list, np.ndarray)):
                    return obj
                else:
                    return [obj]
            else:
                return [[obj]]
        pidx_2d = _ensure_2d(pidx)
        ptyp_2d = _ensure_2d(ptyp)
        for idx_list, typ_list in zip(pidx_2d, ptyp_2d):
            m = min(len(idx_list), len(typ_list))
            for k in range(m):
                try:
                    idx_val = int(float(idx_list[k]))
                except Exception:
                    continue
                t = self._to_text(typ_list[k]).upper().strip()
                if itp is None and t == 'P':
                    itp = idx_val
                elif its is None and t == 'S':
                    its = idx_val
                if itp is not None and its is not None:
                    return (itp, its)
        return (itp, its)

    def _get_phase_picks(self, rec: dict) -> tuple[int | None, int | None]:
        itp = self._get_pick_index(rec, self.p_key) if self.p_key is not None else None
        its = self._get_pick_index(rec, self.s_key) if self.s_key is not None else None
        if self.p_key is not None and itp is None or (self.s_key is not None and its is None):
            return (itp, its)
        itp2, its2 = self._extract_phase_picks_from_lists_robust(rec)
        return (itp if itp is not None else itp2, its if its is not None else its2)

    def __getitem__(self, i: int):
        try:
            rec = self.ds[int(i)]
        except TypeError as e:
            if 'maps_as_pydicts' in str(e) or 'unexpected keyword argument' in str(e):
                if not hasattr(self, '_pyarrow_warning_shown'):
                    import warnings
                    warnings.warn(f"Detected pyarrow version incompatibility error: {e}\\nRecommend installing a compatible pyarrow version:\\n  pip install 'pyarrow>=8.0.0,<15.0.0'\\nOr upgrade the datasets library:\\n  pip install --upgrade datasets pyarrow", UserWarning)
                    self._pyarrow_warning_shown = True
                try:
                    if not hasattr(self, '_ds_pandas'):
                        self._ds_pandas = self.ds.to_pandas()
                    rec = self._ds_pandas.iloc[int(i)].to_dict()
                except Exception as e2:
                    try:
                        if not hasattr(self, '_ds_dict'):
                            self._ds_dict = self.ds.with_format('python')
                        rec = self._ds_dict[int(i)]
                    except Exception as e3:
                        if not hasattr(self, '_ds_list'):
                            print('Warning: Attempting to convert dataset to list format (this may take some time and use a lot of memory)...')
                            self._ds_list = list(self.ds)
                        rec = self._ds_list[int(i)]
            else:
                raise
        x = self._get_waveform(rec)
        T, C = x.shape
        if C != 3:
            if C < 3:
                pad = np.zeros((T, 3 - C), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=1)
            else:
                x = x[:, :3]
        itp, its = self._get_phase_picks(rec)
        picks = [p for p in [itp, its] if p is not None and p >= 0]
        if len(picks) == 2:
            center = int(round(0.5 * (picks[0] + picks[1])))
        elif len(picks) == 1:
            center = int(picks[0])
        else:
            center = T // 2
        if self.training:
            import random as _random
            center = max(0, min(T - 1, center + _random.randint(-200, 200)))
        start = max(0, min(center - self.crop_len // 2, T - self.crop_len))
        end = start + self.crop_len
        x_win = x[start:end, :].astype(np.float32)
        if self.training and self.use_waveform_augmentation:
            scale = np.random.uniform(self.aug_amplitude_min, self.aug_amplitude_max)
            x_win = x_win * scale
            snr_db = np.random.uniform(self.aug_noise_snr_db_min, self.aug_noise_snr_db_max)
            signal_power = np.var(x_win) + 1e-12
            noise_power = signal_power / 10.0 ** (snr_db / 10.0)
            noise_std = np.sqrt(noise_power)
            noise = np.random.normal(0, noise_std, size=x_win.shape).astype(np.float32)
            x_win = x_win + noise
        y = np.zeros((self.crop_len, 3), dtype=np.float32)
        if itp is not None and itp >= 0:
            p_idx = itp - start
            if 0 <= p_idx < self.crop_len:
                self._fill_label(y, int(p_idx), 1)
        if its is not None and its >= 0:
            s_idx = its - start
            if 0 <= s_idx < self.crop_len:
                self._fill_label(y, int(s_idx), 2)
        y[:, 0] = 1.0 - (y[:, 1] + y[:, 2])
        y = np.clip(y, 0.0, 1.0)
        x_win = normalize_zero_mean_unit_std(x_win, axis=0)
        x_t = torch.from_numpy(x_win.T.copy())
        y_t = torch.from_numpy(y.T.copy())
        return (x_t, y_t, str(i))

    def _fill_label(self, target: np.ndarray, idx: int, cls: int):
        T = target.shape[0]
        half = len(self.gauss) // 2
        s = idx - half
        e = idx + half + 1
        if e <= 0 or s >= T:
            return
        ws = max(0, -s)
        we = min(len(self.gauss), len(self.gauss) - (e - T))
        s = max(0, s)
        e = min(T, e)
        target[s:e, cls] = np.maximum(target[s:e, cls], self.gauss[ws:we])
