from __future__ import annotations
import os
import random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    import h5py
except ImportError:
    h5py = None
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import normalize_zero_mean_unit_std, make_gaussian_window_sigma
_SR_ATTR_KEYS = ('sampling_rate', 'sample_rate', 'fs', 'sr')
_DT_ATTR_KEYS = ('dt', 'delta', 'time_step', 'sample_interval')

def _infer_sampling_rate_from_h5_attrs(h5) -> float | None:

    def _as_float(v):
        try:
            if hasattr(v, 'item'):
                return float(v.item())
            a = np.asarray(v).flatten()
            return float(a[0]) if a.size else None
        except Exception:
            return None
    attrs_to_try = [getattr(h5, 'attrs', None)]
    try:
        if 'waveforms' in h5:
            attrs_to_try.append(h5['waveforms'].attrs)
    except Exception:
        pass
    try:
        if 'waveforms/channel_ud' in h5:
            attrs_to_try.append(h5['waveforms/channel_ud'].attrs)
    except Exception:
        pass
    for attrs in attrs_to_try:
        if not attrs:
            continue
        for k in _SR_ATTR_KEYS:
            if k in attrs:
                v = _as_float(attrs.get(k))
                if v is not None and np.isfinite(v) and (v > 0):
                    return float(v)
        for k in _DT_ATTR_KEYS:
            if k in attrs:
                v = _as_float(attrs.get(k))
                if v is not None and np.isfinite(v) and (v > 0):
                    return float(1.0 / v)
    return None

def _read_scalar(dset) -> float:
    v = dset[()]
    if hasattr(v, 'item'):
        return float(v)
    v = np.asarray(v).flatten()
    return float(v[0]) if v.size else np.nan

def _read_pg_sg_at(h5, seg_idx: int | None, sr: float, arrival_relative: bool) -> tuple[int, int]:
    pg_ds = h5['arrival_times/pg']
    sg_ds = h5['arrival_times/sg']
    pg_arr = np.asarray(pg_ds).flatten()
    sg_arr = np.asarray(sg_ds).flatten()
    if seg_idx is not None and pg_arr.size > 1 and (seg_idx < pg_arr.size):
        pg_sec = float(pg_arr[seg_idx])
        sg_sec = float(sg_arr[seg_idx]) if seg_idx < sg_arr.size else float(sg_arr[0])
    else:
        pg_sec = float(pg_arr[0]) if pg_arr.size else np.nan
        sg_sec = float(sg_arr[0]) if sg_arr.size else np.nan
    if arrival_relative:
        p_idx = int(round(pg_sec * sr))
        s_idx = int(round(sg_sec * sr))
    else:
        p_idx = int(round(10.0 * sr))
        s_idx = int(round((sg_sec - pg_sec + 10.0) * sr))
    return (p_idx, s_idx)

class ThreeChannelH5Dataset(Dataset):

    def __init__(self, root_dir: str, crop_len: int=3000, sampling_rate: float=100.0, label_sigma_sec: float=0.1, label_width: int=51, training: bool=True, file_pattern: str='*.h5', arrival_relative_to_segment: bool=True, filter_natural_only: bool=False, allow_earthquake_types: tuple | None=None, strict_check: bool=False, strict_sr_tol_hz: float=0.001, limit: int | None=None):
        if h5py is None:
            raise ImportError('请安装 h5py: pip install h5py')
        self.root_dir = os.path.abspath(root_dir)
        self.crop_len = int(crop_len)
        self.sampling_rate = float(sampling_rate)
        self.training = bool(training)
        self.arrival_relative_to_segment = bool(arrival_relative_to_segment)
        self.filter_natural_only = bool(filter_natural_only)
        self.allow_earthquake_types = allow_earthquake_types or ()
        self.strict_check = bool(strict_check)
        self.strict_sr_tol_hz = float(strict_sr_tol_hz)
        sigma_samples = label_sigma_sec * sampling_rate
        self.gauss = make_gaussian_window_sigma(sigma_samples=sigma_samples, cover=3.0)
        files = sorted(glob.glob(os.path.join(self.root_dir, file_pattern)))
        self._samples: list[tuple[str, int]] = []
        for f in files:
            if limit is not None and len(self._samples) >= limit:
                break
            if filter_natural_only or self.allow_earthquake_types:
                try:
                    with h5py.File(f, 'r') as h5:
                        if filter_natural_only:
                            if 'labels/natural_earthquake' not in h5:
                                continue
                            nat = _read_scalar(h5['labels/natural_earthquake'])
                            if int(nat) != 1:
                                continue
                        if self.allow_earthquake_types:
                            if 'labels/earthquake_type' not in h5:
                                continue
                            et = h5['labels/earthquake_type'][()]
                            if hasattr(et, 'decode'):
                                et = et.decode('utf-8') if isinstance(et, bytes) else str(et)
                            else:
                                et = str(np.asarray(et).flatten()[0])
                            if et not in self.allow_earthquake_types:
                                continue
                except Exception:
                    continue
            try:
                with h5py.File(f, 'r') as h5:
                    d = h5['waveforms/channel_ud']
                    shp = d.shape
                    ndim = d.ndim
                    if ndim == 1:
                        n_seg = 1
                    elif ndim == 2:
                        if shp[0] == 1 or shp[1] == 1:
                            n_seg = 1
                        else:
                            n_seg = int(shp[0])
                    else:
                        n_seg = 1
                    for seg_idx in range(n_seg):
                        self._samples.append((f, seg_idx))
                        if limit is not None and len(self._samples) >= limit:
                            break
            except Exception:
                self._samples.append((f, 0))
            if limit is not None and len(self._samples) >= limit:
                break

    def __len__(self) -> int:
        return len(self._samples)

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

    def __getitem__(self, i: int):
        path, seg_idx = self._samples[i]
        base_name = os.path.splitext(os.path.basename(path))[0]
        name = f'{base_name}_{seg_idx}' if seg_idx > 0 else base_name
        with h5py.File(path, 'r') as h5:
            if self.strict_check:
                sr_in_file = _infer_sampling_rate_from_h5_attrs(h5)
                if sr_in_file is not None:
                    if abs(float(sr_in_file) - float(self.sampling_rate)) > self.strict_sr_tol_hz:
                        raise ValueError(f"[ThreeChannelH5Dataset strict_check] 采样率不一致：file_sr={sr_in_file}Hz vs expected_sr={self.sampling_rate}Hz (tol={self.strict_sr_tol_hz}Hz), file='{path}'")
            ud_d = h5['waveforms/channel_ud']
            ns_d = h5['waveforms/channel_ns']
            ew_d = h5['waveforms/channel_ew']
            if ud_d.ndim == 2 and ud_d.shape[0] > 1 and (ud_d.shape[1] > 1):
                ud = np.asarray(ud_d[seg_idx, :], dtype=np.float32).flatten()
                ns = np.asarray(ns_d[seg_idx, :], dtype=np.float32).flatten()
                ew = np.asarray(ew_d[seg_idx, :], dtype=np.float32).flatten()
            else:
                ud = np.asarray(ud_d, dtype=np.float32).flatten()
                ns = np.asarray(ns_d, dtype=np.float32).flatten()
                ew = np.asarray(ew_d, dtype=np.float32).flatten()
            if ud.size <= 0 or ns.size <= 0 or ew.size <= 0:
                raise ValueError(f"[ThreeChannelH5Dataset] 空波形数据：file='{path}', seg_idx={seg_idx}")
            T = max(int(ud.size), int(ns.size), int(ew.size))
            if self.strict_check and (not ud.size == ns.size == ew.size):
                raise ValueError(f"[ThreeChannelH5Dataset strict_check] 三通道长度不一致：ud={ud.size}, ns={ns.size}, ew={ew.size}, file='{path}', seg_idx={seg_idx}")
            if ud.size != T:
                ud = np.resize(ud, T)
            if ns.size != T:
                ns = np.resize(ns, T)
            if ew.size != T:
                ew = np.resize(ew, T)
            x = np.stack([ud, ns, ew], axis=1)
            sr = self.sampling_rate
            p_idx_raw, s_idx_raw = _read_pg_sg_at(h5, seg_idx, sr, self.arrival_relative_to_segment)
            if self.strict_check:
                if not (np.isfinite(p_idx_raw) and np.isfinite(s_idx_raw)):
                    raise ValueError(f"[ThreeChannelH5Dataset strict_check] pg/sg 非法（NaN/Inf）：p_idx={p_idx_raw}, s_idx={s_idx_raw}, file='{path}', seg_idx={seg_idx}")
                if not 0 <= int(p_idx_raw) < T:
                    raise ValueError(f"[ThreeChannelH5Dataset strict_check] P 到时越界：p_idx={p_idx_raw}, T={T}, file='{path}', seg_idx={seg_idx}. 提示：采样率/到时单位/arrival_relative_to_segment 可能不匹配。")
                if not 0 <= int(s_idx_raw) < T:
                    raise ValueError(f"[ThreeChannelH5Dataset strict_check] S 到时越界：s_idx={s_idx_raw}, T={T}, file='{path}', seg_idx={seg_idx}. 提示：采样率/到时单位/arrival_relative_to_segment 可能不匹配。")
            p_idx = max(0, min(T - 1, int(p_idx_raw)))
            s_idx = max(0, min(T - 1, int(s_idx_raw)))
        if T >= self.crop_len:
            if self.training:
                center = p_idx + random.randint(-max(1, self.crop_len // 10), max(1, self.crop_len // 10))
                center = max(self.crop_len // 2, min(T - 1 - self.crop_len // 2, center))
            else:
                center = p_idx
            start = max(0, min(center - self.crop_len // 2, T - self.crop_len))
            end = start + self.crop_len
            x_win = x[start:end, :]
            p_in_win = p_idx - start
            s_in_win = s_idx - start
        else:
            pad_left = (self.crop_len - T) // 2
            pad_right = self.crop_len - T - pad_left
            x_win = np.pad(x, ((pad_left, pad_right), (0, 0)), mode='constant', constant_values=0.0)
            p_in_win = p_idx + pad_left
            s_in_win = s_idx + pad_left
        if self.strict_check:
            if x_win.shape[0] != self.crop_len or x_win.shape[1] != 3:
                raise ValueError(f"[ThreeChannelH5Dataset strict_check] 裁剪/补零后形状异常：x_win.shape={x_win.shape}, expected=({self.crop_len},3), file='{path}', seg_idx={seg_idx}")
            if not 0 <= int(p_in_win) < self.crop_len:
                raise ValueError(f"[ThreeChannelH5Dataset strict_check] P 在窗口内越界：p_in_win={p_in_win}, crop_len={self.crop_len}, file='{path}', seg_idx={seg_idx}")
            if not 0 <= int(s_in_win) < self.crop_len:
                raise ValueError(f"[ThreeChannelH5Dataset strict_check] S 在窗口内越界：s_in_win={s_in_win}, crop_len={self.crop_len}, file='{path}', seg_idx={seg_idx}")
        y = np.zeros((self.crop_len, 3), dtype=np.float32)
        if 0 <= p_in_win < self.crop_len:
            self._fill_label(y, int(p_in_win), 1)
        if 0 <= s_in_win < self.crop_len:
            self._fill_label(y, int(s_in_win), 2)
        y[:, 0] = 1.0 - (y[:, 1] + y[:, 2])
        y = np.clip(y, 0.0, 1.0)
        x_win = normalize_zero_mean_unit_std(x_win, axis=0)
        x_t = torch.from_numpy(x_win.T.copy())
        y_t = torch.from_numpy(y.T.copy())
        return (x_t, y_t, name)
