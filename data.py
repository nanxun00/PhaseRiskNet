from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def normalize_zero_mean_unit_std(x: np.ndarray, axis=0, eps: float=1e-08) -> np.ndarray:
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std[std < eps] = 1.0
    return (x - mean) / std

def make_gaussian_window(width: int=30) -> np.ndarray:
    half = width // 2
    t = np.arange(-half, half + 1)
    sigma = width / 5.0
    w = np.exp(-t ** 2 / (2 * sigma * sigma))
    return w

def make_gaussian_window_sigma(sigma_samples: float, cover: float=3.0) -> np.ndarray:
    half = int(np.ceil(cover * float(sigma_samples)))
    t = np.arange(-half, half + 1, dtype=float)
    sigma = max(1e-06, float(sigma_samples))
    w = np.exp(-t ** 2 / (2.0 * sigma * sigma))
    return w.astype(np.float32)

class WaveformDataset(Dataset):

    def __init__(self, data_dir: str, csv_file: str, crop_len: int=3000, label_width: int=30, training: bool=True, sampling_rate: float=100.0, label_sigma_sec: float=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.crop_len = int(crop_len)
        self.label_width = int(label_width)
        self.training = bool(training)
        self.sampling_rate = float(sampling_rate)
        self.label_sigma_sec = float(label_sigma_sec)
        df = pd.read_csv(csv_file)
        if 'fname' not in df.columns:
            raise ValueError("CSV must contain a 'fname' column")
        self.files = df['fname'].astype(str).tolist()
        if self.label_sigma_sec is not None:
            sigma_samples = self.label_sigma_sec * self.sampling_rate
            self.gauss = make_gaussian_window_sigma(sigma_samples=sigma_samples, cover=3.0)
        else:
            self.gauss = make_gaussian_window(self.label_width)

    def __len__(self):
        return len(self.files)

    def _read_npz(self, path: str):
        npz = np.load(path)
        data = npz['data']
        if data.ndim == 3:
            data = data[:, 0, :]
        if data.ndim != 2:
            raise ValueError(f'Unexpected data shape {data.shape} in {path}')

        def _get_pick(npz_obj, keys):
            for k in keys:
                if k in npz_obj.files:
                    v = npz_obj[k]
                    if np.isscalar(v):
                        return int(v)
                    v = np.array(v)
                    if v.size == 0:
                        return None
                    try:
                        return int(np.asarray(v).flatten()[0])
                    except Exception:
                        return None
            return None
        itp = _get_pick(npz, ['itp', 'p_idx'])
        its = _get_pick(npz, ['its', 's_idx'])
        return (data.astype(np.float32), itp, its)

    def __getitem__(self, i: int):
        fname = self.files[i]
        path = os.path.join(self.data_dir, fname)
        x, itp, its = self._read_npz(path)
        T, C = x.shape
        if C != 3:
            if C < 3:
                pad = np.zeros((T, 3 - C), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=1)
            else:
                x = x[:, :3]
        pick_list = [p for p in [itp, its] if isinstance(p, (int, np.integer)) and p >= 0]
        if len(pick_list) == 2:
            center = int(round(0.5 * (pick_list[0] + pick_list[1])))
        elif len(pick_list) == 1:
            center = int(pick_list[0])
        else:
            center = T // 2
        if self.training:
            center = max(0, min(T - 1, center + random.randint(-200, 200)))
        start = max(0, min(center - self.crop_len // 2, T - self.crop_len))
        end = start + self.crop_len
        x_win = x[start:end, :]
        y = np.zeros((self.crop_len, 3), dtype=np.float32)
        if itp is not None:
            p_idx = itp - start
            if 0 <= p_idx < self.crop_len:
                self._fill_label(y, int(p_idx), 1)
        if its is not None:
            s_idx = its - start
            if 0 <= s_idx < self.crop_len:
                self._fill_label(y, int(s_idx), 2)
        y[:, 0] = 1.0 - (y[:, 1] + y[:, 2])
        y = np.clip(y, 0.0, 1.0)
        x_win = normalize_zero_mean_unit_std(x_win, axis=0)
        x_t = torch.from_numpy(x_win.T.copy())
        y_t = torch.from_numpy(y.T.copy())
        return (x_t, y_t, fname)

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
