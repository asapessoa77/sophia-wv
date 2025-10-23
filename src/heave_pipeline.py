# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence, Callable, Dict
from scipy import signal
import warnings

def _validate_fs(fs: float) -> None:
    if fs <= 0:
        raise ValueError("Taxa de amostragem fs deve ser positiva (Hz).")

def _butter_highpass(hp: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    if hp is None or hp <= 0:
        return np.array([1.0]), np.array([1.0])
    if hp >= nyq:
        raise ValueError(f"Frequência de corte high-pass ({hp} Hz) deve ser < Nyquist ({nyq} Hz).")
    b, a = signal.butter(order, hp/nyq, btype="highpass")
    return b, a

def _to_numpy_1d(x) -> np.ndarray:
    arr = np.asarray(x).reshape(-1)
    if not np.isfinite(arr).all():
        warnings.warn("Existem valores não finitos; usando interpolação linear para preencher.")
        nans = ~np.isfinite(arr)
        if nans.all():
            raise ValueError("A série contém somente NaNs/inf.")
        idx = np.arange(arr.size)
        arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr

@dataclass
class PreprocessConfig:
    detrend: bool = True
    highpass_hz: float = 0.03
    hp_order: int = 4
    taper_frac: float = 0.05

def preprocess_heave(heave_m: Sequence[float], fs: float, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    _validate_fs(fs)
    h = _to_numpy_1d(heave_m)
    if cfg.detrend:
        h = signal.detrend(h, type="linear")
    if cfg.highpass_hz and cfg.highpass_hz > 0:
        b, a = _butter_highpass(cfg.highpass_hz, fs, order=cfg.hp_order)
        h = signal.filtfilt(b, a, h, method="gust")
    if cfg.taper_frac and cfg.taper_frac > 0:
        n = len(h); m = int(np.floor(cfg.taper_frac * n))
        if m > 0:
            window = np.ones(n)
            ramp = (1 - np.cos(np.linspace(0, np.pi, m))) / 2.0
            window[:m] = ramp; window[-m:] = ramp[::-1]
            h = h * window
    return h

def _interp_rao(freq_hz: np.ndarray, rao_freq: np.ndarray, rao_vals: np.ndarray) -> np.ndarray:
    rao_interp = np.interp(freq_hz, rao_freq, rao_vals, left=rao_vals[0], right=rao_vals[-1])
    eps = 1e-12
    rao_interp = np.where(np.abs(rao_interp) < eps, eps, rao_interp)
    return rao_interp

def estimate_eta_from_heave(heave_m: Sequence[float], fs: float,
                            rao: Optional[Tuple[np.ndarray, np.ndarray] | Callable[[np.ndarray], np.ndarray]] = None):
    _validate_fs(fs)
    h = _to_numpy_1d(heave_m)
    n = len(h)
    if rao is None:
        return h.copy(), np.ones(1)
    H = np.fft.rfft(h)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    if callable(rao):
        R = rao(freqs)
    else:
        f_rao, rao_vals = rao
        R = _interp_rao(freqs, np.asarray(f_rao).reshape(-1), np.asarray(rao_vals).reshape(-1))
    eps = 1e-12
    R = np.where(np.abs(R) < eps, eps, R)
    E = H / R
    eta = np.fft.irfft(E, n=n).real
    return eta, R

@dataclass
class WelchConfig:
    nperseg: Optional[int] = None
    noverlap: Optional[int] = None
    window: str = "hann"
    detrend: str | None = None
    scaling: str = "density"

def _auto_nperseg(fs: float, target_minutes: float = 20.0, min_nperseg: int = 256) -> int:
    nps = int(round(target_minutes * 60 * fs))
    return max(min_nperseg, nps)

# def psd_welch(eta_m: Sequence[float], fs: float, cfg: WelchConfig = WelchConfig()):
#     _validate_fs(fs)
#     x = _to_numpy_1d(eta_m)
#     nperseg = cfg.nperseg or _auto_nperseg(fs)
#     noverlap = cfg.noverlap if cfg.noverlap is not None else nperseg // 2
#     f, Pxx = signal.welch(x, fs=fs, window=cfg.window, nperseg=nperseg,
#                           noverlap=noverlap, detrend=cfg.detrend, scaling=cfg.scaling,
#                           return_onesided=True)
#     return f, Pxx

def psd_welch(eta_m, fs, cfg=WelchConfig()):
    _validate_fs(fs)
    x = _to_numpy_1d(eta_m)
    n = x.shape[-1]

    # 1) nperseg sugerido (≈20 min), mas nunca maior que n
    nperseg = cfg.nperseg or _auto_nperseg(fs)
    nperseg = int(min(nperseg, n))
    if nperseg < 8:
        raise ValueError(
            f"nperseg={nperseg} muito pequeno para estimar PSD. "
            f"Aumente a duração do registro ou a taxa fs."
        )

    # 2) noverlap seguro (< nperseg)
    if cfg.noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(cfg.noverlap)
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)  # garante estritamente menor

    f, Pxx = signal.welch(
        x, fs=fs, window=cfg.window, nperseg=nperseg,
        noverlap=noverlap, detrend=cfg.detrend, scaling=cfg.scaling,
        return_onesided=True
    )
    return f, Pxx


def spectral_moments(f_hz: np.ndarray, S: np.ndarray, orders=(0,1,2)):
    f = np.asarray(f_hz); S = np.asarray(S)
    if f.ndim != 1 or S.ndim != 1 or f.size != S.size:
        raise ValueError("f e S devem ser vetores 1D do mesmo tamanho.")
    out = {}
    for n in orders:
        out[n] = np.trapz((f**n) * S, f)
    return out

def wave_parameters_from_psd(f_hz: np.ndarray, S: np.ndarray):
    m = spectral_moments(f_hz, S, orders=(0,1,2))
    m0, m1, m2 = m[0], m[1], m[2]
    Hs = 4.0 * np.sqrt(m0)
    idxp = int(np.nanargmax(S)) if S.size > 0 else 0
    fp = f_hz[idxp] if S.size > 0 else np.nan
    Tp = 1.0/fp if fp > 0 else np.nan
    Tm01 = m0/m1 if m1 > 0 else np.nan
    Tm02 = np.sqrt(m0/m2) if m2 > 0 else np.nan
    return {"Hs": Hs, "Tp": Tp, "Tm01": Tm01, "Tm02": Tm02, "fp": fp, "m0": m0, "m1": m1, "m2": m2}

@dataclass
class HeavePipelineConfig:
    fs_hz: float
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    welch: WelchConfig = field(default_factory=WelchConfig)
    rao: Optional[Tuple[np.ndarray, np.ndarray] | Callable[[np.ndarray], np.ndarray]] = None
    fmin_hz: float = 0.03
    fmax_hz: Optional[float] = None

class HeaveWavePipeline:
    def __init__(self, config: HeavePipelineConfig):
        self.cfg = config
        _validate_fs(self.cfg.fs_hz)

    def run(self, heave_m: Sequence[float]):
        fs = self.cfg.fs_hz
        hpp = preprocess_heave(heave_m, fs, cfg=self.cfg.preprocess)
        eta, rao_used = estimate_eta_from_heave(hpp, fs, rao=self.cfg.rao)
        f, S = psd_welch(eta, fs, cfg=self.cfg.welch)
        fmin = self.cfg.fmin_hz if self.cfg.fmin_hz else 0.0
        fmax = self.cfg.fmax_hz if self.cfg.fmax_hz else fs/2.0
        mask = (f >= fmin) & (f <= fmax)
        f_b, S_b = f[mask], S[mask]
        params = wave_parameters_from_psd(f_b, S_b)
        return {"eta": eta, "heave_pp": hpp, "f_hz": f_b, "S_etaeta": S_b, "params": params, "rao_used": rao_used}
