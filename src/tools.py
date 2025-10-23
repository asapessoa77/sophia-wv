from pathlib import Path
import re
import pandas as pd
from datetime import datetime, timedelta

def datetimes_from_filename(path: str, N: int, dt_seconds: float,
                            tz: str | None = "America/Sao_Paulo") -> pd.DatetimeIndex:
    """
    Lê data/hora do nome do arquivo no formato: YYYY-MM-DDTHHhMM
    Ex.: '2021-06-17T00h00.raw' → 2021-06-17 00:00

    path       : caminho do arquivo (string ou Path-like)
    N          : número de amostras
    dt_seconds : passo entre amostras (em segundos, pode ser float)
    tz         : fuso horário (ex.: 'America/Sao_Paulo'); use None para timezone-naive
    """
    fname = Path(path).name
    m = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})h(\d{2})", fname)
    if not m:
        raise ValueError(
            f"Não encontrei padrão 'YYYY-MM-DDTHHhMM' em: {fname}"
        )
    date_str, hh, mm = m.groups()
    t0 = pd.Timestamp(f"{date_str} {hh}:{mm}", tz=tz)

    freq = pd.to_timedelta(dt_seconds, unit="s")
    return pd.date_range(start=t0, periods=N, freq=freq)

def create_syntetic_time(df):
    #Criar datas sinteticas
    start_time = datetime(1950, 1, 1, 0, 0, 0)

    # Criar a coluna 'ds' com datas a partir do tempo em segundos    
    df['ds'] = pd.date_range(start=start_time, periods=len(df), freq='D')

    df['ds'] = pd.to_datetime(df['ds'])
    df = df[['ds','ds_ori','y']]

    return df


def read_files(arqs, dt):
    dfs = []
    for a in arqs:                
        dfs.append(pd.read_csv(a, header=None))
    
    df = pd.concat(dfs, ignore_index=True)
    df = df[[1]]
    df.rename(columns={1:'y'}, inplace=True)

    df['ds_ori'] = datetimes_from_filename(arqs[0], len(df), dt_seconds=dt, tz=None)
    df = df[['ds_ori', 'y']]

    df = create_syntetic_time(df)

    df['y'] = df['y'] / 100

    #eixo do tempo original
    t_ori = df['ds_ori']

    return df, t_ori

def split_train_test(df, cut):
    return df[['ds','y']][(df.ds_ori <= cut)].copy(), df[['ds','y']][(df.ds_ori > cut)].copy()

from typing import Iterable, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_power_spectrum_pair(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    freq_col1: Optional[str] = None,
    freq_col2: Optional[str] = None,
    col1: Optional[Union[str, Iterable[str]]] = None,
    col2: Optional[Union[str, Iterable[str]]] = None,
    logx: bool = False,
    logy: bool = True,
    xlim: Optional[Iterable[float]] = None,
    ylim: Optional[Iterable[float]] = None,
    title: Optional[str] = None,
    units: str = "Hz",
    density_label: str = "PSD",
    grid: bool = True,    
    peak_min_freq: Optional[float] = None,
    savepath: Optional[str] = None,
    figsize=(8,5),
    show: bool = True,
    return_fig: bool = False,
    fill: bool = True,
    fill_alpha1: float = 0.25,
    fill_alpha2: float = 0.25,
    label1: str = "Curva 1",
    label2: str = "Curva 2",
):
    """
    Plot a power spectrum / PSD stored in a DataFrame, com opção de preencher a área sob a curva.

    Parâmetros novos
    ----------------
    fill : bool
        Se True, preenche a área sob cada série do espectro.
    fill_alpha : float
        Opacidade do preenchimento (0 a 1).
    """
    # 1) Frequência
    if freq_col1 is not None:
        if freq_col1 not in df1.columns:
            raise ValueError(f"Coluna de frequência '{freq_col1}' não encontrada no DataFrame.")
        f1 = df1[freq_col1].to_numpy()
        data_df1 = df1.drop(columns=[freq_col1])
    else:
        if df1.index.dtype.kind in "if" or (df1.index.name and df1.index.name.lower() in {"f", "freq", "frequency"}):
            f1 = np.asarray(df1.index, dtype=float)
            data_df1 = df1.copy()
        else:
            for cand in ("f", "freq", "frequency"):
                if cand in df1.columns:
                    f1 = df1[cand].to_numpy()
                    data_df1 = df1.drop(columns=[cand])
                    break
            else:
                raise ValueError("Não foi possível identificar a coluna/índice de frequência. "
                                 "Passe `freq_col='f'` (ou nome correto) ou coloque a frequência no índice.")
    
    if freq_col2 is not None:
        if freq_col2 not in df1.columns:
            raise ValueError(f"Coluna de frequência '{freq_col2}' não encontrada no DataFrame.")
        f2 = df2[freq_col2].to_numpy()
        data_df2 = df2.drop(columns=[freq_col2])
    else:
        if df2.index.dtype.kind in "if" or (df2.index.name and df2.index.name.lower() in {"f", "freq", "frequency"}):
            f2 = np.asarray(df2.index, dtype=float)
            data_df2 = df2.copy()
        else:
            for cand in ("f", "freq", "frequency"):
                if cand in df2.columns:
                    f2 = df2[cand].to_numpy()
                    data_df2 = df2.drop(columns=[cand])
                    break
            else:
                raise ValueError("Não foi possível identificar a coluna/índice de frequência. "
                                 "Passe `freq_col='f'` (ou nome correto) ou coloque a frequência no índice.")
       
    
    # 3) Figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # 4) Traçados + preenchimento
    
    y1 = data_df1[col1].to_numpy()
    y2 = data_df2[col2].to_numpy()
    ax.plot(f1, y1, label=label1)
    ax.plot(f2, y2, label=label2)
    if fill:
        # Preencher abaixo da curva até y=0
        ax.fill_between(f1, y1, 0.0, alpha=fill_alpha1)
        ax.fill_between(f2, y2, 0.0, alpha=fill_alpha2)
    
    # 5) Eixos
    if logx:
        ax.set_xscale("log")
    else:
        ax.set_xlim(0.0, None)        
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0.0, None)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # 6) Rótulos
    ax.set_xlabel(f"f [{units}]")
    ax.set_ylabel(density_label)
    if title:
        ax.set_title(title)
    
    # 7) Grid e legenda
    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    ax.legend()    
    
    
    # 9) Salvar
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    
    # 10) Mostrar/Retornar
    if show:
        plt.show()
        return None
    else:
        if return_fig:
            return fig
        return None

from typing import Iterable, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_power_spectrum(
    df: pd.DataFrame,
    freq_col: Optional[str] = None,
    cols: Optional[Union[str, Iterable[str]]] = None,
    logx: bool = False,
    logy: bool = True,
    xlim: Optional[Iterable[float]] = None,
    ylim: Optional[Iterable[float]] = None,
    title: Optional[str] = None,
    units: str = "Hz",
    density_label: str = "PSD",
    grid: bool = True,
    annotate_peaks: bool = False,
    n_peaks: int = 3,
    peak_min_freq: Optional[float] = None,
    savepath: Optional[str] = None,
    figsize=(8,5),
    show: bool = True,
    return_fig: bool = False,
    fill: bool = True,
    fill_alpha: float = 0.25,
):
    """
    Plot a power spectrum / PSD stored in a DataFrame, com opção de preencher a área sob a curva.

    Parâmetros novos
    ----------------
    fill : bool
        Se True, preenche a área sob cada série do espectro.
    fill_alpha : float
        Opacidade do preenchimento (0 a 1).
    """
    # 1) Frequência
    if freq_col is not None:
        if freq_col not in df.columns:
            raise ValueError(f"Coluna de frequência '{freq_col}' não encontrada no DataFrame.")
        f = df[freq_col].to_numpy()
        data_df = df.drop(columns=[freq_col])
    else:
        if df.index.dtype.kind in "if" or (df.index.name and df.index.name.lower() in {"f", "freq", "frequency"}):
            f = np.asarray(df.index, dtype=float)
            data_df = df.copy()
        else:
            for cand in ("f", "freq", "frequency"):
                if cand in df.columns:
                    f = df[cand].to_numpy()
                    data_df = df.drop(columns=[cand])
                    break
            else:
                raise ValueError("Não foi possível identificar a coluna/índice de frequência. "
                                 "Passe `freq_col='f'` (ou nome correto) ou coloque a frequência no índice.")
    
    # 2) Colunas
    if cols is None:
        data_cols = [c for c in data_df.columns if np.issubdtype(data_df[c].dtype, np.number)]
    else:
        if isinstance(cols, str):
            data_cols = [cols]
        else:
            data_cols = list(cols)
    if len(data_cols) == 0:
        raise ValueError("Nenhuma coluna numérica para plotar foi encontrada.")
    
    # 3) Figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # 4) Traçados + preenchimento
    for c in data_cols:
        y = data_df[c].to_numpy()
        ax.plot(f, y, label=c)
        if fill:
            # Preencher abaixo da curva até y=0
            ax.fill_between(f, y, 0.0, alpha=fill_alpha)
    
    # 5) Eixos
    if logx:
        ax.set_xscale("log")
    else:
        ax.set_xlim(0.0, None)        
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0.0, None)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # 6) Rótulos
    ax.set_xlabel(f"f [{units}]")
    ax.set_ylabel(density_label)
    if title:
        ax.set_title(title)
    
    # 7) Grid e legenda
    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if len(data_cols) > 1:
        ax.legend()
    
    # 8) Anotar picos
    if annotate_peaks:
        mask = np.ones_like(f, dtype=bool)
        if peak_min_freq is not None:
            mask &= f >= peak_min_freq
        
        for c in data_cols:
            y = data_df[c].to_numpy()
            f_m = f[mask]
            y_m = y[mask]
            if len(y_m) == 0:
                continue
            valid = np.isfinite(y_m)
            f_v = f_m[valid]
            y_v = y_m[valid]
            if len(y_v) == 0:
                continue
            top_idx = np.argsort(y_v)[-n_peaks:][::-1]
            for i in top_idx:
                ax.annotate(
                    f"{c}\n f={f_v[i]:.4g}\n y={y_v[i]:.3g}",
                    xy=(f_v[i], y_v[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", lw=0.5)
                )
    
    # 9) Salvar
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    
    # 10) Mostrar/Retornar
    if show:
        plt.show()
        return None
    else:
        if return_fig:
            return fig
        return None
