import matplotlib.pyplot as plt
import numpy as np
import IPython
import ipywidgets
from .utils import *

def new_axes(n_rows=1, n_cols=1, *, figsize=None):
    """
    Return axes object.

    Args:
        n_rows (int): number of rows.
        n_cols (int): number of columns.
        figsize ([width, height]): size of display. If none is given then it is proportional to n_rows and n_cols.

    Returns:
        ax (matplotlib.Axes): Axes object for further handling.
    """
    if figsize is None:
        figsize=[n_cols * 6, n_rows * 4]

    _, ax = plt.subplots(n_rows, n_cols, constrained_layout=True, figsize=figsize)

    return ax

def plot_signal(signal, *, ax=None, sr=None, times=None, title=None):
    """
    Plot a time signal.

    Args:
        signal (np.ndarray): signal to plot.
        times  (np.ndarray): timestamps.

    """
    if ax is None:
        ax = new_axes()

    if times is not None:
        ax.plot(times, signal, c='black', linewidth=.4)
    elif sr is not None:
        times = np.arange(signal.size) / sr
        ax.plot(times, signal, c='black', linewidth=.4)
    else:
        ax.plot(signal, c='black', linewidth=.4)
        ax.set_xticks([])

    ax.set_yticks([])
    ax.set_xlabel('Time (s)')
    ax.set_title(title)

def display_spectrogram(spec, 
                        *, 
                        ax=None, 
                        title=None, 
                        f_bins=None, 
                        t_frames=None,
                        low_f=None,
                        high_f=None,
                        low_t=None,
                        high_t=None,
                        log=False,
                        logmin=-80):
    """
    Display spectrogram.

    Args:
        spec (np.ndarray): Spectrogram.
        ax (matplotlib.Axes): Axes object in which to display spectrogram.
        title (string): title for plot.
        f_bins (np.ndarray): frequency bins of spectrogram (see eq. 7).
        t_frames (np.ndarray): time frames of spectrogram (see eq. 8)
        low_f (double): Remove frequencies below low_f (in Hz).
        high_f (double): Remove frequencies above high_f (in Hz).
        low_t (double): Remove frames below low_t (in s).
        high_t (double): Remove frames above high_t (in s).
        log  (bool): Display spectrogram in log/decibel scale.
        logmin (int): Minimum value for scale display if log scale (in dB)
    """

    if ax is None:
        ax = new_axes()

    extent = None

    has_axis = f_bins is not None and t_frames is not None # can only filter if frequency/time axis values given.
    if has_axis:
        if low_f:
            idx = idx_at_value(low_f, f_bins)
            f_bins = f_bins[idx:]
            spec = spec[idx:, :]
        if high_f:
            idx = idx_at_value(high_f, f_bins)
            f_bins = f_bins[:idx]
            spec = spec[:idx, :]
        if low_t:
            idx = idx_at_value(low_t, t_frames)
            t_frames = t_frames[idx:]
            spec = spec[:, idx:]
        if high_t:
            idx = idx_at_value(high_t, t_frames)
            t_frames = t_frames[:idx]
            spec = spec[:, :idx]
        
        extent = [t_frames[0], t_frames[-1], f_bins[0], f_bins[-1]]

    if log:
        spec = 10 * np.log10(spec + 1e-30)
        spec = np.maximum(spec, spec.max() + logmin)

    ax.imshow(spec, origin='lower', aspect='auto', cmap='magma', interpolation='none', extent=extent)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)

def audio_widget(signal, sr, title=None):
    """
    Return audio ipywidget to play audio.

    Args:
        signal (np.ndarray): Audio signal.
        sr (int)           : Sample rate (in Hz).
        title (string)     : Title of the signal.
    """
    audio_player = IPython.display.Audio(data=signal, rate=sr)
    out = ipywidgets.Output()
    with out:
        IPython.display.display(audio_player)
    combined_widget = ipywidgets.VBox([ipywidgets.Label(title), out])

    return combined_widget

def multiple_audio_widgets(sr, *signals, titles=None, cols=1):
    """
        Return multiple audio widgets into a single GridBox.

        Args:
            sr (int)             : sample rate (in Hz).
            *signals (np.ndarray): signals to display.
            titles (list[string]): titles of displays.
            cols (int)           : number of columns of GridBox.
    """
    if titles is None:
        titles = [None] * len(signals)
    
    widgets = []
    for i in range(len(signals)):
        title = None if i >= len(titles) else titles[i]
        widgets.append(audio_widget(signals[i], sr, title))

    return combine_widgets(*widgets, cols=cols)
