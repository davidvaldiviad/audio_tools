import numpy as np
from math import floor
from scipy.signal.windows import hann

from .utils import * 

class Spectrogram:
    """
        Spectrogram class.
    """
    def __init__(self, 
                 signal,
                 *,
                 window_size=None,
                 hop_size=None,
                 nfft=None,
                 sr=None,
                 window_size_s=None,
                 hop_size_s=None,
                 window=None,
                 real=True):
        self.signal        = signal
        self.window_size   = window_size
        self.hop_size      = hop_size
        self.nfft          = nfft
        self.sr            = sr
        self.window_size_s = window_size_s
        self.hop_size_s    = hop_size_s
        self.window        = window
        self.real          = real

        if self.window_size and self.sr:
            self.window_size_s = time_at_sample(self.sr, self.window_size)
        elif self.window_size_s and self.sr:
            self.window_size = sample_at_time(self.sr, self.window_size_s)

        assert self.window_size is not None, "Must provide window size!"

        if self.window is None:
            self.window = hann(self.window_size)

        if self.hop_size is None and self.hop_size_s is None:
            self.hop_size = self.window_size // 2

        if self.hop_size and self.sr:
            self.hop_size_s = time_at_sample(self.sr, self.hop_size)
        elif self.hop_size_s and self.sr:
            self.hop_size = sample_at_time(self.sr, self.hop_size_s)
            
        if self.nfft is None:
            self.nfft = self.window_size

        self.f_bins = np.arange(self.nfft)
        self.t_frames = np.arange(self.last_frame_index())

        if self.sr:
            self.f_bins = np.arange(self.nfft) / self.nfft * sr
            if self.real:
                self.f_bins = self.f_bins[:self.nfft // 2 + 1]
            self.t_frames = np.arange(self.last_frame_index() + 1) * self.hop_size_s

        self.f = self.f_bins.size
        self.t = self.t_frames.size

        self.shape = (self.t, self.f)
        self.size  = self.t * self.f

    def last_frame_index(self):
        return floor((self.signal.size + self.window_size // 2) / self.hop_size)

    def stft_pad(self):
        lfi       = self.last_frame_index()
        pad_left  = self.window_size // 2
        pad_right = lfi * self.hop_size + self.window_size // 2 - self.signal.size
        return np.pad(self.signal, (pad_left, pad_right))

    def segment_signal(self):
        window_size = self.window.size
        padded_signal = self.stft_pad()
        n_segments = self.last_frame_index() + 1
        segments = np.zeros((n_segments, window_size))
        for i in range(n_segments):
            i1, i2 = i * self.hop_size, i * self.hop_size + window_size
            segments[i] = padded_signal[i1:i2] * self.window
        return segments

    def stft(self):
        segments = self.segment_signal().T
        spectrum = np.fft.fft(segments, self.nfft, axis=0)
        if self.real:
            return spectrum[:self.nfft // 2 + 1, :]
        return spectrum
        
    def spectrogram(self):
        return np.abs(self.stft()**2)