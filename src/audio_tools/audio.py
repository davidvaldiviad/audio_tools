import numpy as np

from .utils import *

def trim_audio(signal, sr, start_t, end_t):
    """
        Trim an audio signal from start_t to end_t, in seconds.

        Args:
            signal (np.ndarray): audio signal to cut.
            sr (int)           : sample rate (in Hz).
            start_t (double)   : start of audio trimming (in seconds).
            end (double)       : end of audio trimming (in seconds).
    """

    s1, s2 = sample_at_time(sr, start_t, end_t)

    return signal[s1:s2]
    