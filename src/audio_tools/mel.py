import numpy as np

def mel(f):
    """
        Mel scale O'Shaughnessy's formula (eq. 54).
        
        Args:
            f (double): Frequency (in Hz).

        Returns:
            Mel associated to f.
    """
    return 2595 * np.log10(1 + f / 700)

def imel(m):
    """
        Inverse-mel scale O'Shaughnessy's formula (eq. 55).
        
        Args:
            m (double): Mels.

        Returns:
            Frequency associated to m.
    """
    return 700 * (10 ** (m / 2595) - 1)

def mel_frequency_bins(n_bins, sr):
    """
        Compute mel frequency scale as explained in Section V-D.

        Args:
            n_bins (int): number of mel bins.
            sr (int)    : sample rate (in Hz).

        Returns:
            Mel scale with n_bins.
    """
    mr = mel(sr / 2)
    mels = np.linspace(0, mr, n_bins)
    m_bins = imel(mels)

    return m_bins
