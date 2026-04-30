import numpy as np
import ipywidgets

def generate_toy_signal(sr=1000, f1=100, f2=120, duration=.5, times=[.05, .3, .32, .45]):
    """
        Generate toy signal used in Fig. 1.

        Args:
            sr (int)   : sample rate of the signal.
            f1 (double): frequency of first sinusoid (in Hz).
            f2 (double): frequency of second sinusoid (in Hz).

        Returns:
            t      (np.ndarray): time values of the signal.
            signal (np.ndarray): toy signal.
    """
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples)

    # timestamps
    t1, t2, t3, t4 = times
    s1, s2, s3, s4 = int(t1 * sr), int(t2 * sr), int(t3 * sr), int(t4 * sr) # as samples

    # sinusoids
    signal = np.zeros_like(t)
    signal[s1:s2] += .2 * np.sin(2 * np.pi * f1 * t[s1:s2])
    signal[s1:s2] += .2 * np.sin(2 * np.pi * f2 * t[s1:s2])
    signal[s3:s4] += .2 * np.sin(2 * np.pi * f1 * t[s3:s4])

    return t, signal

def idx_at_value(array, *values):
    """
        Return index of closest value inside array.

        Args:
            array (np.ndarray): array of values
            values: values to find.

        Returns:
            index of closest value, 0 or array.size - 1 if value outside bounds of array or None.
    """
    return parse_args(lambda v: array.size - 1 if v is None else np.abs(array - v).argmin(), *values)

def closest_neighbor(value, array):
    """
        Return closest value inside array.

        Args:
            value: value to find.
            array (np.ndarray): array of values

        Returns:
            closest value, 0 or array.size - 1 if value outside bounds of array or None.
    """   
    return array[idx_at_value(value, array)]

def normalize(array):
    """
        Returns sorted array scaled between 0 and 1.

        Args:
            array (np.ndarray): array to normalize, sorted.
    """
    return (array - array[0]) / (array[-1] - array[0])

def parse_args(parser, *args):
    """
        Parse args given a parser function.

        Args:
            parser (function): parser function.
            *args            : list of arguments to parse.
    """
    if len(args) == 1:
        return parser(args[0])
    res = []
    for arg in args:
        res.append(parser(arg))
    return np.array(res)

def sample_at_time(sr, *args):
    """
        Returns sample indices given sample rate and time instants.

        Args:
            sr (int)  : sample rate (in Hz).
            *args (double(s)): time instants (in seconds).
    """
    return parse_args(lambda t: int(sr * t), *args)

def time_at_sample(sr, *args):
    """
        Returns time values given sample and sample rate.

        Args:
            sr (int)  : sample rate (in Hz).
            *args (double(s)): sample indices.
    """

    return parse_args(lambda s: s / sr, *args)

def combine_widgets(*widgets, cols=1):
    """
        Combine multiple widgets into a grid.

        Args:
            widgets (ipywidgets.widgets): widgets to combine.
            cols (int)                  : number of columns.
    """
    layout = ipywidgets.Layout(display='grid', grid_template_columns=f'repeat({cols}, 1fr)', grid_gap="10px")
    combined_widgets = [widget for widget in widgets]
    
    return ipywidgets.GridBox(combined_widgets, layout=layout)
