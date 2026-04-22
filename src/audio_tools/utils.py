import numpy as np
import ipywidgets

def idx_at_value(value, array):
    """
        Return index of closest value inside array.

        Args:
            value: value to find.
            array (np.ndarray): array of values

        Returns:
            index of closest value, 0 or array.size - 1 if value outside bounds of array or None.
    """
    if value is None:
        return array.size - 1
    return np.abs(array - value).argmin()

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

    return (array - array[0]) / array[-1]

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