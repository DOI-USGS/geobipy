import numpy as np
from ..statistics import StatArray

class Waveform(object):
    """Defines a waveform using piece-wise linear segments


    Waveform(time, amplitude, current)

    Parameters
    ----------
    time : array_like
        The times of each changepoint within the waveform. e.g. [0.0, 1.0, 2.0, 3.0] for 3 linear segments
    amplitude : array_like
        The amplitude at each changepoint, e.g. [0.0, 1.0, 1.0, 0.0] would be a ramp up, flat, then ramp down.
    current : float
        The peak current of the waveform.

    """



    def __init__(self, time, amplitude, current):

        self._time = StatArray.StatArray(time, 'Time', 's')
        self._amplitude = StatArray.StatArray(amplitude, 'Amplitude')
        self._current = current

    @property
    def time(self):
        return self._time

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def current(self):
        return self._current

    def plot(self, **kwargs):
        self.amplitude.plot(x = self.time, **kwargs)