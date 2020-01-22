from scipy.signal import (butter, freqs)
class butterworth(object):

    def __init__(self, *args, **kwargs):

        self.butter = butter(*args, **kwargs)


    def frequencyResponse(self, *args, **kwargs):

        return freqs(*self.butter, *args, **kwargs)