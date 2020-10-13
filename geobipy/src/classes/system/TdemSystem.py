import numpy as np
from ...classes.core.myObject import myObject
from ...base import fileIO as fIO
from ...classes.core import StatArray
from .EmLoop import EmLoop
from .CircularLoop import CircularLoop
import empymod

class TdemSystem(myObject):
    """ Initialize a Time domain system class

    TdemSystem(systemFileName)

    Parameters
    ----------
    systemFileName : str
        The system file to read from

    Returns
    -------
    out : TdemSystem
        A time domain system class

    """

    def __init__(self, offTimes=None, transmitterLoop = None, receiverLoop=None, loopOffset=None, waveform=None, offTimeFilters=None):
        """Instantiate"""

        if offTimes is None:
            return

        self.offTimes = StatArray.StatArray(offTimes, 'Time', 's')
        self.transmitterLoop = transmitterLoop
        self.receiverLoop = receiverLoop
        self.loopOffset = loopOffset
        self.waveform = waveform
        self.offTimeFilters = offTimeFilters
        self.delayTime = 1.8e-7


        self.modellingTimes, self.modellingFrequencies, self.ft, self.ftarg = empymod.utils.check_time(
            time=self.get_modellingTimes,          # Required times
            signal=-1,           # Switch-off response
            ft='dlf',           # Use DLF
            ftarg={'fftfilt': 'key_81_CosSin_2009'},
            verb=0)

    @property
    def isGA(self):
        return False

    @property
    def loopOffset(self):
        return self._loopOffset

    @loopOffset.setter
    def loopOffset(self, values):
        if values is None:
            self._loopOffset = StatArray.StatArray(3, "Loop Offset", "m")
        else:
            assert np.size(values) == 3, ValueError("loopOffset must have size 3 for offset in x, y, z.")
            self._loopOffset = StatArray.StatArray(values, "Loop Offset", "m")

    @property
    def nTimes(self):
        return np.size(self.offTimes)

    @property
    def receiverLoop(self):
        return self._receiverLoop

    @receiverLoop.setter
    def receiverLoop(self, value):
        if value is None:
            self._receiverLoop = CircularLoop()
        else:
            assert isinstance(value, EmLoop), TypeError("transmitterLoop must have type geobipy.EmLoop")
            self._receiverLoop = value.deepcopy()

    def summary(self, out=False):
        msg = ("TdemSystem: ")
        return msg if out else print(msg)

    @property
    def times(self):
        return self.offTimes

    @times.setter
    def times(self, values):
        if values is None:
            self._times = StatArray.StatArray(self.nTimes, "Time", "s")
        else:
            assert np.size(values) == self.nTimes, ValueError("times must have size {}".format(self.nTimes))
            self._times = StatArray.StatArray(values, "Time", "s")

    @property
    def transmitterLoop(self):
        return self._transmitterLoop

    @transmitterLoop.setter
    def transmitterLoop(self, value):
        if value is None:
            self._transmitterLoop = CircularLoop()
        else:
            assert isinstance(value, EmLoop), TypeError("transmitterLoop must have type geobipy.EmLoop")
            self._transmitterLoop = value.deepcopy()


    def read(self, systemFilename):
        # Read in the System file
        from .TdemSystem_GAAEM import TdemSystem_GAAEM
        self = TdemSystem_GAAEM(systemFilename)
        assert np.min(np.diff(self.windows.centre)) > 0.0, ValueError("Receiver window times must monotonically increase for system "+systemFilename)

        self.readCurrentWaveform(systemFilename)
        return self


    @property
    def get_modellingTimes(self):
        """Generates regularly log spaced times that covers both the waveform and measurment times.

        Parameters
        ----------
        waveformTimes : array_like
            Times of the waveform change points

        measurementTimes : array_like
            Measurement times for the system

        Returns
        -------
        out : array_like
            Times spanning both the waveform and measrement times

        """

        tmin = np.log10(np.maximum(np.min(self.times) - np.max(self.waveform.time-self.delayTime), 1e-10))
        tmax = np.log10(np.max(self.times) - np.min(self.waveform.time-self.delayTime))
        return np.logspace(tmin, tmax, self.nTimes+2)