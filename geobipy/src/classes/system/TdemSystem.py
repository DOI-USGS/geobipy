from numpy import log10, logspace, max, maximum, min, size

from ...classes.core.myObject import myObject
from ...base import fileIO as fIO
from ..statistics import StatArray
from .EmLoop import EmLoop
from .CircularLoop import CircularLoop
from .TdemSystem_GAAEM import TdemSystem_GAAEM
# import empymod


class TdemSystem(TdemSystem_GAAEM):
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

    def __init__(self, offTimes=None, transmitterLoop=None, receiverLoop=None, loopOffset=None, waveform=None, offTimeFilters=None, components=['z'], system_filename=None):
        """Instantiate"""

        if not system_filename is None:
            return super().__init__(system_filename)

        # self.offTimes = StatArray.StatArray(offTimes, 'Time', 's')
        # self._off_time = StatArray.StatArray(offTimes, 'Time', 's')
        # self.transmitterLoop = transmitterLoop
        # self.receiverLoop = receiverLoop
        # self.loopOffset = loopOffset
        # self.waveform = waveform
        # self.offTimeFilters = offTimeFilters
        # self.delayTime = 1.8e-7
        # self._components = None

        # self.modellingTimes, self.modellingFrequencies, self.ft, self.ftarg = empymod.utils.check_time(
        #     time=self.get_modellingTimes,          # Required times
        #     signal=-1,           # Switch-off response
        #     ft='dlf',           # Use DLF
        #     ftarg={'fftfilt': 'key_81_CosSin_2009'},
        #     verb=0)

    def __deepcopy__(self, memo={}):
        return None

    @property
    def n_channels(self):
        return self.nTimes * self.n_components

    @property
    def nTimes(self):
        return self.off_time.size

    @property
    def off_time(self):
        """Time windows."""
        return self._off_time

    @off_time.setter
    def off_time(self, values):
        # if values is None:
        #     values = self.nTimes
        # else:
        #     assert size(values) == self.nTimes, ValueError("off_time must have size {}".format(self.nTimes))
        self._off_time = StatArray.StatArray(values, "Time", "s")

    @property
    def isGA(self):
        return False

    # @property
    # def loopOffset(self):
    #     return self._loopOffset

    # @loopOffset.setter
    # def loopOffset(self, values):
    #     if values is None:
    #         self._loopOffset = StatArray.StatArray(3, "Loop Offset", "m")
    #     else:
    #         assert size(values) == 3, ValueError(
    #             "loopOffset must have size 3 for offset in x, y, z.")
    #         self._loopOffset = StatArray.StatArray(values, "Loop Offset", "m")

    # @property
    # def nTimes(self):
    #     return size(self.off_times)

    # @property
    # def receiverLoop(self):
    #     return self._receiverLoop

    # @receiverLoop.setter
    # def receiverLoop(self, value):
    #     if value is None:
    #         self._receiverLoop = CircularLoop()
    #     else:
    #         assert isinstance(value, EmLoop), TypeError(
    #             "transmitterLoop must have type geobipy.EmLoop")
    #         self._receiverLoop = deepcopy(values)

    @property
    def summary(self):
        msg = ("TdemSystem: ")
        return msg

    # @property
    # def times(self):
    #     return self.offTimes

    # @property
    # def transmitterLoop(self):
    #     return self._transmitterLoop

    # @transmitterLoop.setter
    # def transmitterLoop(self, value):
    #     if value is None:
    #         self._transmitterLoop = CircularLoop()
    #     else:
    #         assert isinstance(value, EmLoop), TypeError(
    #             "transmitterLoop must have type geobipy.EmLoop")
    #         self._transmitterLoop = deepcopy(values)

    @classmethod
    def read(cls, system_filename):
        return cls(system_filename=system_filename)

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

        tmin = log10(maximum(min(self.off_time) - max(self.waveform.time-self.delayTime), 1e-10))
        tmax = log10(max(self.off_time) - min(self.waveform.time-self.delayTime))
        return logspace(tmin, tmax, self.nTimes+2)
