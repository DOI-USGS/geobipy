""" @FdemSystem_Class
Module describing a frequency domain EM acquisition system
"""
from ...classes.core.myObject import myObject
from .FdemSystem import FdemSystem
from ...classes.core.StatArray import StatArray
import numpy as np
from ...base import fileIO as fIO
from ...base import MPI as myMPI



class MTSystem(myObject):
    """ Defines a Frequency Domain ElectroMagnetic acquisition system """

    def __init__(self, nFreq=0):
        """ Initialize an FdemSystem """
        # Number of Frequencies
        self.nFreq = np.int64(nFreq)
        # StatArray of frequencies
        self.freq = StatArray(self.nFreq, "Frequency", "Hz")


    def read(self, fname):
        """ Read in a file containing the system information
        The file contains a line with the header names, and a row for each frequency
        freq
        378
        1776
        ...
        """
        self.__init__(fIO.getNlines(fname, 1))
        with open(fname) as f:
            tmp = f.readline().lower().split()

            assert 'freq' in tmp, ('Cannot read headers from MTSystem File ' + fname +
                                   '\nFirst line of system file should contain\nfreq')
            for j, line in enumerate(f):  # For each line in the file
                line = fIO.parseString(line)
                try:
                    self.freq[j] = np.float64(line[0])
                except Exception:
                    raise SystemExit(
                        "Could not read from system file:" + fname + " Line:" + str(j + 2))


    def summary(self):
        """ print a summary of the MTSystem """
        print("MTSystem:")
        self.freq.summary()
        print('')


    def hdfName(self):
        return('MTSystem(0)')


    def Bcast(self, world):
        """ Broadcast the MTSystem using MPI """
        nFreq = myMPI.Bcast(self.nFreq, world)
        if (world.rank > 0):
            self = MTSystem(nFreq)
        this = MTSystem(nFreq)
        this.freq = self.freq.Bcast(world)
        return this