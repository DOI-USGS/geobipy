""" @Histogram_Class
Module describing an efficient histogram class
"""
from .baseDistribution import baseDistribution
from ...classes.statistics.Histogram1D import Histogram1D
from ...classes.mesh.RectilinearMesh2D import RectilinearMesh2D
from ...classes.core.StatArray import StatArray
from ...base import customPlots as cP
import numpy as np
import matplotlib.pyplot as plt


class Histogram2D(RectilinearMesh2D):
    """ 2D Histogram class that can update and plot efficiently """

    def __init__(self, x=None, y=None, name=None, units=None):
        """ Instantiate a 2D histogram """
        if (x is None):
            return
        # Instantiate the parent class
        RectilinearMesh2D.__init__(self, x=x, y=y, name='Frequency', units=None, dtype=np.int64)
        # Point xBins to self.x to make variable names more intuitive
        self.xBins = self.x
        # Point yBins to self.y to make variable names more intuitive
        self.yBins = self.y
        # Point counts to self.arr to make variable names more intuitive
        self.counts = self.arr


    def create2DjointProbabilityDistribution(self, H1, H2):
        """ Given two histograms each of a single variable, regrid them to the
        same number of bins if necessary and take their outer product to obtain
         a 2D joint probability density """
        assert H1.bins.size == H2.bins.size, "Cannot do unequal bins yet"
        assert isinstance(H1, Histogram1D), TypeError("H1 must be a Histogram1D")
        assert isinstance(H2, Histogram1D), TypeError("H2 must be a Histogram1D")

        self.__init__(x=H1.bins, y=H2.bins)
        self.counts[:,:] = np.outer(H1.counts, H2.counts)


#    def update(self, values):
#        """ Update the histogram by counting the entries in values and incrementing the counts accordingly """
#        values = np.reshape(values, np.size(values))
#        if (self.isRegular):
#            self.update_Regular(values)
#            return
#        self.update_irregular(values)
#
#
#
#
#
#    def update_Regular(self, values):
#        """ Update the counts of regular binned histogram given the values """
#        iBin = np.int64((values - self.bins[0]) / self.dBin)
#        iBin = np.maximum(iBin,0)
#        iBin = np.minimum(iBin,self.counts.size-1)
#        tmp = np.bincount(iBin,minlength = self.counts.size)
#        self.counts += tmp
#
#
#    def update_irregular(self, values):
#        """ Update the counts of regular binned histogram given the values """
#        iBin = self.bins.searchsorted(values)
#        self.counts += np.bincount(iBin,minlength = self.counts.size)


















