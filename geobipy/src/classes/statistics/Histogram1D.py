""" @Histogram_Class
Module describing an efficient histogram class
"""
from .baseDistribution import baseDistribution
from ...classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ...classes.core.StatArray import StatArray
from ...base.customFunctions import safeEval
from ...base import customPlots as cP
import numpy as np
import matplotlib.pyplot as plt
import sys

class Histogram1D(RectilinearMesh1D):
    """ Histogram class that can update and plot efficiently """

    def __init__(self, values=None, nBins=None, bins=None, name=None, units=None):
        """ Initialize a histogram """

        # Allow an null instantiation
        tmp = [values, nBins, bins, name, units]
        if (all([x is None for x in tmp])):
            return

        if not bins is None:
            if (isinstance(bins, StatArray)):
                tmp = StatArray(bins, name=name, units=units)
            else:
                tmp = StatArray(bins, name=name, units=units)
        else:
            assert not values is None, ValueError("missing 1 required argument: counts")
            assert not nBins is None, ValueError("missing 1 required argument: 'nBins'")

            a = np.nanmin(values)
            b = np.nanmax(values)
            tmp = StatArray(np.linspace(a,b,nBins), name=name, units=units)

        # Initialize the parent class
        RectilinearMesh1D.__init__(self, x=tmp, name=None, units=None, dtype=np.int64)
        self.arr = StatArray(tmp.size, 'Frequency', dtype=np.int64)
        # Create pointers for better variable naming
        self.bins = self.x
        self.counts = self.arr
        self.dBin = self.dx

        # Get the cell widths
        self.width = np.abs(np.diff(self.bins))
        self.width = self.width.append(self.width[-1])

        # Add the incoming values as counts to the histogram
        if (not values is None):
            self.update(values)


    def update(self, values):
        """ Update the histogram by counting the entries in values and incrementing the counts accordingly """
        values = np.reshape(values, np.size(values))
        if (self.isRegular):
            self.update_Regular(values)
            return
        self.update_irregular(values)


    def update_Regular(self, values):
        """ Update the counts of regular binned histogram given the values """
        iBin = np.int64((values - self.bins[0]) / self.dBin)
        iBin = np.maximum(iBin,0)
        iBin = np.minimum(iBin,self.counts.size-1)
        tmp = np.bincount(iBin,minlength = self.counts.size)
        self.counts += tmp


    def update_irregular(self, values):
        """ Update the counts of regular binned histogram given the values """
        iBin = self.bins.searchsorted(values)
        iBin[iBin>=self.counts.size] = self.counts.size - 1

        tmp = np.bincount(iBin,minlength = self.counts.size)
        self.counts += tmp


    def plot(self,rotate=False,flipY=False,trim=True,**kwargs):
        """ Plots the histogram """

        c = kwargs.pop('color',cP.wellSeparated[0])
        lw = kwargs.pop('linewidth',0.5)
        ec = kwargs.pop('edgecolor','k')

        ax = plt.gca()
        cP.pretty(ax)

        if (rotate):
            plt.barh(self.bins,self.counts,height=self.width,align='center', color = c, linewidth = lw, edgecolor = ec, **kwargs)
            cP.ylabel(self.bins.getNameUnits())
            cP.xlabel('Frequency')
        else:
            plt.bar(self.bins,self.counts,width=self.width,align='center', color = c, linewidth = lw, edgecolor = ec, **kwargs)
            cP.xlabel(self.bins.getNameUnits())
            cP.ylabel('Frequency')

        i0 = 0
        i1 = np.size(self.bins) - 1
        if (trim):
            while self.counts[i0] == 0:
                i0 += 1
            while self.counts[i1] == 0:
                i1 -= 1
        if (i1 > i0):
            if (rotate):
                plt.ylim(self.bins[i0] - 0.5 * self.width[i0], self.bins[i1] + 0.5 * self.width[i1])
            else:
                plt.xlim(self.bins[i0] - 0.5 * self.width[i0], self.bins[i1] + 0.5 * self.width[i1])
        if (flipY):
            ax = plt.gca()
            lim = ax.get_ylim()
            if (lim[1] > lim[0]):
                ax.set_ylim(lim[::-1])


    def hdfName(self):
        """ Reprodicibility procedure """
        return('Histogram1D()')


    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.bins.toHdf(grp, 'bins')
        self.counts.createHdf(grp, 'counts', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        self.counts.writeHdf(parent, myName+'/counts', index=index)


    def toHdf(self, h5obj, myName):
        """ Write the StatArray to an HDF object
        h5obj: :An HDF File or Group Object.
        """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.bins.toHdf(grp, 'bins')
        self.counts.toHdf(grp, 'counts')


    def fromHdf(self, grp, index=None):
        """ Reads in the object froma HDF file """
        item = grp.get('bins')
        obj = eval(safeEval(item.attrs.get('repr')))
        bins = obj.fromHdf(item)
        Hist = Histogram1D(bins=bins)
        item = grp.get('counts')
        obj = eval(safeEval(item.attrs.get('repr')))
        if (index is None):
            Hist.counts = obj.fromHdf(item)
        else:
            Hist.counts = obj.fromHdf(item, index=np.s_[index,:])
        Hist.arr = Hist.counts
        return Hist
