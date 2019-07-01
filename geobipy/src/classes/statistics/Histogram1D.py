""" @Histogram_Class
Module describing an efficient histogram class
"""
from ...classes.mesh.RectilinearMesh1D import RectilinearMesh1D
from ...classes.core import StatArray
from ...base import customFunctions as cF
from ...base import customPlots as cP
import numpy as np
import matplotlib.pyplot as plt
import sys

class Histogram1D(RectilinearMesh1D):
    """1D Histogram class that updates efficiently.
    
    Fast updating relies on knowing the bins ahead of time.

    Histogram1D(values, bins, binCentres, log, relativeTo)

    Parameters
    ----------
    bins : geobipy.StatArray or array_like, optional
        Specify the bin edges for the histogram. Can be regularly or irregularly increasing.
    binCentres : geobipy.StatArray or array_like, optional
        Specify the bin centres for the histogram. Can be regular or irregularly sized.
    log : 'e' or float, optional
        Entries are given in linear space, but internally bins and values are logged.
        Plotting is in log space.
    relativeTo : float, optional
        If a float is given, updates will be relative to this value.

    Returns
    -------
    out : Histogram1D
        1D histogram
    
    """

    def __init__(self, bins=None, binCentres=None, log=None, relativeTo=None):
        """ Initialize a histogram """

        # Allow an null instantiation
        if (bins is None and binCentres is None):
            return

        # assert not (not log is None and not relativeTo is None), ValueError("Cannot use log option when histogram is relative.")

        if not bins is None:
            assert isinstance(bins, StatArray.StatArray), TypeError("bins must be a geobpiy.StatArray")
            if relativeTo:
                relativeTo, label = cF._log(relativeTo, log=log)
                bins, label = cF._log(bins, log=log)
                bins -= relativeTo
            else:
                bins, label = cF._log(bins, log=log)

            bins.name = cF._logLabel(log) + bins.getName()
            
        if not binCentres is None:
            assert isinstance(binCentres, StatArray.StatArray), TypeError("binCentres must be a geobpiy.StatArray")
            if relativeTo:
                relativeTo, label = cF._log(relativeTo, log=log)
                binCentres, label = cF._log(binCentres, log=log)
                binCentres -= relativeTo
            else:
                binCentres, label = cF._log(binCentres, log=log)

            binCentres.name = cF._logLabel(log) + binCentres.getName()

        # Initialize the parent class
        super().__init__(cellEdges=bins, cellCentres=binCentres)

        self._counts = StatArray.StatArray(self.nCells, 'Frequency', dtype=np.int64)

        self.log = log
        self.relativeTo = None if relativeTo is None else StatArray.StatArray(relativeTo)


    @property
    def counts(self):
        return self._counts
    
    @property
    def bins(self):
        return self.cellEdges

    @property
    def binCentres(self):
        return self.cellCentres

    @property
    def nBins(self):
        return self.nCells

    @property
    def nSamples(self):
        return self._counts.sum()

    @property
    def isRelative(self):
        return not self.relativeTo is None


    @property
    def linearAbsoluteBinCentres(self):
        tmp = cF._power(self.log, self.binCentres)
        if self.relativeTo:
            tmp = tmp + self.relativeTo
        return tmp


    @property
    def linearAbsoluteBins(self):
        tmp = cF._power(self.log, self.bins)
        if self.relativeTo:
            tmp = tmp + self.relativeTo
        return tmp


    def __deepcopy__(self, memo):
        out = type(self)()
        out._cellCentres = self._cellCentres.deepcopy()
        out._cellEdges = self._cellEdges.deepcopy()
        out.isRegular = self.isRegular
        out.dx = self.dx
        out._counts = self._counts.deepcopy()
        out.log = self.log
        out.relativeTo = self.relativeTo
        
        return out


    def cellIndex(self, values, **kwargs):

        cc, dum = cF._log(values.flatten(), self.log)

        if self.isRelative:
            cc = cc - self.relativeTo

        return super().cellIndex(cc, **kwargs)


    
    def combine(self, other):
        """Combine two histograms together.

        Using the bin centres of other, finds the corresponding bins in self.  The counts of other are then added to self.

        Parameters
        ----------
        other : geobipy.Histogram1D
            A histogram to combine.

        """

        cc = other.cellCentres
        if other.relativeTo:
            cc = other.cellCentres + other.relativeTo
            
        cc = cF._power(cc, other.log)

        cc, dum = cF._log(cc, self.log)

        if self.isRelative:
            cc = cc - self.relativeTo

        iBin = self.cellIndex(cc, clip=True)
        self._counts[iBin] = self._counts[iBin] + other.counts


    def sample(self, nSamples):
        """Generates samples from the histogram.

        A uniform distribution is used for each bin to generate samples.  
        The number of samples generated per bin is scaled by the count for that bin using the requested number of samples.

        parameters
        ----------
        nSamples : int
            Number of samples to generate.

        Returns
        -------
        out : geobipy.StatArray
            The samples.

        """
        samplesPerBin = np.ceil(self.counts / self.counts.sum() * nSamples)
        samplesPerBin[self.counts == 0] = 0
        samples = StatArray.StatArray(np.empty(np.int(np.sum(samplesPerBin))), self.bins.name, self.bins.units)
        i0 = 0
        for i in range(self.nBins):
            i1 = i0 + np.int(samplesPerBin[i])
            samples[i0:i1] = np.random.uniform(low=self.bins[i], high=self.bins[i+1], size=np.int(samplesPerBin[i]))
            i0 = i1
        return samples


    def update(self, values, trim=False):
        """Update the histogram by counting the entry of values into the bins of the histogram.

        Updates the bin counts in the histogram using fast methods.
        The values are clipped so that values outside the bins are forced inside.
        Optionally, one can trim the values so that those outside the bins are not counted.
        
        Parameters
        ----------
        values : array_like
            Increments the count for the bin that each value falls into.
        trim : bool
            A negative index which would normally wrap will clip to 0 and self.bins.size instead.

        """
        # tmp, dum = cF._log(values, self.log)

        # if self.isRelative:
        #     tmp = tmp - self.relativeTo

        iBin = np.atleast_1d(self.cellIndex(values, clip=True, trim=trim))
        tmp = np.bincount(iBin, minlength = self.nBins)
        
        self._counts += tmp


    def pcolor(self, **kwargs):
        """pcolor the histogram

        Other Parameters
        ----------------
        alpha : scalar or array_like, optional
            If alpha is scalar, behaves like standard matplotlib alpha and opacity is applied to entire plot
            If array_like, each pixel is given an individual alpha value.
        log : 'e' or float, optional
            Take the log of the colour to a base. 'e' if log = 'e', and a number e.g. log = 10.
            Values in c that are <= 0 are masked.
        equalize : bool, optional
            Equalize the histogram of the colourmap so that all colours have an equal amount.
        nbins : int, optional
            Number of bins to use for histogram equalization.
        xscale : str, optional
            Scale the x axis? e.g. xscale = 'linear' or 'log'
        yscale : str, optional
            Scale the y axis? e.g. yscale = 'linear' or 'log'.
        flipX : bool, optional
            Flip the X axis
        flipY : bool, optional
            Flip the Y axis
        grid : bool, optional
            Plot the grid
        noColorbar : bool, optional
            Turn off the colour bar, useful if multiple customPlots plotting routines are used on the same figure.   
        trim : bool, optional
            Set the x and y limits to the first and last non zero values along each axis.

        See Also
        --------
        geobipy.customPlots.pcolor : For additional keywords

        """
        if self.isRelative:
            kwargs['y'] = self.bins + self.relativeTo
        return super().pcolor(self._counts, **kwargs)
        

    def plot(self, rotate=False, flipX=False, flipY=False, trim=True, normalize=False, **kwargs):
        """ Plots the histogram """

        if self.isRelative:
            bins = self.bins + self.relativeTo
        else:
            bins = self.bins

        # if "reciprocateX" in kwargs:
        #     bins = cF._power(self.bins, self.log)
        #     bins = 1.0 / bins
        #     bins, dum = cF._log(bins, self.log)

        ax = cP.hist(self.counts, bins, rotate=rotate, flipX=flipX, flipY=flipY, trim=trim, normalize=normalize, **kwargs)
        return ax


    def hdfName(self):
        """ Reprodicibility procedure """
        return('Histogram1D()')



    def createHdf(self, parent, myName, withPosterior=True, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()

        self._counts.createHdf(grp, 'counts', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)

        if not self.log is None:
            grp.create_dataset('log', data = self.log)

        if self.isRelative:
            self.bins.toHdf(grp, 'bins')
            self.relativeTo.createHdf(grp, 'relativeTo', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)
        else:
            self.bins.createHdf(grp, 'bins', withPosterior=withPosterior, nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, withPosterior=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        self._counts.writeHdf(parent, myName+'/counts', withPosterior=withPosterior, index=index)

        if self.isRelative:
            self.relativeTo.writeHdf(parent, myName+'/relativeTo', withPosterior=withPosterior, index=index)
        else:
            self.bins.writeHdf(parent, myName+'/bins', withPosterior=withPosterior, index=index)


    def toHdf(self, h5obj, myName):
        """ Write the StatArray to an HDF object
        h5obj: :An HDF File or Group Object.
        """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        self.bins.toHdf(grp, 'bins')
        self._counts.toHdf(grp, 'counts')
        if not self.log is None:
            grp.create_dataset('log', data = self.log)
        if not self.relativeTo is None:
            self.relativeTo.toHdf(grp, 'relativeTo')


    def fromHdf(self, grp, index=None):
        """ Reads in the object froma HDF file """

        try:
            item = grp.get('relativeTo')
            obj = eval(cF.safeEval(item.attrs.get('repr')))
            relativeTo = obj.fromHdf(item, index=index)
        except:
            relativeTo = None

        item = grp.get('bins')
        obj = eval(cF.safeEval(item.attrs.get('repr')))

        
        if relativeTo is None:
            if index is None:
                bins = obj.fromHdf(item)
            else:
                # slic = np.s_[index, :]
                bins = obj.fromHdf(item, index=index)
        else:
            bins = obj.fromHdf(item)


        item = grp.get('counts')
        obj = eval(cF.safeEval(item.attrs.get('repr')))

        if (index is None):
            counts = obj.fromHdf(item)
        else:
            slic = np.s_[index, :]
            counts = obj.fromHdf(item, index=index)

        try:
            log = np.asscalar(np.asarray(grp.get('log')))
        except:
            log = None

        if bins.shape[-1] == counts.shape[-1]:
            Hist = Histogram1D(binCentres = bins)
        else:
            Hist = Histogram1D(bins = bins)

        Hist._counts = counts
        Hist.log = log
        Hist.relativeTo = relativeTo

        return Hist


    def summary(self, out=False):

        msg = ("{}\n"
              "Bins: \n{}"
              "Counts:\n{}"
              "Values are logged to base {}\n"
              "Relative to: {}").format(type(self), RectilinearMesh1D.summary(self, True), self.counts.summary(True), self.log, self.relativeTo)

        return msg if out else print(msg)
        
        