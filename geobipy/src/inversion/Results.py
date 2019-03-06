""" @EMinversion1D_MCMC_Results
Class to store EMinv1D inversion results. Contains plotting and writing to file procedures
"""
from os.path import join
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import pause
from matplotlib.ticker import MaxNLocator
from ..base import customPlots as cP
from ..base import customFunctions as cF
import numpy as np
from ..base import fileIO as fIO
import h5py
from ..base.HDF.hdfWrite import writeNumpy
from ..classes.core.StatArray import StatArray
from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.statistics.Histogram1D import Histogram1D
from ..classes.statistics.Distribution import Distribution
from ..classes.core.myObject import myObject
from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.model.Model1D import Model1D
from ..classes.core.Stopwatch import Stopwatch

class Results(myObject):
    """Define the results for the Bayesian MCMC Inversion.
    
    Contains histograms and inversion related variables that can be updated as the Bayesian inversion progresses.

    Results(saveMe, plotMe, savePNG, dataPoint, model, ID, \*\*kwargs)

    Parameters
    ----------
    saveMe : bool, optional
        Whether to save the results to HDF5 files.
    plotMe : bool, optional
        Whether to plot the results on the fly. Only use this in serial mode.
    savePNG : bool, optional
        Whether to save a png of each single data point results. Don't do this in parallel please.
    dataPoint : geobipy.dataPoint
        Datapoint to use in the inversion.
        The relative error prior must have been set with dataPoint.relErr.setPrior()
        The additive error prior must have been set with dataPoint.addErr.setPrior()
        The height prior must have been set with dataPoint.z.setPrior()
    model : geobipy.model
        Model representing the subsurface.
    ID : int, optional

    OtherParameters
    ---------------
    nMarkovChains : int, optional
        Number of markov chains that will be tested.
    plotEvery : int, optional
        When plotMe = True, update the plot when plotEvery iterations have progressed.
    parameterDisplayLimits : sequence of ints, optional
        Limits of the parameter axis in the hitmap plot.
    reciprocateParameters : bool, optional
        Take the reciprocal of the parameters when plotting the hitmap.
    reciprocateName : str, optional
        Name of the parameters if they are reciprocated.
    reciprocateUnits : str, optional
        Units of the parameters if they are reciprocated.
    priMu : float, optional
        Initial prior mean for the halfspace parameter.  Usually set to the numpy.log of the initial halfspace parameter value.
    priStd : float, optional
        Initial prior standard deviation for the halfspace parameter. Usually set to numpy.log(11)   
    
    """

    def __init__(self, dataPoint=None, model=None, ID=0.0, **kwargs):
        """ Initialize the results of the inversion """

        # Initialize a stopwatch to keep track of time
        self.clk = Stopwatch()
        self.invTime = np.float64(0.0)
        self.saveTime = np.float64(0.0)

        # Logicals of whether to plot or save
        self.saveMe = kwargs.pop('save', True)
        self.plotMe = kwargs.pop('plot', False)
        self.savePNG = kwargs.pop('savePNG', False)
        # Return none if important parameters are not used (used for hdf 5)
        if all(x1 is None for x1 in [dataPoint, model]):
            return

        assert self.plotMe or self.saveMe, Exception('You have chosen to neither view or save the inversion results!')

        nMarkovChains = kwargs.pop('nMarkovChains', 100000)
        plotEvery = kwargs.pop('plotEvery', nMarkovChains / 20)
        parameterDisplayLimits = kwargs.pop('parameterDisplayLimits', [0.0, 1.0])
        reciprocateParameters = kwargs.pop('reciprocateParameters', False)
        priMu = kwargs.pop('priMu', 1.0)
        priStd = kwargs.pop('priStd', np.log(11))

        verbose = kwargs.pop('verbose', False)

        # Set the ID for the data point the results pertain to
        # Data Point identifier
        self.ID = np.float64(ID)
        # Set the increment at which to plot results
        # Increment at which to update the results
        self.iPlot = np.int64(plotEvery)
        # Set the display limits of the parameter in the HitMap
        # Display limits for parameters
        self.limits = np.zeros(2, dtype=np.float64) + parameterDisplayLimits
        # Should we plot resistivity or Conductivity?
        # Logical whether to take the reciprocal of the parameters
        self.invertPar = reciprocateParameters
        # Set the screen resolution
        # Screen Size
        self.sx = np.int32(1920)
        self.sy = np.int32(1080)
        # Copy the number of systems
        # Number of systems in the DataPoint
        self.nSystems = np.int32(dataPoint.nSystems)
        # Copy the number of Markov Chains
        # Number of Markov Chains to use
        self.nMC = np.int64(nMarkovChains)
        # Initialize a list of iteration number (This might seem like a waste of memory, but is faster than calling np.arange(nMC) every time)
        # StatArray of precomputed integers
        self.iRange = StatArray(np.arange(2 * self.nMC), name="Iteration #", dtype=np.int64)
        # Initialize the current iteration number
        # Current iteration number
        self.i = np.int64(0)
        # Initialize the vectors to save results
        # StatArray of the data misfit
        self.PhiDs = StatArray(2 * self.nMC, name = 'Data Misfit')
        # Multiplier for discrepancy principle
        self.multiplier = np.float64(0.0)
        # Initialize the acceptance level
        # Model acceptance rate
        self.acceptance = 0.0
#    self.rate=np.zeros(np.int(self.nMC/1000)+1)
        n = 2 * np.int(self.nMC / 1000)
        self.rate = StatArray(n, name='% Acceptance')
        self.ratex = StatArray(np.arange(1, n + 1) * 1000, name='Iteration #')
        # Initialize the burned in state
        self.iBurn = self.nMC
        self.burnedIn = False
        # Initialize the index for the best model
        self.iBest = 0
        self.iBestV = StatArray(2*self.nMC, name='Iteration of best model')

        # Initialize the number of layers for the histogram

        self.kHist = Histogram1D(binCentres=StatArray(np.arange(0.0, model.maxLayers + 1.5), name="# of Layers"))
        # Initialize the histograms for the relative and Additive Errors
        rBins = dataPoint.relErr.prior.getBins()
        aBins = dataPoint.addErr.prior.getBins()

        log = None
        if isinstance(dataPoint, TdemDataPoint):
            log = 10
            aBins = np.exp(aBins)

        self.relErr = []
        self.addErr = []
        if (self.nSystems > 1):
            for i in range(self.nSystems):
                self.relErr.append(Histogram1D(bins = StatArray(rBins[i, :], name='$\epsilon_{Relative}x10^{2}$', units='%')))
                self.addErr.append(Histogram1D(bins = StatArray(aBins[i, :], name='$\epsilon_{Additive}$', units=dataPoint._data.units), log=log))
        else:
            self.relErr.append(Histogram1D(bins = StatArray(rBins, name='$\epsilon_{Relative}x10^{2}$', units='%')))
            self.addErr.append(Histogram1D(bins = StatArray(aBins, name='$\epsilon_{Additive}$', units=dataPoint._data.units), log=log))


        # Initialize the hit map of layers and conductivities
        zGrd = StatArray(np.arange(0.5 * np.exp(model.minDepth), 1.1 * np.exp(model.maxDepth), 0.5 * np.exp(model.minThickness)), model.depth.name, model.depth.units)

        tmp = 3.0 * priStd
        mGrd = StatArray(
                np.logspace(np.log10(np.exp(priMu - tmp)), np.log10(np.exp(priMu + tmp)), 250), 
                'Conductivity', '$Sm^{-1}$')

        self.iz = np.arange(zGrd.size)

        self.Hitmap = Hitmap2D(xBinCentres = mGrd, yBinCentres = zGrd)

        # Initialize the doi
        self.doi = self.Hitmap.y.cellCentres[0]

        self.meanInterp = StatArray(zGrd.size)
        self.bestInterp = StatArray(zGrd.size)
        self.opacityInterp = StatArray(zGrd.size)

        # Initialize the Elevation Histogram
        self.DzHist = Histogram1D(bins = StatArray(dataPoint.z.prior.getBins(), name=dataPoint.z.name, units=dataPoint.z.units))

        # Initialize the Model Depth Histogram
        self.MzHist = Histogram1D(binCentres = zGrd)

        # Set a tag to catch data points that are not minimizing
        self.zeroCount = 0

        self.fig = None
        # Initialize the figure region
        if self.plotMe:
            self.fig = plt.figure(0, facecolor='white', figsize=(10,7))
        self.initFigure()
        if self.plotMe:
            plt.show(block=False)

        # Initialize times in seconds
        self.invTime = np.float64(0.0)
        self.saveTime = np.float64(0.0)

        # Initialize the best data, current data and best model
        self.bestD = dataPoint
        self.bestModel = model


        self.verbose = verbose
        if verbose:
            self.allRelErr = StatArray([self.nSystems, self.nMC], name='$\epsilon_{Relative}x10^{2}$', units='%')
            self.allAddErr = StatArray([self.nSystems, self.nMC], name='$\epsilon_{Additive}$', units=dataPoint.d.units)
            self.allZ = StatArray(self.nMC, name='Height', units='m')
            self.posterior = StatArray(self.nMC, name='log(posterior)')
            self.posteriorComponents = StatArray([9, self.nMC], 'Components of the posterior')


#         Initialize and save the first figure
#        if self.savePNG:
#            figName = 'PNG/_tmp' + \
#                fIO.getFileNameInteger(self.i, np.int(np.log10(self.nMC))) + '.png'
#            plt.savefig(figName)

    def update(self, i, iBest, bestDataPoint, bestModel, dataPoint, multiplier, PhiD, model, posterior, posteriorComponents, clipRatio):
        """ Update the attributes of the plotter """
        if (not self.plotMe and not self.saveMe):
            return
        self.i = np.int64(i)
        self.iBest = np.int64(iBest)
        self.PhiDs[self.i - 1] = PhiD.copy()  # Store the data misfit
        self.multiplier = np.float64(multiplier)

        if (self.burnedIn):  # We need to update some plotting options
            # Added the layer depths to a list, we histogram this list every
            # iPlot iterations
            self.kHist.update(model.nCells[0], clip=True)
            self.DzHist.update(dataPoint.z[0], clip=True)
            for j in range(self.nSystems):
                self.relErr[j].update(dataPoint.relErr[j], clip=True)
                self.addErr[j].update(dataPoint.addErr[j], clip=True)

            model.addToHitMap(self.Hitmap)

            # Update the layer interface histogram
            if (model.nCells > 1):
                ratio = np.exp(np.diff(np.log(model.par)))
                m1 = ratio <= 1.0 - clipRatio
                m2 = ratio >= 1.0 + clipRatio
                #keep = np.ma.mask_or(m1, m2)
                keep = np.logical_not(np.ma.masked_invalid(ratio).mask) & np.ma.mask_or(m1,m2)
                tmp = model.depth[:-1]
                if (len(tmp) > 0):
                    self.MzHist.update(tmp[keep], clip=True)

            if (self.verbose):
                iTmp = self.i - self.iBurn
                for j in range(self.nSystems):
                    self.allRelErr[j,iTmp]=dataPoint.relErr[j]
                    self.allAddErr[j,iTmp]=dataPoint.addErr[j]
                self.posterior[iTmp] = np.log(posterior)
                self.allZ[iTmp] = dataPoint.z[0]
                self.posteriorComponents[:,iTmp] = posteriorComponents


        if (np.mod(i, 1000) == 0):
            ratePercent = 100.0 * (np.float64(self.acceptance) / np.float64(1000))
            self.rate[np.int(self.i / 1000) - 1] = ratePercent
            self.acceptance = 0
            if (ratePercent < 2.0):
                self.zeroCount += 1
            else:
                self.zeroCount = 0

        self.bestD = bestDataPoint
        self.bestModel = bestModel


    def initFigure(self, iFig=0, forcePlot=False):
        """ Initialize the plotting region """

        if self.plotMe or forcePlot:
            pass
        else:
            return
        # Setup the figure region. The figure window is split into a 4x3
        # region. Columns are able to span multiple rows

        # plt.ion()

        # self.fig = plt.figure(iFig, facecolor='white', figsize=(10,7))
        mngr = plt.get_current_fig_manager()
        try:
            mngr.window.setGeometry(0, 10, self.sx, self.sy)
        except:
            pass
        nCols = 3 * self.nSystems

        self.gs = gridspec.GridSpec(12, nCols)
        self.gs.update(wspace=0.3 * self.nSystems, hspace=6.0)
        self.ax = [None]*(7+(2*self.nSystems))

        self.ax[0] = plt.subplot(self.gs[:3, :self.nSystems]) # Acceptance Rate
        self.ax[1] = plt.subplot(self.gs[3:6, :self.nSystems]) # Data misfit vs iteration
        self.ax[2] = plt.subplot(self.gs[6:9, :self.nSystems]) # Histogram of data point elevations
        self.ax[3] = plt.subplot(self.gs[9:, :self.nSystems]) # Histogram of # of layers
        self.ax[4] = plt.subplot(self.gs[:6,self.nSystems:2 * self.nSystems]) # Data fit plot
        self.ax[5] = plt.subplot(self.gs[6:,self.nSystems:2 * self.nSystems]) # 1D layer plot
        # Histogram of data errors 
        j = 5
        for i in range(self.nSystems):
            self.ax[j+1] = plt.subplot(self.gs[:3, 2 * self.nSystems + i]) # Relative Errors
            self.ax[j+2] = plt.subplot(self.gs[3:6,2 * self.nSystems + i]) # Additive Errors
            j += 2

        # Histogram of layer depths
        self.ax[(2*self.nSystems)+6] = plt.subplot(self.gs[6:, 2 * self.nSystems:])
        for ax in self.ax:
            cP.pretty(ax)

        if self.plotMe:
            plt.show(block=False)
        # plt.draw()


    def _plotAcceptanceVsIteration(self, **kwargs):
        """ Plots the acceptance percentage against iteration. """


        m = kwargs.pop('marker', 'o')
        a = kwargs.pop('alpha', 0.7)
        ls = kwargs.pop('linestyle', 'none')
        mec = kwargs.pop('markeredgecolor', 'k')

        self.rate.plot(self.ratex, i=np.s_[:np.int64(self.i / 1000)], marker=m, markeredgecolor=mec, linestyle=ls, **kwargs)
        cP.xlabel('Iteration #')
        cP.ylabel('% Acceptance')
        cP.title('Acceptance rate')


    def _plotMisfitVsIteration(self, **kwargs):
        """ Plot the data misfit against iteration. """

        m = kwargs.pop('marker', '.')
        ms = kwargs.pop('markersize', 2)
        a = kwargs.pop('alpha', 0.7)
        ls = kwargs.pop('linestyle', 'none')
        c = kwargs.pop('color', 'k')
        lw = kwargs.pop('linewidth', 3)
        
        self.PhiDs.plot(self.iRange, i=np.s_[:self.i], marker=m, alpha=a, markersize=ms, linestyle=ls, color=c, **kwargs)
        plt.ylabel('Data Misfit')
        dum = self.multiplier * len(self.bestD.iActive)
        plt.axhline(dum, color='#C92641', linestyle='dashed', linewidth=lw)
        if (self.burnedIn):
            plt.axvline(self.iBurn, color='#C92641', linestyle='dashed', linewidth=lw)
        plt.yscale('log')


    def _plotObservedPredictedData(self, **kwargs):
        """ Plot the observed and predicted data """

        self.bestD.plot(**kwargs)

        c = kwargs.pop('color', cP.wellSeparated[3])
        self.bestD.plotPredicted(color=c, **kwargs)


    def _plotNumberOfLayersPosterior(self, **kwargs):
        """ Plot the histogram of the number of layers """

        self.kHist.plot(**kwargs)
        plt.axvline(self.bestModel.nCells, color=cP.wellSeparated[3], linestyle='dashed', linewidth=3)


    def _plotElevationPosterior(self, **kwargs):
        """ Plot the histogram of the elevation """
        self.DzHist.plot(**kwargs)
        plt.axvline(self.bestD.z, color=cP.wellSeparated[3], linestyle='dashed', linewidth=3)

    
    def _plotRelativeErrorPosterior(self, system=0, **kwargs):
        """ Plots the histogram of the relative errors """

        self.relErr[system].plot(**kwargs)
        plt.locator_params(axis='x', nbins=4)
        plt.axvline(self.bestD.relErr[system], color=cP.wellSeparated[3], linestyle='dashed', linewidth=3)

    
    def _plotAdditiveErrorPosterior(self, system=0, **kwargs):
        """ Plot the histogram of the additive errors """
        
        self.addErr[system].plot(**kwargs)
        plt.locator_params(axis='x', nbins=4)
        log = self.addErr[system].log
        if self.bestD.addErr[system] > self.addErr[system].bins[-1]:
            log = 10
        loc, dum = cF._log(self.bestD.addErr[system], log=log)
        plt.axvline(loc, color=cP.wellSeparated[3], linestyle='dashed', linewidth=3)


    def _plotLayerDepthPosterior(self, **kwargs):
        """ Plot the histogram of layer interface depths """

        r = kwargs.pop('rotate', True)
        fY = kwargs.pop('flipY', True)
        tr = kwargs.pop('trim', False)

        self.MzHist.plot(rotate=r, flipY=fY, trim=tr, **kwargs)

    
    def _plotHitmapPosterior(self, confidenceInterval = 95.0, opacityPercentage = 67.0, **kwargs):
        """ Plot the hitmap posterior of conductivity with depth """

        # Get the mean and 95% confidence intervals
        (sigMed, sigLow, sigHigh) = self.Hitmap.confidenceIntervals(confidenceInterval)

        if (self.invertPar):
            x = 1.0 / self.Hitmap.x.cellCentres
            sl = 1.0 / sigLow
            sh = 1.0 / sigHigh
            xlabel = 'Resistivity ($\Omega m$)'
        else:
            x = self.Hitmap.x
            sl = sigLow
            sh = sigHigh
            xlabel = 'Conductivity ($Sm^{-1}$)'

        plt.pcolor(x, self.Hitmap.y.cellEdges, self.Hitmap.counts, cmap=mpl.cm.Greys)
        plt.plot(sl, self.Hitmap.y.cellCentres, color='#5046C8', linestyle='dashed', linewidth=2, alpha=0.6)
        plt.plot(sh, self.Hitmap.y.cellCentres, color='#5046C8', linestyle='dashed', linewidth=2, alpha=0.6)
        cP.xlabel(xlabel)

        # Plot the DOI cutoff based on percentage variance
        self.doi = self.Hitmap.getOpacityLevel(opacityPercentage)
        plt.axhline(self.doi, color='#5046C8', linestyle='dashed', linewidth=3)

        # Plot the best model
        self.bestModel.plot(flipY=False, reciprocateX=True, noLabels=True)
        plt.axis([self.limits[0], self.limits[1], self.Hitmap.y.cellEdges[0], self.Hitmap.y.cellEdges[-1]])
        ax = plt.gca()
        lim = ax.get_ylim()
        if (lim[1] > lim[0]):
            ax.set_ylim(lim[::-1])
        cP.ylabel(self.MzHist.bins.getNameUnits())
        plt.xscale('log')


    def plot(self, title="", iFig=0, forcePlot=False):
        """ Updates the figures for MCMC Inversion """
        # Plots that change with every iteration
        if self.plotMe or forcePlot:
            pass
        else:
            return

        if (not hasattr(self, 'gs')):
            self.initFigure(iFig, forcePlot=forcePlot)


        # fig = plt.figure(iFig)

#        if (np.mod(self.i, 1000) == 0 or forcePlot):

        if (np.mod(self.i, self.iPlot) == 0 or forcePlot):

            # Update the acceptance plot
            plt.sca(self.ax[0])
            plt.cla()
            self._plotAcceptanceVsIteration()
            
            # Update the data misfit vs iteration
            plt.sca(self.ax[1])
            plt.cla()
            self._plotMisfitVsIteration()

            # If the Best Data have changed, update the plot
            plt.sca(self.ax[4])
            # ax = plt.subplot(self.gs[:6, self.nSystems:2 * self.nSystems])
            plt.cla()
            self._plotObservedPredictedData()

            if (self.burnedIn):
                # Update the histogram of the number of layers
                plt.sca(self.ax[3])
                # plt.subplot(self.gs[9:, :self.nSystems])
                plt.cla()
                self._plotNumberOfLayersPosterior()
                self.ax[3].xaxis.set_major_locator(MaxNLocator(integer=True))

                # Histogram of the data point elevation
                plt.sca(self.ax[2])
                # plt.subplot(self.gs[6:9, :self.nSystems])
                plt.cla()
                self._plotElevationPosterior()

                j = 5
                for i in range(self.nSystems):
                    # Update the histogram of relative data errors
                    # plt.subplot(self.gs[:3, 2 * self.nSystems + j])
                    plt.sca(self.ax[j+1])
                    plt.cla()
                    self._plotRelativeErrorPosterior()
                    cP.title('System ' + str(i + 1))

                    # Update the histogram of additive data errors
                    plt.sca(self.ax[j+2])
                    # ax= plt.subplot(self.gs[3:6, 2 * self.nSystems + j])
                    plt.cla()
                    self._plotAdditiveErrorPosterior()
                    j += 2

                # Update the layer depth histogram
                plt.sca(self.ax[(2*self.nSystems)+6])
                # plt.subplot(self.gs[6:, 2 * self.nSystems:])
                plt.cla()
                self._plotLayerDepthPosterior()

                # Update the model plot
                plt.sca(self.ax[5])
                # plt.subplot(self.gs[6:, self.nSystems:2 * self.nSystems])
                plt.cla()
                self._plotHitmapPosterior()

            cP.suptitle(title)


            if self.verbose & self.burnedIn:
                plt.figure(99)
                plt.cla()
                plt.subplot(411)
                iTmp = np.s_[:,:self.i-self.iBurn+1]

                self.allRelErr.plot(self.iRange, i=iTmp, axis=1, c='k')
                plt.subplot(412)
                self.allAddErr.plot(self.iRange, i=iTmp, axis=1, c='k')
                plt.subplot(413)
                self.posterior.plot(self.iRange, i=np.s_[:self.i-self.iBurn+1], c='k')
                plt.subplot(414)
                self.iBestV.plot(self.iRange, i=np.s_[:self.i-self.iBurn+1], c='k')


                plt.figure(100)
                plt.cla()
                plt.subplot(311)
                self.allRelErr.plot(x=self.posterior, i=iTmp, axis=1, marker='o', linestyle='none',markersize=2, alpha=0.7, markeredgewidth=1)
                plt.subplot(312)
                self.allAddErr.plot(x=self.posterior, i=iTmp, axis=1, marker='o', linestyle='none', markersize=2, alpha=0.7, markeredgewidth=1)
                plt.subplot(313)
                self.allZ.plot(x=self.posterior, i=np.s_[:self.i-self.iBurn+1], marker='o', linestyle='none', markersize=2, alpha=0.7, markeredgewidth=1)

                plt.figure(101)
                plt.cla()
                plt.subplot(211)
                iTmp = np.s_[:-1,:self.i-self.iBurn+1]
                self.posteriorComponents.stackedAreaPlot(self.iRange[:self.nMC], i=iTmp, axis=1, labels=['nCells','depth','parameter','gradient','relative','additive','height','calibration'])
                plt.ylim([-40.0, 1.0])
                plt.subplot(212)
                iTmp=np.s_[:self.i-self.iBurn+1]
                cP.plot(self.iRange[iTmp], self.posteriorComponents[-1,iTmp])
                plt.grid(b=True, which ='major', color='k', linestyle='--', linewidth=2)


        cP.pause(1e-9)
        # pause(1e-9)
        # return self.fig



    def saveToLines(self, h5obj, ID):
        """ Save the results to a HDF5 object for a line """
        self.clk.restart()
        self.toHdf(h5obj, str(ID))


    def save(self, outdir, ID):
        """ Save the results to their own HDF5 file """
        with h5py.File(join(outdir,str(ID)+'.h5'),'w') as f:
            self.toHdf(f, str(ID))

    def toPNG(self, directory, ID, dpi=300):
       """ save a png of the results """
       fig = plt.figure(0)
       fig.set_size_inches(19, 11)
       figName = join(directory,str(ID) + '.png')
       plt.savefig(figName, dpi=dpi)

       if (self.verbose):
           fig = plt.figure(99)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(ID) + '_rap.png')
           plt.savefig(figName, dpi=dpi)

           fig = plt.figure(100)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(ID) + '_xp.png')
           plt.savefig(figName, dpi=dpi)

           fig = plt.figure(101)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(ID) + '_stack.png')
           plt.savefig(figName, dpi=dpi)

    def hdfName(self):
        """ Reprodicibility procedure """
        return('Results()')

    def createHdf(self, parent, myName):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        assert isinstance(myName,str), 'myName must be a string'
        # create a new group inside h5obj
        try:
            grp = parent.create_group(myName)
        except:
            # Assume that the file already has the template and return
            return
        grp.attrs["repr"] = self.hdfName()

        grp.create_dataset('i', (1,), dtype=self.i.dtype)
        grp.create_dataset('iplot', (1,), dtype=self.iPlot.dtype)
        grp.create_dataset('plotme', (1,), dtype=type(self.plotMe))
        grp.create_dataset('limits', (2,), dtype=self.limits.dtype)
        grp.create_dataset('dispres', (1,), dtype=type(self.invertPar))
        grp.create_dataset('sx', (1,), dtype=self.sx.dtype)
        grp.create_dataset('sy', (1,), dtype=self.sy.dtype)
        grp.create_dataset('nmc', (1,), dtype=self.nMC.dtype)
        grp.create_dataset('nsystems', (1,), dtype=self.nSystems.dtype)
        grp.create_dataset('iburn', (1,), dtype=self.iBurn.dtype)
        grp.create_dataset('burnedin', (1,), dtype=type(self.burnedIn))
        grp.create_dataset('doi', (1,), dtype=self.doi.dtype)
        grp.create_dataset('multiplier', (1,), dtype=self.multiplier.dtype)
        grp.create_dataset('invtime', (1,), dtype=float)
        grp.create_dataset('savetime', (1,), dtype=float)

        nz=self.Hitmap.y.nCells
        grp.create_dataset('meaninterp', (nz,), dtype=np.float64)
        grp.create_dataset('bestinterp', (nz,), dtype=np.float64)
#        grp.create_dataset('opacityinterp', (nz,), dtype=np.float64)

        self.rate.createHdf(grp,'rate')
        self.ratex.createHdf(grp,'ratex')
        self.PhiDs.createHdf(grp,'phids')
        self.kHist.createHdf(grp, 'khist')
        self.DzHist.createHdf(grp, 'dzhist')
        self.MzHist.createHdf(grp, 'mzhist')
        for i in range(self.nSystems):
            self.relErr[i].createHdf(grp, 'relerr' + str(i))
        for i in range(self.nSystems):
            self.addErr[i].createHdf(grp, 'adderr' + str(i))

        self.Hitmap.createHdf(grp,'hitmap')
        self.bestD.createHdf(grp, 'bestd')

        tmp=self.bestModel.pad(self.bestModel.maxLayers)
        tmp.createHdf(grp, 'bestmodel')


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        index: optional numpy slice of where to place the arr in the hdf data object
        """
        assert isinstance(myName,str), 'myName must be a string'

        grp = parent.get(myName)
        writeNumpy(self.i, grp, 'i')
        writeNumpy(self.iPlot, grp, 'iplot')
        writeNumpy(self.plotMe, grp, 'plotme')
        writeNumpy(self.limits, grp, 'limits')
        writeNumpy(self.invertPar, grp, 'dispres')
        writeNumpy(self.sx, grp, 'sx')
        writeNumpy(self.sy, grp, 'sy')
        writeNumpy(self.nMC, grp, 'nmc')
        writeNumpy(self.nSystems, grp, 'nsystems')
        writeNumpy(self.iBurn, grp, 'iburn')
        writeNumpy(self.burnedIn, grp, 'burnedin')
        self.doi = self.Hitmap.getOpacityLevel(67.0)
        writeNumpy(self.doi, grp, 'doi')
        writeNumpy(self.multiplier, grp, 'multiplier')
        writeNumpy(self.invTime, grp, 'invtime')

        self.rate.writeHdf(grp, 'rate')
        self.ratex.writeHdf(grp, 'ratex')
        self.PhiDs.writeHdf(grp, 'phids')
        self.kHist.writeHdf(grp, 'khist')
        self.DzHist.writeHdf(grp, 'dzhist')
        self.MzHist.writeHdf(grp, 'mzhist')
         # Histograms for each system
        for i in range(self.nSystems):
            self.relErr[i].writeHdf(grp, 'relerr' + str(i))
        for i in range(self.nSystems):
            self.addErr[i].writeHdf(grp, 'adderr' + str(i))

        self.Hitmap.writeHdf(grp,'hitmap')

        mean = self.Hitmap.getMeanInterval()
        best = self.bestModel.interp2depth(self.bestModel.par, self.Hitmap)
#        opacity = self.Hitmap.getOpacity()
        writeNumpy(mean, grp,'meaninterp')
        writeNumpy(best, grp,'bestinterp')
#        writeNumpy(opacity, grp,'opacityinterp')

        self.clk.stop()
        self.saveTime = np.float64(self.clk.timeinSeconds())
        writeNumpy(self.saveTime, grp, 'savetime')
        self.bestD.writeHdf(grp, 'bestd')
        self.bestModel.writeHdf(grp, 'bestmodel')

    def toHdf(self, h5obj, myName):
        """ Write the object to a HDF file """
        # Create a new group inside h5obj
        try:
            grp = h5obj.create_group(myName)
        except:
            del h5obj[myName]
            grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        # Begin by writing the parameters
#        grp.create_dataset('id', data=self.ID)
        grp.create_dataset('i', data=self.i)
        grp.create_dataset('iplot', data=self.iPlot)
        grp.create_dataset('plotme', data=self.plotMe)
        grp.create_dataset('limits', data=self.limits)
        grp.create_dataset('dispres', data=self.invertPar)
        grp.create_dataset('sx', data=self.sx)
        grp.create_dataset('sy', data=self.sy)
        grp.create_dataset('nmc', data=self.nMC)
        grp.create_dataset('nsystems', data=self.nSystems)
        grp.create_dataset('iburn', data=self.iBurn)
        grp.create_dataset('burnedin', data=self.burnedIn)
        self.doi = self.Hitmap.getOpacityLevel(67.0)
        grp.create_dataset('doi', data=self.doi)
        # Large vector of phiD
        grp.create_dataset('multiplier', data=self.multiplier)
        # Rate of acceptance
        self.rate.writeHdf(grp, 'rate')
        # x Axis for acceptance rate
        self.ratex.writeHdf(grp, 'ratex')
        # Data Misfit
        self.PhiDs.writeHdf(grp, 'phids')
        # Histogram of # of Layers
        self.kHist.writeHdf(grp, 'khist')
        # Histogram of Elevations
        self.DzHist.writeHdf(grp, 'dzhist')
        # Histogram of Layer depths
        self.MzHist.writeHdf(grp, 'mzhist')
        # Hit Maps
        self.Hitmap.writeHdf(grp, 'hitmap')
        # Write the Best Data
        self.bestD.writeHdf(grp, 'bestd')
        # Write the Best Model
        self.bestModel.writeHdf(grp, 'bestmodel')
        # Interpolate the mean and best model to the discretized hitmap
        mean = self.Hitmap.getMeanInterval()
        best = self.bestModel.interp2depth(self.bestModel.par, self.Hitmap)
#        opacity = self.Hitmap.getOpacity()

        grp.create_dataset('meaninterp', data=mean)
        grp.create_dataset('bestinterp', data=best)
#        grp.create_dataset('opacityinterp', data=opacity)


        # Histograms for each system
        for i in range(self.nSystems):
            self.relErr[i].toHdf(grp, 'relerr' + str(i))
        for i in range(self.nSystems):
            self.addErr[i].toHdf(grp, 'adderr' + str(i))

        grp.create_dataset('invtime', data=self.invTime)
        self.clk.stop()
        self.saveTime = self.clk.timeinSeconds()
        grp.create_dataset('savetime', data=self.saveTime)

    def fromHdf(self, grp, sysPath = ''):
        """ Reads in the object froma HDF file """
        self.ID = np.array(grp.get('id'))
        self.i = np.array(grp.get('i'))
        tmp = grp.get('iplot')
        if tmp is None:
            tmp = grp.get('iPlot')
        self.iPlot = np.array(tmp)

        tmp = grp.get('plotme')
        if tmp is None:
            tmp = grp.get('plotMe')
        self.plotMe = np.array(tmp)

        self.limits = np.array(grp.get('limits'))

        tmp = grp.get('dispres')
        if tmp is None:
            tmp = grp.get('dispRes')
        self.invertPar = np.array(tmp)

        self.sx = np.array(grp.get('sx'))
        self.sy = np.array(grp.get('sy'))
        tmp = grp.get('nmc')
        if (tmp is None):
            tmp = np.array(grp.get('nMC'))
        self.nMC = np.array(tmp)
        # Initialize a list of iteration number
        self.iRange = StatArray(np.arange(2 * self.nMC), name="Iteration #", dtype=np.int64)

        tmp = grp.get('nsystems')
        if tmp is None:
            tmp = grp.get('nSystems')
        self.nSystems = np.array(tmp)

        tmp = grp.get('iburn')
        if tmp is None:
            tmp = grp.get('iBurn')
        self.iBurn = np.array(tmp)

        tmp = grp.get('burnedin')
        if tmp is None:
            tmp = grp.get('burnedIn')
        self.burnedIn = np.array(tmp)

        self.doi = np.array(grp.get('doi'))
        self.multiplier = np.array(grp.get('multiplier'))

        item = grp.get('rate')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        self.rate = obj.fromHdf(item)

        item = grp.get('ratex')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        self.ratex = obj.fromHdf(item)

        item = grp.get('phids')
        if (item is None):
            item = grp.get('PhiDs')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        self.PhiDs = obj.fromHdf(item)

        item = grp.get('khist')
        if (item is None):
            item = grp.get('kHist')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        self.kHist = obj.fromHdf(item)

        item = grp.get('dzhist')
        if (item is None):
            item = grp.get('DzHist')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        self.DzHist = obj.fromHdf(item)

        item = grp.get('mzhist')
        if (item is None):
            item = grp.get('MzHist')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        self.MzHist = obj.fromHdf(item)

        item = grp.get('hitmap')
        if (item is None):
            item = grp.get('HitMap')
        s = item.attrs.get('repr')
        obj = eval(s)
        self.Hitmap = obj.fromHdf(item)

        item = grp.get('bestd')
        if (item is None):
            item = grp.get('bestD')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        self.bestD = obj.fromHdf(item, sysPath=sysPath)

        item = grp.get('bestmodel')
        if (item is None):
            item = grp.get('bestModel')
        obj = eval(cF.safeEval(item.attrs.get('repr')))
        self.bestModel = obj.fromHdf(item)
        self.bestModel.maxDepth = np.log(self.Hitmap.y.cellCentres[-1])

        self.relErr = []
        self.addErr = []
        for i in range(self.nSystems):
            item = grp.get('relerr' + str(i))
            if (item is None):
                item = grp.get('relErr'+str(i))
            obj = eval(cF.safeEval(item.attrs.get('repr')))
            aHist = obj.fromHdf(item)
            self.relErr.append(aHist)
            item = grp.get('adderr' + str(i))
            if (item is None):
                item = grp.get('addErr'+str(i))
            obj = eval(cF.safeEval(item.attrs.get('repr')))
            aHist = obj.fromHdf(item)
            self.addErr.append(aHist)

        tmp = grp.get('invtime')
        if tmp is None:
            tmp = grp.get('invTime')
        self.invTime = np.array(tmp)

        tmp = grp.get('savetime')
        if tmp is None:
            tmp = grp.get('saveTime')
        self.saveTime = np.array(tmp)

        self.verbose = False

    def read(self, fName, grpName, sysPath = ''):
        """ Reads a data points results from HDF5 file """
        assert fIO.fileExists(fName), "Cannot find file "+fName
        with h5py.File(fName, 'r')as f:
          return self.read_fromH5Obj(f, fName, grpName, sysPath)

    def read_fromH5Obj(self, h5obj, fName, grpName, sysPath = ''):
        """ Reads a data points results from HDF5 file """
        grp = h5obj.get(grpName)
        assert not grp is None, "ID "+str(grpName) + " does not exist in file " + fName
        self.fromHdf(grp, sysPath)
