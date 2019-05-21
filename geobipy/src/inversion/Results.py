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
from ..classes.core import StatArray
from ..classes.statistics.Hitmap2D import Hitmap2D
from ..classes.statistics.Histogram1D import Histogram1D
from ..classes.statistics.Distribution import Distribution
from ..classes.core.myObject import myObject
from ..classes.data.datapoint.FdemDataPoint import FdemDataPoint
from ..classes.data.datapoint.TdemDataPoint import TdemDataPoint
from ..classes.model.Model1D import Model1D
from ..classes.core.Stopwatch import Stopwatch
from ..base.HDF import hdfRead

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

    def __init__(self, dataPoint=None, model=None, fiducial=0.0, **kwargs):
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
        reciprocateParameter = kwargs.pop('reciprocateParameter', False)
        priMu = kwargs.pop('priMu', 1.0)
        priStd = kwargs.pop('priStd', np.log(11))

        verbose = kwargs.pop('verbose', False)

        # Set the ID for the data point the results pertain to
        # Data Point identifier
        self.fiducial = np.float(fiducial)
        # Set the increment at which to plot results
        # Increment at which to update the results
        self.iPlot = np.int64(plotEvery)
        # Set the display limits of the parameter in the HitMap
        # Display limits for parameters
        self.limits = np.asarray(parameterDisplayLimits) if not parameterDisplayLimits is None else None
        # Should we plot resistivity or Conductivity?
        # Logical whether to take the reciprocal of the parameters
        self.reciprocateParameter = reciprocateParameter
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
        self.iRange = StatArray.StatArray(np.arange(2 * self.nMC), name="Iteration #", dtype=np.int64)
        # Initialize the current iteration number
        # Current iteration number
        self.i = np.int64(0)
        # Initialize the vectors to save results
        # StatArray of the data misfit
        self.PhiDs = StatArray.StatArray(2 * self.nMC, name = 'Data Misfit')
        # Multiplier for discrepancy principle
        self.multiplier = np.float64(0.0)
        # Initialize the acceptance level
        # Model acceptance rate
        self.acceptance = 0.0
#    self.rate=np.zeros(np.int(self.nMC/1000)+1)
        n = 2 * np.int(self.nMC / 1000)
        self.rate = StatArray.StatArray(n, name='% Acceptance')
        self.ratex = StatArray.StatArray(np.arange(1, n + 1) * 1000, name='Iteration #')
        # Initialize the burned in state
        self.iBurn = self.nMC
        self.burnedIn = False
        # Initialize the index for the best model
        self.iBest = 0
        self.iBestV = StatArray.StatArray(2*self.nMC, name='Iteration of best model')

        self.iz = np.arange(model.par.posterior.y.nCells)

        # Initialize the doi
        self.doi = model.par.posterior.yBinCentres[0]

        self.meanInterp = StatArray.StatArray(model.par.posterior.y.nCells)
        self.bestInterp = StatArray.StatArray(model.par.posterior.y.nCells)
        self.opacityInterp = StatArray.StatArray(model.par.posterior.y.nCells)

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
        self.currentDataPoint = dataPoint
        self.bestDataPoint = dataPoint

        self.currentModel = model
        self.bestModel = model


        self.verbose = verbose
        if verbose:
            self.allRelErr = StatArray.StatArray([self.nSystems, self.nMC], name='$\epsilon_{Relative}x10^{2}$', units='%')
            self.allAddErr = StatArray.StatArray([self.nSystems, self.nMC], name='$\epsilon_{Additive}$', units=dataPoint.d.units)
            self.allZ = StatArray.StatArray(self.nMC, name='Height', units='m')
            self.posterior = StatArray.StatArray(self.nMC, name='log(posterior)')
            self.posteriorComponents = StatArray.StatArray([9, self.nMC], 'Components of the posterior')


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
            model.updatePosteriors(clipRatio)
            # self.kHist.update(model.nCells[0])
            # self.DzHist.update(dataPoint.z[0])

            # Update the height posterior
            dataPoint.z.updatePosterior()
            dataPoint.relErr.updatePosterior()
            
            dataPoint.addErr.updatePosterior()
            # for j in range(self.nSystems):
            #     self.relErr[j].update(dataPoint.relErr[j])
            #     self.addErr[j].update(dataPoint.addErr[j])

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

        self.currentDataPoint = dataPoint  # Reference
        self.bestDataPoint = bestDataPoint # Reference

        self.currentModel = model
        self.bestModel = bestModel # Reference


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

        ax = self.rate.plot(self.ratex, i=np.s_[:np.int64(self.i / 1000)], marker=m, markeredgecolor=mec, linestyle=ls, **kwargs)
        cP.xlabel('Iteration #')
        cP.ylabel('% Acceptance')
        cP.title('Acceptance rate')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


    def _plotMisfitVsIteration(self, **kwargs):
        """ Plot the data misfit against iteration. """

        m = kwargs.pop('marker', '.')
        ms = kwargs.pop('markersize', 2)
        a = kwargs.pop('alpha', 0.7)
        ls = kwargs.pop('linestyle', 'none')
        c = kwargs.pop('color', 'k')
        lw = kwargs.pop('linewidth', 3)
        
        ax = self.PhiDs.plot(self.iRange, i=np.s_[:self.i], marker=m, alpha=a, markersize=ms, linestyle=ls, color=c, **kwargs)
        plt.ylabel('Data Misfit')
        dum = self.multiplier * len(self.bestDataPoint.iActive)
        plt.axhline(dum, color='#C92641', linestyle='dashed', linewidth=lw)
        if (self.burnedIn):
            plt.axvline(self.iBurn, color='#C92641', linestyle='dashed', linewidth=lw)
        plt.yscale('log')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


    def _plotObservedPredictedData(self, **kwargs):
        """ Plot the observed and predicted data """

        self.bestDataPoint.plot(**kwargs)

        c = kwargs.pop('color', cP.wellSeparated[3])
        self.bestDataPoint.plotPredicted(color=c, **kwargs)


    def _plotNumberOfLayersPosterior(self, **kwargs):
        """ Plot the histogram of the number of layers """

        ax = self.currentModel.nCells.posterior.plot(**kwargs)
        plt.axvline(self.bestModel.nCells, color=cP.wellSeparated[3], linestyle='dashed', linewidth=3)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    def _plotHeightPosterior(self, **kwargs):
        """ Plot the histogram of the height """
        # self.DzHist.plot(**kwargs)
        ax = self.currentDataPoint.z.posterior.plot(**kwargs)
        plt.axvline(self.bestDataPoint.z, color=cP.wellSeparated[3], linestyle='dashed', linewidth=3)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    
    def _plotRelativeErrorPosterior(self, axes, **kwargs):
        """ Plots the histogram of the relative errors """
        self.currentDataPoint.relErr.plotPosteriors(axes=axes, **kwargs)
        plt.locator_params(axis='x', nbins=4)
        for i, a in enumerate(axes):
            plt.sca(a)
            plt.axvline(self.bestDataPoint.relErr[i], color=cP.wellSeparated[3], linestyle='dashed', linewidth=3)
            a.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        


    def _plotAdditiveErrorPosterior(self, axes, **kwargs):
        """ Plot the histogram of the additive errors """
        self.currentDataPoint.addErr.plotPosteriors(axes=axes, **kwargs)
        plt.locator_params(axis='x', nbins=4)

        if self.currentDataPoint.addErr.nPosteriors == 1:
            p = self.currentDataPoint.addErr.posterior
        else:
            p = self.currentDataPoint.addErr.posterior[0]

        log = p.log
        if (self.bestDataPoint.addErr[0] > p.bins[-1]):
            log = 10 
        
        loc, dum = cF._log(self.bestDataPoint.addErr, log=log)
        for i, a in enumerate(axes):
            plt.sca(a)
            plt.axvline(loc[i], color=cP.wellSeparated[3], linestyle='dashed', linewidth=3)
            a.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    def _plotLayerDepthPosterior(self, **kwargs):
        """ Plot the histogram of layer interface depths """

        r = kwargs.pop('rotate', True)
        fY = kwargs.pop('flipY', True)
        tr = kwargs.pop('trim', False)

        ax = self.currentModel.depth.posterior.plot(rotate=r, flipY=fY, trim=tr, **kwargs)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    
    def _plotHitmapPosterior(self, reciprocateX=False, confidenceInterval = 95.0, opacityPercentage = 67.0, **kwargs):
        """ Plot the hitmap posterior of conductivity with depth """

        # Get the mean and 95% confidence intervals
        hm = self.currentModel.par.posterior
        (sigMed, sigLow, sigHigh) = hm.confidenceIntervals(confidenceInterval)

        if (reciprocateX):
            x = 1.0 / hm.x.cellCentres
            sl = 1.0 / sigLow
            sh = 1.0 / sigHigh
            xlabel = 'Resistivity ($\Omega m$)'
        else:
            x = hm.x.cellCentres
            sl = sigLow
            sh = sigHigh
            xlabel = 'Conductivity ($Sm^{-1}$)'

        hm.counts.pcolor(x=x, y=hm.y.cellEdges, cmap=mpl.cm.Greys, **kwargs)
        plt.plot(sl, hm.y.cellCentres, color='#5046C8', linestyle='dashed', linewidth=2, alpha=0.6)
        plt.plot(sh, hm.y.cellCentres, color='#5046C8', linestyle='dashed', linewidth=2, alpha=0.6)
        cP.xlabel(xlabel)

        # Plot the DOI cutoff based on percentage variance
        self.doi = hm.getOpacityLevel(opacityPercentage)
        plt.axhline(self.doi, color='#5046C8', linestyle='dashed', linewidth=3)

        # Plot the best model
        self.bestModel.plot(flipY=False, reciprocateX=reciprocateX, noLabels=True)

        # Set parameter limits on the hitmap
        if self.limits is None:
            plt.axis([x.min(), x.max(), hm.y.cellEdges[0], hm.y.cellEdges[-1]])
        else:
            plt.axis([self.limits[0], self.limits[1], hm.y.cellEdges[0], hm.y.cellEdges[-1]])            

        ax = plt.gca()
        lim = ax.get_ylim()
        if (lim[1] > lim[0]):
            ax.set_ylim(lim[::-1])
        cP.ylabel(hm.yBins.getNameUnits())
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
            plt.cla()
            self._plotObservedPredictedData()

            if (self.burnedIn):

                # Histogram of the data point elevation
                plt.sca(self.ax[2])
                plt.cla()
                self._plotHeightPosterior()


                # Update the histogram of the number of layers
                plt.sca(self.ax[3])
                plt.cla()
                self._plotNumberOfLayersPosterior()
                self.ax[3].xaxis.set_major_locator(MaxNLocator(integer=True))

                

                # Update the model plot
                plt.sca(self.ax[5])
                plt.cla()
                self._plotHitmapPosterior(reciprocateX=self.reciprocateParameter, noColorbar=True)


                j = 5
                relativeAxes = []
                additiveAxes = []
                # Get the axes for the relative and additive errors
                for i in range(self.nSystems):
                    # Update the histogram of relative data errors
                    relativeAxes.append(self.ax[j+1])
                    additiveAxes.append(self.ax[j+2])
                    j += 2

                self._plotRelativeErrorPosterior(axes=relativeAxes)
                self._plotAdditiveErrorPosterior(axes=additiveAxes)

                # Update the layer depth histogram
                plt.sca(self.ax[(2 * self.nSystems) + 6])
                plt.cla()
                self._plotLayerDepthPosterior()

                

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
                self.posterior.plot(self.iRange, i=np.s_[:self.i - self.iBurn + 1], c='k')
                plt.subplot(414)
                self.iBestV.plot(self.iRange, i=np.s_[:self.i - self.iBurn + 1], c='k')


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
        # return self.fig



    def saveToLines(self, h5obj, fiducial):
        """ Save the results to a HDF5 object for a line """
        self.clk.restart()
        self.toHdf(h5obj, str(fiducial))


    def save(self, outdir, fiducial):
        """ Save the results to their own HDF5 file """
        with h5py.File(join(outdir,str(fiducial)+'.h5'),'w') as f:
            self.toHdf(f, str(fiducial))


    def toPNG(self, directory, fiducial, dpi=300):
       """ save a png of the results """
       fig = plt.figure(0)
       fig.set_size_inches(19, 11)
       figName = join(directory, '{}.png'.format(fiducial))
       plt.savefig(figName, dpi=dpi)

       if (self.verbose):
           fig = plt.figure(99)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(fiducial) + '_rap.png')
           plt.savefig(figName, dpi=dpi)

           fig = plt.figure(100)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(fiducial) + '_xp.png')
           plt.savefig(figName, dpi=dpi)

           fig = plt.figure(101)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(fiducial) + '_stack.png')
           plt.savefig(figName, dpi=dpi)

#     def hdfName(self):
#         """ Reprodicibility procedure """
#         return('Results()')


#     def createHdf(self, parent, myName):
#         """ Create the hdf group metadata in file
#         parent: HDF object to create a group inside
#         myName: Name of the group
#         """
#         assert isinstance(myName,str), 'myName must be a string'
#         # create a new group inside h5obj
#         try:
#             grp = parent.create_group(myName)
#         except:
#             # Assume that the file already has the template and return
#             return
#         grp.attrs["repr"] = self.hdfName()

#         grp.create_dataset('i', (1,), dtype=self.i.dtype)
#         grp.create_dataset('iplot', (1,), dtype=self.iPlot.dtype)
#         grp.create_dataset('plotme', (1,), dtype=type(self.plotMe))
#         if not self.limits is None:
#             grp.create_dataset('limits', (2,), dtype=self.limits.dtype)
#         grp.create_dataset('dispres', (1,), dtype=type(self.reciprocateParameter))
#         grp.create_dataset('sx', (1,), dtype=self.sx.dtype)
#         grp.create_dataset('sy', (1,), dtype=self.sy.dtype)
#         grp.create_dataset('nmc', (1,), dtype=self.nMC.dtype)
#         grp.create_dataset('nsystems', (1,), dtype=self.nSystems.dtype)
#         grp.create_dataset('iburn', (1,), dtype=self.iBurn.dtype)
#         grp.create_dataset('burnedin', (1,), dtype=type(self.burnedIn))
#         grp.create_dataset('doi', (1,), dtype=self.doi.dtype)
#         grp.create_dataset('multiplier', (1,), dtype=self.multiplier.dtype)
#         grp.create_dataset('invtime', (1,), dtype=float)
#         grp.create_dataset('savetime', (1,), dtype=float)

#         nz=self.Hitmap.y.nCells
#         grp.create_dataset('meaninterp', (nz,), dtype=np.float64)
#         grp.create_dataset('bestinterp', (nz,), dtype=np.float64)
# #        grp.create_dataset('opacityinterp', (nz,), dtype=np.float64)

#         self.rate.createHdf(grp,'rate')
#         self.ratex.createHdf(grp,'ratex')
#         self.PhiDs.createHdf(grp,'phids')
#         # self.kHist.createHdf(grp, 'khist')
#         # self.currentDataPoint.z.posterior.createHdf(grp, 'dzhist')
#         self.MzHist.createHdf(grp, 'mzhist')
#         for i in range(self.nSystems):
#             self.relErr[i].createHdf(grp, 'relerr' + str(i))
#         for i in range(self.nSystems):
#             self.addErr[i].createHdf(grp, 'adderr' + str(i))

#         self.Hitmap.createHdf(grp,'hitmap')
#         self.currentDataPoint.createHdf(grp, 'currentdatapoint')
#         self.bestDataPoint.z._posterior = None
#         self.bestDataPoint.relErr._posterior = None
#         self.bestDataPoint.addErr._posterior = None
#         self.bestDataPoint.createHdf(grp, 'bestd')

#         tmp = self.currentModel.pad(self.currentModel.maxLayers)
#         tmp.createHdf(grp, 'currentmodel')

#         tmp = self.bestModel.pad(self.bestModel.maxLayers)
#         tmp.nCells._posterior = None
#         tmp.createHdf(grp, 'bestmodel')


#     def writeHdf(self, parent, myName, index=None):
#         """ Write the StatArray to an HDF object
#         parent: Upper hdf file or group
#         myName: object hdf name. Assumes createHdf has already been called
#         create: optionally create the data set as well before writing
#         index: optional numpy slice of where to place the arr in the hdf data object
#         """
#         assert isinstance(myName,str), 'myName must be a string'

#         grp = parent.get(myName)
#         writeNumpy(self.i, grp, 'i')
#         writeNumpy(self.iPlot, grp, 'iplot')
#         writeNumpy(self.plotMe, grp, 'plotme')
#         if not self.limits is None:
#             writeNumpy(self.limits, grp, 'limits')
#         writeNumpy(self.reciprocateParameter, grp, 'dispres')
#         writeNumpy(self.sx, grp, 'sx')
#         writeNumpy(self.sy, grp, 'sy')
#         writeNumpy(self.nMC, grp, 'nmc')
#         writeNumpy(self.nSystems, grp, 'nsystems')
#         writeNumpy(self.iBurn, grp, 'iburn')
#         writeNumpy(self.burnedIn, grp, 'burnedin')
#         self.doi = self.Hitmap.getOpacityLevel(67.0)
#         writeNumpy(self.doi, grp, 'doi')
#         writeNumpy(self.multiplier, grp, 'multiplier')
#         writeNumpy(self.invTime, grp, 'invtime')

#         self.rate.writeHdf(grp, 'rate')
#         self.ratex.writeHdf(grp, 'ratex')
#         self.PhiDs.writeHdf(grp, 'phids')
#         # self.kHist.writeHdf(grp, 'khist')
#         # self.currentDataPoint.z.posterior.writeHdf(grp, 'dzhist')
#         self.MzHist.writeHdf(grp, 'mzhist')
#          # Histograms for each system
#         for i in range(self.nSystems):
#             self.relErr[i].writeHdf(grp, 'relerr' + str(i))
#         for i in range(self.nSystems):
#             self.addErr[i].writeHdf(grp, 'adderr' + str(i))

#         self.Hitmap.writeHdf(grp,'hitmap')

#         mean = self.Hitmap.getMeanInterval()
#         best = self.bestModel.interp2depth(self.bestModel.par, self.Hitmap)
# #        opacity = self.Hitmap.getOpacity()
#         writeNumpy(mean, grp,'meaninterp')
#         writeNumpy(best, grp,'bestinterp')
# #        writeNumpy(opacity, grp,'opacityinterp')

#         self.clk.stop()
#         self.saveTime = np.float64(self.clk.timeinSeconds())
#         writeNumpy(self.saveTime, grp, 'savetime')
#         self.currentDataPoint.writeHdf(grp, 'currentdatapoint')

#         self.bestDataPoint.z._posterior = None
#         self.bestDataPoint.relErr._posterior = None
#         self.bestDataPoint.addErr._posterior = None
#         self.bestDataPoint.writeHdf(grp, 'bestd')

#         self.currentModel.writeHdf(grp, 'currentmodel')

#         self.bestModel.nCells._posterior = None
#         self.bestModel.writeHdf(grp, 'bestmodel')


#     def toHdf(self, h5obj, myName):
#         """ Write the object to a HDF file """
#         # Create a new group inside h5obj
#         try:
#             grp = h5obj.create_group(myName)
#         except:
#             del h5obj[myName]
#             grp = h5obj.create_group(myName)
#         grp.attrs["repr"] = self.hdfName()
#         # Begin by writing the parameters
# #        grp.create_dataset('id', data=self.ID)
#         grp.create_dataset('i', data=self.i)
#         grp.create_dataset('iplot', data=self.iPlot)
#         grp.create_dataset('plotme', data=self.plotMe)
#         if not self.limits is None:
#             grp.create_dataset('limits', data=self.limits)
#         grp.create_dataset('dispres', data=self.reciprocateParameter)
#         grp.create_dataset('sx', data=self.sx)
#         grp.create_dataset('sy', data=self.sy)
#         grp.create_dataset('nmc', data=self.nMC)
#         grp.create_dataset('nsystems', data=self.nSystems)
#         grp.create_dataset('iburn', data=self.iBurn)
#         grp.create_dataset('burnedin', data=self.burnedIn)
#         self.doi = self.Hitmap.getOpacityLevel(67.0)
#         grp.create_dataset('doi', data=self.doi)
#         # Large vector of phiD
#         grp.create_dataset('multiplier', data=self.multiplier)
#         # Rate of acceptance
#         self.rate.writeHdf(grp, 'rate')
#         # x Axis for acceptance rate
#         self.ratex.writeHdf(grp, 'ratex')
#         # Data Misfit
#         self.PhiDs.writeHdf(grp, 'phids')
#         # Histogram of # of Layers
#         # self.kHist.writeHdf(grp, 'khist')
#         # Histogram of Elevations
#         # self.currentDataPoint.z.posterior.writeHdf(grp, 'dzhist')
#         # Histogram of Layer depths
#         self.MzHist.writeHdf(grp, 'mzhist')
#         # Hit Maps
#         self.Hitmap.writeHdf(grp, 'hitmap')
#         # Write the current data
#         self.currentDataPoint.writeHdf(grp, 'currentdatapoint')
#         # Write the Best Data
#         self.bestDataPoint.z._posterior = None
#         self.bestDataPoint.relErr._posterior = None
#         self.bestDataPoint.addErr._posterior = None
#         self.bestDataPoint.writeHdf(grp, 'bestd')

#         self.currentModel.writeHdf(grp, 'currentmodel')
        
#         # Write the Best Model
#         self.bestModel.nCells._posterior = None
#         self.bestModel.writeHdf(grp, 'bestmodel')

#         # Interpolate the mean and best model to the discretized hitmap
#         mean = self.Hitmap.getMeanInterval()
#         best = self.bestModel.interp2depth(self.bestModel.par, self.Hitmap)
# #        opacity = self.Hitmap.getOpacity()

#         grp.create_dataset('meaninterp', data=mean)
#         grp.create_dataset('bestinterp', data=best)
# #        grp.create_dataset('opacityinterp', data=opacity)


#         # Histograms for each system
#         for i in range(self.nSystems):
#             self.relErr[i].toHdf(grp, 'relerr' + str(i))
#         for i in range(self.nSystems):
#             self.addErr[i].toHdf(grp, 'adderr' + str(i))

#         grp.create_dataset('invtime', data=self.invTime)
#         self.clk.stop()
#         self.saveTime = self.clk.timeinSeconds()
#         grp.create_dataset('savetime', data=self.saveTime)


    def fromHdf(self, hdfFile, index, fid, sysPath):
    

        s = np.s_[index, :]

        self.fiducial = np.float64(fid)

        self.iPlot = np.array(hdfFile.get('iplot'))
        self.plotMe = np.array(hdfFile.get('plotme'))

        tmp = hdfFile.get('limits')
        self.limits = None if tmp is None else np.array(tmp)
        self.reciprocateParameter = np.array(hdfFile.get('reciprocateParameter'))
        self.nMC = np.array(hdfFile.get('nmc'))
        self.nSystems = np.array(hdfFile.get('nsystems'))
        self.ratex = hdfRead.readKeyFromFile(hdfFile,'','/','ratex')

        self.i = hdfRead.readKeyFromFile(hdfFile,'','/','i', index=index)
        self.iBurn = hdfRead.readKeyFromFile(hdfFile,'','/','iburn', index=index)
        self.burnedIn = hdfRead.readKeyFromFile(hdfFile,'','/','burnedin', index=index)
        self.doi = hdfRead.readKeyFromFile(hdfFile,'','/','doi', index=index)
        self.multiplier = hdfRead.readKeyFromFile(hdfFile,'','/','multiplier', index=index)
        self.rate = hdfRead.readKeyFromFile(hdfFile,'','/','rate', index=s)
        self.PhiDs = hdfRead.readKeyFromFile(hdfFile,'','/','phids', index=s)

        self.bestDataPoint = hdfRead.readKeyFromFile(hdfFile,'','/','bestd', index=index, sysPath=sysPath)
        try:
            self.currentDataPoint = hdfRead.readKeyFromFile(hdfFile,'','/','currentdatapoint', index=index, sysPath=sysPath)
        except:
            self.currentDataPoint = self.bestDataPoint
            p = hdfRead.readKeyFromFile(hdfFile,'','/','dzhist', index=index)
            self.currentDataPoint.z.setPosterior(p)
        
        
        try:
            self.currentModel = hdfRead.readKeyFromFile(hdfFile,'','/','currentmodel', index=index)
            self.Hitmap = self.currentModel.par.posterior
            self.currentModel.maxDepth = np.log(self.Hitmap.y.cellCentres[-1])            
        except:
            self.Hitmap = hdfRead.readKeyFromFile(hdfFile,'','/','hitmap', index=index)


        self.bestModel = hdfRead.readKeyFromFile(hdfFile,'','/','bestmodel', index=index)
        self.bestModel.maxDepth = np.log(self.Hitmap.y.cellCentres[-1])

        

        # self.kHist = hdfRead.readKeyFromFile(hdfFile,'','/','khist', index=i)
        # self.DzHist = hdfRead.readKeyFromFile(hdfFile,'','/','dzhist', index=i)
        # self.MzHist = hdfRead.readKeyFromFile(hdfFile,'','/','mzhist', index=i)

        # Hack to recentre the altitude histogram go this datapoints altitude
        # self.DzHist._cellEdges -= (self.DzHist.bins[int(self.DzHist.bins.size/2)-1] - self.bestD.z[0])
        # self.DzHist._cellCentres = self.DzHist._cellEdges[:-1] + 0.5 * np.abs(np.diff(self.DzHist._cellEdges))

        # self.relErr = []
        # self.addErr = []
        # for j in range(self.nSystems):
        #     self.relErself.append(hdfRead.readKeyFromFile(hdfFile,'','/','relerr'+str(j), index=i))
        #     self.addErself.append(hdfRead.readKeyFromFile(hdfFile,'','/','adderr'+str(j), index=i))


        self.invTime=np.array(hdfFile.get('invtime')[index])
        self.saveTime=np.array(hdfFile.get('savetime')[index])

        # Initialize a list of iteration number
        self.iRange = StatArray.StatArray(np.arange(2 * self.nMC), name="Iteration #", dtype=np.int64)

        self.verbose = False

        return self




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
