""" @Inference1D
Class to store inversion results. Contains plotting and writing to file procedures
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

class Inference1D(myObject):
    """Define the results for the Bayesian MCMC Inversion.

    Contains histograms and inversion related variables that can be updated as the Bayesian inversion progresses.

    Inference1D(saveMe, plotMe, savePNG, dataPoint, model, ID, \*\*kwargs)

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

        self.fig = None
        # Return none if important parameters are not used (used for hdf 5)
        if all(x1 is None for x1 in [dataPoint, model]):
            return

        assert self.plotMe or self.saveMe, Exception('You have chosen to neither view or save the inversion results!')

        nMarkovChains = kwargs.pop('nMarkovChains', 100000)
        plotEvery = kwargs.pop('plotEvery', nMarkovChains / 20)
        parameterDisplayLimits = kwargs.pop('parameterDisplayLimits', [0.0, 1.0])
        reciprocateParameter = kwargs.pop('reciprocateParameters', False)

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
        self.iBest = np.int64(0)
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
        self.iBest = np.int32(0)
        self.iBestV = StatArray.StatArray(2*self.nMC, name='Iteration of best model')

        self.iz = np.arange(model.par.posterior.y.nCells)

        # Initialize the doi
        # self.doi = model.par.posterior.yBinCentres[0]

        self.meanInterp = StatArray.StatArray(model.par.posterior.y.nCells)
        self.bestInterp = StatArray.StatArray(model.par.posterior.y.nCells)
        self.opacityInterp = StatArray.StatArray(model.par.posterior.y.nCells)

        # Set a tag to catch data points that are not minimizing
        self.zeroCount = 0

        self.verbose = verbose

        # Initialize times in seconds
        self.invTime = np.float64(0.0)
        self.saveTime = np.float64(0.0)

        # Initialize the best data, current data and best model
        self.currentDataPoint = dataPoint
        self.bestDataPoint = dataPoint

        self.currentModel = model
        self.bestModel = model

        if verbose:
            n = np.int(1.2*self.nMC)
            self.allRelErr = StatArray.StatArray(np.full([self.nSystems, n], np.nan), name='$\epsilon_{Relative}x10^{2}$', units='%')
            self.allAddErr = StatArray.StatArray(np.full([self.nSystems, n], np.nan), name='$\epsilon_{Additive}$', units=dataPoint.data.units)
            self.allZ = StatArray.StatArray(np.full(n, np.nan), name='Height', units='m')
            self.allK = StatArray.StatArray(np.full(n, np.nan), name='Number of Layers')
            self.posteriorComponents = StatArray.StatArray(np.full([8, n], np.nan), 'Components of the posterior')
            self.ratioComponents = StatArray.StatArray(np.full([7, n], np.nan), 'log(Ratio Components)')
            self.accepted = StatArray.StatArray(np.zeros(n, dtype=bool), name='Accepted')
            self.dimensionChange = StatArray.StatArray(np.zeros(n, dtype=bool), name='Dimensions were changed')


    @property
    def hitmap(self):
        return self.currentModel.par.posterior


    def doi(self, percentage = 67.0, log=None):
        return self.hitmap.getOpacityLevel(percentage, log=log)


    def update(self, i, model, dataPoint, iBest, bestDataPoint, bestModel, multiplier, PhiD, posterior, posteriorComponents, ratioComponents, accepted, dimensionChange, clipRatio):
        """Update the posteriors of the McMC algorithm. """
        self.i = np.int32(i)
        self.iBest = np.int32(iBest)
        self.PhiDs[self.i - 1] = PhiD.copy()  # Store the data misfit
        self.multiplier = np.float64(multiplier)

        if (self.burnedIn):  # We need to update some plotting options
            # Added the layer depths to a list, we histogram this list every
            # iPlot iterations
            model.updatePosteriors(clipRatio)

            # Update the height posterior
            dataPoint.updatePosteriors()

            if (self.verbose):
                iTmp = self.i - self.iBurn
                for j in range(self.nSystems):
                    self.allRelErr[j, iTmp] = dataPoint.relErr[j]
                    self.allAddErr[j, iTmp] = dataPoint.addErr[j]
                self.allZ[iTmp] = dataPoint.z[0]
                self.allK[iTmp] = model.nCells[0]
                self.posteriorComponents[:, iTmp] = posteriorComponents
                self.ratioComponents[:, iTmp] = ratioComponents
                self.accepted[iTmp] = accepted
                self.dimensionChange[iTmp] = dimensionChange


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


    def initFigure(self, fig = None):
        """ Initialize the plotting region """
        # Setup the figure region. The figure window is split into a 4x3
        # region. Columns are able to span multiple rows

        # plt.ion()

        if fig is None:
            self.fig = plt.figure(facecolor='white', figsize=(10,7))
        else:
            self.fig = plt.figure(fig.number)

        mngr = plt.get_current_fig_manager()
        try:
            mngr.window.setGeometry(0, 10, self.sx, self.sy)
        except:
            pass
        nCols = 3 * self.nSystems
        nRows = 12

        gs = gridspec.GridSpec(nRows, nCols)
        gs.update(wspace=0.3 * self.nSystems, hspace=6.0)
        self.ax = []

        self.ax.append(plt.subplot(gs[:3, :self.nSystems])) # Acceptance Rate 0
        self.ax.append(plt.subplot(gs[3:6, :self.nSystems])) # Data misfit vs iteration 1
        self.ax.append(plt.subplot(gs[6:9, :self.nSystems])) # Histogram of data point elevations 2
        self.ax.append(plt.subplot(gs[9:12, :self.nSystems])) # Histogram of # of layers 3
        self.ax.append(plt.subplot(gs[:6,self.nSystems:2 * self.nSystems])) # Data fit plot 4
        self.ax.append(plt.subplot(gs[6:12,self.nSystems:2 * self.nSystems])) # 1D layer plot 5
        self.ax.append(plt.subplot(gs[6:12, 2 * self.nSystems:])) # Histogram of layer depths 6
        # Histogram of data errors

        for i in range(self.nSystems):
            if not self.currentDataPoint.errorPosterior is None:
                self.ax.append(plt.subplot(gs[1:5, 2 * self.nSystems + i])) # 2D Histogram
            else:
                self.ax.append(plt.subplot(gs[:3,  2 * self.nSystems + i])) # Relative Errors
                self.ax.append(plt.subplot(gs[3:6, 2 * self.nSystems + i])) # Additive Errors

        for ax in self.ax:
            cP.pretty(ax)

        if self.verbose:
            self.verboseFigs = []
            self.verboseAxs = []

            # Posterior components
            fig = plt.figure(facecolor='white', figsize=(10,7))
            self.verboseFigs.append(fig)
            self.verboseAxs.append(fig.add_subplot(511))
            self.verboseAxs.append(fig.add_subplot(512))
            self.verboseAxs.append(fig.add_subplot(513))

            fig = plt.figure(facecolor='white', figsize=(10,7))
            self.verboseFigs.append(fig)
            for i in range(8):
                self.verboseAxs.append(fig.add_subplot(8, 1, i+1))

            # Cross Plots
            fig = plt.figure(facecolor='white', figsize=(10,7))
            self.verboseFigs.append(fig)
            for i in range(4):
                self.verboseAxs.append(fig.add_subplot(1, 4, i+1))

            # ratios vs iteration number
            fig = plt.figure(facecolor='white', figsize=(10,7))
            self.verboseFigs.append(fig)
            for i in range(5):
                self.verboseAxs.append(fig.add_subplot(5, 1, i+1))

            for ax in self.verboseAxs:
                cP.pretty(ax)


        if self.plotMe:
            plt.show(block=False)
        # plt.draw()


    def plot(self, title="", increment=None):
        """ Updates the figures for MCMC Inversion """
        # Plots that change with every iteration
        if self.i == 0:
            return

        if (self.fig is None):
            self.initFigure()

        plt.figure(self.fig.number)

        plot = True
        if not increment is None:
            if (np.mod(self.i, increment) != 0):
                plot = False

        if plot:
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

            # Update the model plot
            plt.sca(self.ax[5])
            plt.cla()
            self._plotParameterPosterior(reciprocateX=self.reciprocateParameter, noColorbar=True)

            if (self.burnedIn):
                # Histogram of the data point elevation
                plt.sca(self.ax[2])
                plt.cla()
                self._plotHeightPosterior(normalize=True)


                # Update the histogram of the number of layers
                plt.sca(self.ax[3])
                plt.cla()
                self._plotNumberOfLayersPosterior(normalize=True)
                self.ax[3].xaxis.set_major_locator(MaxNLocator(integer=True))

                # Update the layer depth histogram
                plt.sca(self.ax[6])
                plt.cla()
                self._plotLayerDepthPosterior(normalize=True)

                j = 7
                self._plotErrorPosterior(axes=self.ax[7:], normalize=True)


            cP.suptitle(title)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


            if self.verbose & self.burnedIn:

                plt.figure(self.verboseFigs[0].number)
                plt.sca(self.verboseAxs[0])
                plt.cla()
                self.allRelErr[0, :].plot(self.iRange, i=np.s_[:self.i], c='k')
                plt.sca(self.verboseAxs[1])
                plt.cla()
                self.allAddErr[0, :].plot(self.iRange, i=np.s_[:self.i], axis=1, c='k')
                plt.sca(self.verboseAxs[2])
                plt.cla()
                self.allZ.plot(x=self.iRange, i=np.s_[:self.i], marker='o', linestyle='none', markersize=2, alpha=0.3, markeredgewidth=1)


                # Posterior components plot Figure 1
                labels=['nCells','depth','parameter','gradient','relative','additive','height','calibration']
                plt.figure(self.verboseFigs[1].number)
                for i in range(8):
                    plt.sca(self.verboseAxs[3 + i])
                    plt.cla()
                    self.posteriorComponents[i, :].plot(linewidth=0.5)
                    plt.ylabel('')
                    plt.title(labels[i])
                    if labels[i] == 'gradient':
                        plt.ylim([-30.0, 1.0])


                ira = self.iRange[:np.int(1.2*self.nMC)][self.accepted]
                irna = self.iRange[:np.int(1.2*self.nMC)][~self.accepted]

                plt.figure(self.verboseFigs[3].number)
                # Number of layers vs iteration
                plt.sca(self.verboseAxs[15])
                plt.cla()
                self.allK[~self.accepted].plot(x = irna, marker='o', markersize=1,  linestyle='None', alpha=0.3, color='k')
                self.allK[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
                plt.title('black = rejected')


                plt.figure(self.verboseFigs[2].number)
                # Cross plot of current vs candidate prior
                plt.sca(self.verboseAxs[11])
                plt.cla()
                x = StatArray.StatArray(self.ratioComponents[0, :], 'Candidate Prior')
                y = StatArray.StatArray(self.ratioComponents[1, :], 'Current Prior')

                x[x == -np.inf] = np.nan
                y[y == -np.inf] = np.nan
                x[~self.accepted].plot(x = y[~self.accepted], linestyle='', marker='.', color='k', alpha=0.3)
                x[self.accepted].plot(x = y[self.accepted], linestyle='', marker='.', alpha=0.3)
                # v1 = np.maximum(np.minimum(np.nanmin(x), np.nanmin(y)), -20.0)
                v2 = np.maximum(np.nanmax(x), np.nanmax(y))
                v1 = v2 - 25.0
                plt.xlim([v1, v2])
                plt.ylim([v1, v2])
                plt.plot([v1,v2], [v1,v2])

                # Prior ratio vs iteration
                plt.figure(self.verboseFigs[3].number)
                plt.sca(self.verboseAxs[16])
                plt.cla()
                r = x - y
                r[~self.accepted].plot(x = irna, marker='o', markersize=1, linestyle='None', alpha=0.3, color='k')
                r[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
                plt.ylim([v1, 5.0])
                cP.ylabel('Prior Ratio')



                plt.figure(self.verboseFigs[2].number)
                # Cross plot of the likelihood ratios
                plt.sca(self.verboseAxs[12])
                plt.cla()
                x = StatArray.StatArray(self.ratioComponents[2, :], 'Candidate Likelihood')
                y = StatArray.StatArray(self.ratioComponents[3, :], 'Current Likelihood')
                x[~self.accepted].plot(x = y[~self.accepted], linestyle='', marker='.', color='k', alpha=0.3)
                x[self.accepted].plot(x = y[self.accepted], linestyle='', marker='.', alpha=0.3)

                v2 = np.maximum(np.nanmax(x), np.nanmax(y)) + 5.0
                v1 = v2 - 200.0
                # v1 = -100.0
                # v2 = -55.0
                plt.xlim([v1, v2])
                plt.ylim([v1, v2])
                plt.plot([v1, v2], [v1, v2])
                plt.title('black = rejected')

                plt.figure(self.verboseFigs[3].number)
                # Likelihood ratio vs iteration
                plt.sca(self.verboseAxs[17])
                plt.cla()
                r = x - y
                r[~self.accepted].plot(x = irna, marker='o', markersize=1, linestyle='None', alpha=0.3, color='k')
                r[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
                cP.ylabel('Likelihood Ratio')
                plt.ylim([-20.0, 20.0])

                plt.figure(self.verboseFigs[2].number)
                # Cross plot of the proposal ratios
                plt.sca(self.verboseAxs[13])
                plt.cla()
                y = StatArray.StatArray(self.ratioComponents[4, :], 'Current Proposal')
                x = StatArray.StatArray(self.ratioComponents[5, :], 'Candidate Proposal')
                x[~self.accepted].plot(x = y[~self.accepted], linestyle='', marker='.', color='k', alpha=0.3)
                x[self.accepted].plot(x = y[self.accepted], linestyle='', marker='.', alpha=0.3)
                # v1 = np.maximum(np.minimum(np.nanmin(x), np.nanmin(y)), -200.0)
                v2 = np.maximum(np.nanmax(x), np.nanmax(y)) + 10.0
                v1 = v2 - 60.0
                v1 = -20.0
                v2 = 20.0
                # plt.plot([v1,v2], [v1,v2])
                plt.xlim([v1, v2])
                plt.ylim([v1, v2])


                plt.figure(self.verboseFigs[2].number)
                # Cross plot of the proposal ratios coloured by a change in dimension
                plt.sca(self.verboseAxs[14])
                plt.cla()
                y = StatArray.StatArray(self.ratioComponents[4, :], 'Current Proposal')
                x = StatArray.StatArray(self.ratioComponents[5, :], 'Candidate Proposal')
                x[~self.dimensionChange].plot(x = y[~self.dimensionChange], linestyle='', marker='.', color='k', alpha=0.3)
                x[self.dimensionChange].plot(x = y[self.dimensionChange], linestyle='', marker='.', alpha=0.3)
                # v1 = np.maximum(np.minimum(np.nanmin(x), np.nanmin(y)), -200.0)
                # v2 = np.maximum(np.nanmax(x), np.nanmax(y)) + 10.0
                # v1 = v2 - 60.0

                # plt.plot([v1,v2], [v1,v2])
                plt.xlim([v1, v2])
                plt.ylim([v1, v2])
                plt.title('black = no dimension change')

                plt.figure(self.verboseFigs[3].number)
                # Proposal ratio vs iteration
                plt.sca(self.verboseAxs[18])
                plt.cla()
                r = x - y
                r[~self.accepted].plot(x = irna, marker='o', markersize=1, linestyle='None', alpha=0.3, color='k')
                r[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
                cP.ylabel('Proposal Ratio')
                plt.ylim([v1, v2])

                # Acceptance ratio vs iteration
                plt.sca(self.verboseAxs[19])
                plt.cla()
                x = StatArray.StatArray(self.ratioComponents[6, :], 'Acceptance Ratio')
                x[~self.accepted].plot(x = irna, marker='o', markersize=1, linestyle='None', alpha=0.3, color='k')
                x[self.accepted].plot(x = ira, marker='o', markersize=1, linestyle='None', alpha=0.3)
                plt.ylim([-20.0, 20.0])


                for fig in self.verboseFigs:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            cP.pause(1e-9)


    def _plotAcceptanceVsIteration(self, **kwargs):
        """ Plots the acceptance percentage against iteration. """

        kwargs['marker'] = kwargs.pop('marker', 'o')
        kwargs['alpha'] = kwargs.pop('alpha', 0.7)
        kwargs['linestyle'] = kwargs.pop('linestyle', 'none')
        kwargs['markeredgecolor'] = kwargs.pop('markeredgecolor', 'k')

        ax = self.rate.plot(self.ratex, i=np.s_[:np.int64(self.i / 1000)], **kwargs)
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
        dum = self.multiplier * self.currentDataPoint.active.size
        plt.axhline(dum, color='#C92641', linestyle='dashed', linewidth=lw)
        if (self.burnedIn):
            plt.axvline(self.iBurn, color='#C92641', linestyle='dashed', linewidth=lw)
            try:
                plt.axvline(self.iBest, color=cP.wellSeparated[3])
            except:
                pass
        plt.yscale('log')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlim([0, self.iRange[self.i]])


    def _plotObservedPredictedData(self, **kwargs):
        """ Plot the observed and predicted data """
        self.currentDataPoint.plot(**kwargs)
        if self.burnedIn:
            self.bestDataPoint.plotPredicted(color=cP.wellSeparated[3], **kwargs)
        else:
            self.currentDataPoint.plotPredicted(color='g', **kwargs)


    def _plotNumberOfLayersPosterior(self, **kwargs):
        """ Plot the histogram of the number of layers """

        ax = self.currentModel.nCells.posterior.plot(**kwargs)
        plt.axvline(self.bestModel.nCells, color=cP.wellSeparated[3], linewidth=1)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    def _plotHeightPosterior(self, **kwargs):
        """ Plot the histogram of the height """
        if self.currentDataPoint.z.hasPosterior:
            ax = self.currentDataPoint.z.posterior.plot(**kwargs)
            if self.burnedIn:
                plt.axvline(self.bestDataPoint.z, color=cP.wellSeparated[3], linewidth=1)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        else:
            plt.text(0.1, 0.5, "{} = {} {}".format(self.currentDataPoint.z.name, self.currentDataPoint.z[0], self.currentDataPoint.z.units))


    def _plotErrorPosterior(self, axes, **kwargs):

        if not self.currentDataPoint.errorPosterior is None:
            log = self.currentDataPoint.errorPosterior[0].x.log
            loc, _ = cF._log(self.bestDataPoint.addErr, log=log)

            for i in range(self.nSystems):
                ax = axes[i]
                ax.cla()
                self.currentDataPoint.errorPosterior[i].plot(ax=ax, cmap='gray_r', noColorbar=True, **kwargs)
                plt.sca(ax)
                plt.axhline(self.bestDataPoint.relErr[i], color=cP.wellSeparated[3], linewidth=1)
                plt.axvline(loc[i], color=cP.wellSeparated[3], linewidth=1)

                if i > 0:
                    plt.ylabel('')

            return

        if self.currentDataPoint.relErr.hasPosterior and self.currentDataPoint.addErr.hasPosterior:
            assert len(axes) == 2 * self.nSystems, ValueError("Need {} axes to plot error posteriors".format(2 * self.nSystems))
            self._plotRelativeErrorPosterior(axes[::2], **kwargs)
            plt.title('')
            self._plotAdditiveErrorPosterior(axes[1::2], **kwargs)
            plt.title('')
            return

        if self.currentDataPoint.relErr.hasPosterior:
            self._plotRelativeErrorPosterior(axes, **kwargs)
            return

        if self.currentDataPoint.addErr.hasPosterior:
            self._plotAdditiveErrorPosterior(axes, **kwargs)
            return

        for i in range(self.nSystems):
                ax = axes[i]
                ax.cla()
                ax.text(0.1, 0.5,
                        " {} = {} {} \n {} = {} {}".format(
                        self.bestDataPoint.relErr.name, self.bestDataPoint.relErr[i], self.bestDataPoint.relErr.units,
                        self.bestDataPoint.addErr.name, self.bestDataPoint.addErr[i], self.bestDataPoint.addErr.units))


    def _plotRelativeErrorPosterior(self, axes, **kwargs):
        """ Plots the histogram of the relative errors """

        if not isinstance(axes, list):
            axes = [axes]

        self.currentDataPoint.relErr.plotPosteriors(axes, **kwargs)

        if self.burnedIn:
            plt.locator_params(axis='x', nbins=4)
            for i, a in enumerate(axes):
                plt.sca(a)
                plt.axvline(self.bestDataPoint.relErr[i], color=cP.wellSeparated[3], linewidth=1)
                a.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                cP.title("Best {} was {:.3e} {}".format(self.bestDataPoint.addErr.name, self.bestDataPoint.addErr[i], self.bestDataPoint.addErr.units))


    def _plotAdditiveErrorPosterior(self, axes, **kwargs):
        """ Plot the histogram of the additive errors """
        if not isinstance(axes, list):
            axes = [axes]

        self.currentDataPoint.addErr.plotPosteriors(axes=axes, **kwargs)
        plt.locator_params(axis='x', nbins=4)

        if self.burnedIn:
            log = np.atleast_1d(self.currentDataPoint.addErr.posterior)[0].log
            loc, _ = cF._log(self.bestDataPoint.addErr, log=log)
            for i, a in enumerate(axes):
                plt.sca(a)
                plt.axvline(loc[i], color=cP.wellSeparated[3], linewidth=1)
                a.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                cP.title("Best {} was {:.3e} {}".format(self.bestDataPoint.relErr.name, self.bestDataPoint.relErr[i], self.bestDataPoint.relErr.units))


    def _plotLayerDepthPosterior(self, **kwargs):
        """ Plot the histogram of layer interface depths """

        kwargs['rotate'] = kwargs.pop('rotate', True)
        kwargs['flipY'] = kwargs.pop('flipY', True)
        kwargs['trim'] = kwargs.pop('trim', False)
        kwargs['normalize'] = kwargs.pop('normalize', True)


        ax = self.currentModel.depth.posterior.plot(**kwargs)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        return ax


    def _plotParameterPosterior(self, reciprocateX=False, credibleInterval = 95.0, opacityPercentage = 67.0, overlayModel=True, **kwargs):
        """ Plot the hitmap posterior of conductivity with depth """


        xlim = kwargs.pop('xlim', None)
        ylim = kwargs.pop('ylim', None)

        hm = self.currentModel.par.posterior

        if (reciprocateX):
            x = 1.0 / hm.x.cellCentres
            xlabel = 'Resistivity ($\Omega m$)'
        else:
            x = hm.x.cellCentres
            xlabel = 'Conductivity ($Sm^{-1}$)'


        if self.burnedIn:
            # Get the mean and 95% credible intervals
            (sigMed, sigLow, sigHigh) = hm.credibleIntervals(credibleInterval)

            if (reciprocateX):
                sl = 1.0 / sigLow
                sh = 1.0 / sigHigh
            else:
                sl = sigLow
                sh = sigHigh

            hm.counts.pcolor(x=x, y=hm.y.cellEdges, cmap=mpl.cm.Greys, **kwargs)

            CI_kw = {'color':'#5046C8', 'linestyle':'dashed', 'linewidth':1, 'alpha':0.6}
            plt.plot(sl, hm.y.cellCentres, **CI_kw)
            plt.plot(sh, hm.y.cellCentres, **CI_kw)

            # Plot the DOI cutoff based on percentage variance
            plt.axhline(self.doi(log=10), **CI_kw)

        # Plot the best model
        if overlayModel:
            if self.burnedIn:
                self.bestModel.plot(flipY=False, reciprocateX=reciprocateX, noLabels=True, linewidth=1, color=cP.wellSeparated[3])
            else:
                self.currentModel.plot(flipY=False, reciprocateX=reciprocateX, noLabels=True, linewidth=1, color='g')

        # Set parameter limits on the hitmap
        if xlim is None:
            plt.axis([x.min(), x.max(), hm.y.cellEdges[0], hm.y.cellEdges[-1]])
        else:
            assert np.size(xlim) == 2, ValueError("xlim must have size 2")
            plt.axis([xlim[0], xlim[1], hm.y.cellEdges[0], hm.y.cellEdges[-1]])

        cP.xlabel(xlabel)
        cP.ylabel(hm.yBins.getNameUnits())

        ax = plt.gca()
        lim = ax.get_ylim()

        if not ylim is None:
            ax.set_ylim(ylim)
            lim = ax.get_ylim()

        if (lim[1] > lim[0]):
            ax.set_ylim(lim[::-1])
        plt.xscale('log')
        plt.margins(0.01)


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
       self.fig.set_size_inches(19, 11)
       figName = join(directory, '{}.png'.format(fiducial))
       self.fig.savefig(figName, dpi=dpi)

       if (self.verbose):
           fig = plt.figure(1)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(fiducial) + '_rap.png')
           plt.savefig(figName, dpi=dpi)

           fig = plt.figure(2)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(fiducial) + '_posterior_components.png')
           plt.savefig(figName, dpi=dpi)

           fig = plt.figure(3)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(fiducial) + '_ratio_crossplot.png')
           plt.savefig(figName, dpi=dpi)

           fig = plt.figure(4)
           fig.set_size_inches(19, 11)
           figName = join(directory,str(fiducial) + '_ratios_vs_iteration.png')
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


    def read(self, fileName, systemFilePath, fiducial=None, index=None):
        """ Reads a data point's results from HDF5 file """

        with h5py.File(fileName, 'r')as f:
            R = self.fromHdf(f, systemFilePath, index=index, fiducial=fiducial)

        self.plotMe = True
        return self


    def read_fromH5Obj(self, h5obj, fName, grpName, systemFilepath = ''):
        """ Reads a data points results from HDF5 file """
        grp = h5obj.get(grpName)
        assert not grp is None, "ID "+str(grpName) + " does not exist in file " + fName
        self.fromHdf(grp, systemFilepath)


    def fromHdf(self, hdfFile, systemFilePath, index=None, fiducial=None):

        iNone = index is None
        fNone = fiducial is None

        assert not (iNone and fNone) ^ (not iNone and not fNone), Exception("Must specify either an index OR a fiducial.")

        fiducials = StatArray.StatArray().fromHdf(hdfFile['fiducials'])

        if not fNone:
            index = fiducials.searchsorted(fiducial)

        s = np.s_[index, :]

        self.fiducial = np.float64(fiducials[index])

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
        # self.doi = hdfRead.readKeyFromFile(hdfFile,'','/','doi', index=index)
        self.multiplier = hdfRead.readKeyFromFile(hdfFile,'','/','multiplier', index=index)
        self.rate = hdfRead.readKeyFromFile(hdfFile,'','/','rate', index=s)
        self.PhiDs = hdfRead.readKeyFromFile(hdfFile,'','/','phids', index=s)

        self.bestDataPoint = hdfRead.readKeyFromFile(hdfFile,'','/','bestd', index=index, systemFilepath=systemFilePath)
        self.currentDataPoint = hdfRead.readKeyFromFile(hdfFile,'','/','currentdatapoint', index=index, systemFilepath=systemFilePath)
        # except:
        #     self.currentDataPoint = self.bestDataPoint
        #     p = hdfRead.readKeyFromFile(hdfFile,'','/','dzhist', index=index)
        #     self.currentDataPoint.z.setPosterior(p)


        # try:
        self.currentModel = hdfRead.readKeyFromFile(hdfFile,'','/','currentmodel', index=index)
        self.Hitmap = self.currentModel.par.posterior
        self.currentModel.maxDepth = np.log(self.Hitmap.y.cellCentres[-1])
        # except:
        #     self.Hitmap = hdfRead.readKeyFromFile(hdfFile,'','/','hitmap', index=index)


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

        self.plotMe = True

        return self

