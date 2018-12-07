""" @EMinversion1D_MCMC_Results
Class to store EMinv1D inversion results. Contains plotting and writing to file procedures
"""
from os.path import join
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import pause
from ..base import customPlots as cP
from ..base.customFunctions import safeEval
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
    """ Define the results handler for the MCMC Inversion """

    def __init__(self, saveMe=False,plotMe=True,savePNG=False,paras=None,D=None,M=None,ID=0, verbose=False):
        """ Initialize the results of the inversion """

        # Initialize a stopwatch to keep track of time
        self.clk = Stopwatch()
        self.invTime = np.float64(0.0)
        self.saveTime = np.float64(0.0)

        # Logicals of whether to plot or save
        self.saveMe = saveMe
        self.plotMe = plotMe
        self.savePNG = savePNG
        # Return none if important parameters are not used (used for hdf 5)
        if all(x1 is None for x1 in [paras, D, M]):
            return

        if (not self.plotMe and not self.saveMe):
            print('Warning! You chosen to neither view or save the inversion results!')
            return
        # Set the ID for the data point the results pertain to
        # Data Point identifier
        self.ID = np.float64(ID)
        # Set the increment at which to plot results
        # Increment at which to update the results
        self.iPlot = np.int64(paras.iPlot)
        # Set the display limits of the parameter in the HitMap
        # Display limits for parameters
        self.limits = np.zeros(2,dtype=np.float64)+paras.dispLimits
        # Should we plot resistivity or Conductivity?
        # Logical whether to take the reciprocal of the parameters
        self.invertPar = paras.invertPar
        # Set the screen resolution
        # TODO: Need to make this automatic
        # Screen Size
        self.sx = np.int32(1920)
        self.sy = np.int32(1080)
        # Copy the number of systems
        # Number of systems in the DataPoint
        self.nSystems = np.int32(D.nSystems)
        # Copy the number of Markov Chains
        # Number of Markov Chains to use
        self.nMC = np.int64(paras.nMC)
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
        self.rate = StatArray(n)
        self.ratex = StatArray(np.arange(1, n + 1) * 1000)
        # Initialize the burned in state
        self.iBurn = self.nMC
        self.burnedIn = False
        # Initialize the index for the best model
        self.iBest = 0
        self.iBestV = StatArray(2*self.nMC, name='Iteration of best model')

        # Initialize the number of layers for the histogram
        self.kHist = Histogram1D(bins=np.arange(0.0,paras.maxLayers + 1.5),name="# of Layers")
        # Initialize the histograms for the relative and Additive Errors
        rBins = D.relErr.prior.getBins()
        aBins = D.addErr.prior.getBins()

        self.relErr = []
        self.addErr = []
        if (self.nSystems > 1):
            for i in range(self.nSystems):
                self.relErr.append(Histogram1D(bins=rBins[i,:],name='$\epsilon_{Relative}x10^{2}$',units='%'))
                self.addErr.append(Histogram1D(bins=aBins[i,:],name='$log_{10} \epsilon_{Additive}$',units=D.d.units))
        else:
            self.relErr.append(Histogram1D(bins=rBins,name='$\epsilon_{Relative}x10^{2}$',units='%'))
            self.addErr.append(Histogram1D(bins=aBins,name='$log_{10} \epsilon_{Additive}$',units=D.d.units))

        # Initialize the hit map of layers and conductivities
        zGrd = StatArray(np.arange(0.5 * np.exp(M.minDepth), 1.1 * np.exp(M.maxDepth), 0.5 * np.exp(M.minThickness)), M.depth.name, M.depth.units)

        mGrd = StatArray(np.logspace(np.log10(np.exp(paras.priMu -3.0 * paras.priStd)),
                           np.log10(np.exp(paras.priMu + 3.0 * paras.priStd)),250), 'Conductivity','$Sm^{-1}$')

        self.iz = np.arange(zGrd.size)

        self.Hitmap = Hitmap2D(x=mGrd, y=zGrd)

        # Initialize the doi
        self.doi = self.Hitmap.y[0]
#    self.Hori=Rmesh2D([zGrd.size,mGrd.size],'','',dtype=np.int32)

        self.meanInterp = StatArray(zGrd.size)
        self.bestInterp = StatArray(zGrd.size)
#        self.opacityInterp = StatArray(zGrd.size)

        # Initialize the Elevation Histogram
        self.DzHist = Histogram1D(bins=D.z.prior.getBins(), name=D.z.name, units=D.z.units)

        # Initialize the Model Depth Histogram
        self.MzHist = Histogram1D(bins=zGrd)

        # Set a tag to catch data points that are not minimizing
        self.zeroCount = 0

        # Initialize the figure region
        self.initFigure()

        # Initialize times in seconds
        self.invTime = np.float64(0.0)
        self.saveTime = np.float64(0.0)

        # Initialize the best data, current data and best model
        self.bestD = D
        self.currentD = D
        self.bestModel = M


        self.verbose = verbose
        if verbose:
            self.allRelErr = StatArray([self.nSystems,self.nMC], name = '$\epsilon_{Relative}x10^{2}$',units='%')
            self.allAddErr = StatArray([self.nSystems,self.nMC], name = '$log_{10} \epsilon_{Additive}$',units=D.d.units)
            self.allZ = StatArray(self.nMC, name = 'Height', units='m')
            self.posterior = StatArray(self.nMC, name = 'log(posterior)')
            self.posteriorComponents = StatArray([9,self.nMC], 'Components of the posterior')


#         Initialize and save the first figure
#        if self.savePNG:
#            figName = 'PNG/_tmp' + \
#                fIO.getFileNameInteger(self.i, np.int(np.log10(self.nMC))) + '.png'
#            plt.savefig(figName)

    def update(self, i, iBest, bestD, bestModel, D, multiplier, PhiD, Mod, posterior, posteriorComponents, clipRatio):
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
            self.kHist.update(Mod.nCells[0])
            self.DzHist.update(D.z[0])
            for j in range(self.nSystems):
                self.relErr[j].update(D.relErr[j])
                self.addErr[j].update(D.addErr[j])

            Mod.addToHitMap(self.Hitmap)

            # Update the layer interface histogram
            if (Mod.nCells > 1):
                ratio = np.exp(np.diff(np.log(Mod.par)))
                m1 = ratio <= 1.0 - clipRatio
                m2 = ratio >= 1.0 + clipRatio
                #keep = np.ma.mask_or(m1, m2)
                keep = np.logical_not(np.ma.masked_invalid(ratio).mask) & np.ma.mask_or(m1,m2)
                tmp = Mod.depth[:-1]
                if (len(tmp) > 0):
                    self.MzHist.update(tmp[keep])

            if (self.verbose):
                iTmp = self.i - self.iBurn
                for j in range(self.nSystems):
                    self.allRelErr[j,iTmp]=D.relErr[j]
                    self.allAddErr[j,iTmp]=D.addErr[j]
                self.posterior[iTmp] = np.log(posterior)
                self.allZ[iTmp] = D.z[0]
                self.posteriorComponents[:,iTmp] = posteriorComponents


        if (np.mod(i, 1000) == 0):
            ratePercent = 100.0 * (np.float64(self.acceptance) / np.float64(1000))
            self.rate[np.int(self.i / 1000) - 1] = ratePercent
            self.acceptance = 0
            if (ratePercent < 2.0):
                self.zeroCount += 1
            else:
                self.zeroCount = 0

        self.bestD = bestD
        self.currentD = D
        self.bestModel = bestModel


    def initFigure(self, iFig=0, forcePlot=False):
        """ Initialize the plotting region """
        if (not self.plotMe and not forcePlot):
            return
        # Setup the figure region. The figure window is split into a 4x3
        # region. Columns are able to span multiple rows
        self.fig = plt.figure(iFig, facecolor='white', figsize=(10,7))
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
#        plt.show()
#        plt.draw()

    def _plotAcceptanceVsIteration(self, **kwargs):
        """ Plots the acceptance percentage against iteration. """


        m = kwargs.pop('marker', 'o')
        a = kwargs.pop('alpha', 0.7)
        ls = kwargs.pop('linestyle', 'none')
        mec = kwargs.pop('markeredgecolor', 'k')

        self.rate.plot(self.ratex, i=np.s_[:np.int64(self.i / 1000)], marker=m, markeredgecolor=mec, linestyle=ls, **kwargs)
        cP.xlabel('Iteration #')
        cP.ylabel('% Acceptance')
        cP.title('Rate of model acceptance per 1000 iterations')


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
        dum = self.multiplier * len(self.currentD.iActive)
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
        plt.axvline(self.bestD.addErr[system], color=cP.wellSeparated[3], linestyle='dashed', linewidth=3)


    def _plotLayerDepthPosterior(self, **kwargs):
        """ Plot the histogram of layer interface depths """

        r = kwargs.pop('rotate', True)
        fY = kwargs.pop('flipY', True)
        tr = kwargs.pop('trim', False)

        self.MzHist.plot(rotate=r, flipY=fY, trim=tr, **kwargs)

    
    def _plotHitmapPosterior(self, confidenceInterval = 95.0, opacityPercentage = 67.0, **kwargs):
        """ Plot the hitmap posterior of conductivity with depth """

        # Get the mean and 95% confidence intervals
        (sigMed, sigLow, sigHigh) = self.Hitmap.getConfidenceIntervals(confidenceInterval)

        if (self.invertPar):
            x = 1.0 / self.Hitmap.x
            sl = 1.0 / sigLow
            sh = 1.0 / sigHigh
            xlabel = 'Resistivity ($\Omega m$)'
        else:
            x = self.Hitmap.x
            sl = sigLow
            sh = sigHigh
            xlabel = 'Conductivity ($Sm^{-1}$)'

        plt.pcolor(x, self.Hitmap.y, self.Hitmap.arr, cmap=mpl.cm.Greys)
        plt.plot(sl, self.Hitmap.y, color='#5046C8', linestyle='dashed', linewidth=2, alpha=0.6)
        plt.plot(sh, self.Hitmap.y, color='#5046C8', linestyle='dashed', linewidth=2, alpha=0.6)
        cP.xlabel(xlabel)

        # Plot the DOI cutoff based on percentage variance
        self.doi = self.Hitmap.getOpacityLevel(opacityPercentage)
        plt.axhline(self.doi, color='#5046C8', linestyle='dashed', linewidth=3)

        # Plot the best model
        self.bestModel.plot(flipY=False, invX=True, noLabels=True)
        plt.axis([self.limits[0], self.limits[1], self.Hitmap.y[0], self.Hitmap.y[-1]])
        ax = plt.gca()
        lim = ax.get_ylim()
        if (lim[1] > lim[0]):
            ax.set_ylim(lim[::-1])
        cP.ylabel(self.MzHist.bins.getNameUnits())
        plt.xscale('log')


    def plot(self, title="", iFig=0, forcePlot=False):
        """ Updates the figures for MCMC Inversion """
        # Plots that change with every iteration
        if (not self.plotMe and not forcePlot):
            return

        if (not hasattr(self, 'gs')):
            self.initFigure(iFig, forcePlot=forcePlot)


        plt.figure(iFig)

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
                    cP.title('System ' + str(system + 1))

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

        pause(1e-10)



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

        nz=self.Hitmap.y.size
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
        self.currentD.createHdf(grp, 'currentd')
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
        self.currentD.writeHdf(grp, 'currentd')
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
        # Write the current Data
        self.currentD.writeHdf(grp, 'currentd')
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
        obj = eval(safeEval(item.attrs.get('repr')))
        self.rate = obj.fromHdf(item)

        item = grp.get('ratex')
        obj = eval(safeEval(item.attrs.get('repr')))
        self.ratex = obj.fromHdf(item)

        item = grp.get('phids')
        if (item is None):
            item = grp.get('PhiDs')
        obj = eval(safeEval(item.attrs.get('repr')))
        self.PhiDs = obj.fromHdf(item)

        item = grp.get('khist')
        if (item is None):
            item = grp.get('kHist')
        obj = eval(safeEval(item.attrs.get('repr')))
        self.kHist = obj.fromHdf(item)

        item = grp.get('dzhist')
        if (item is None):
            item = grp.get('DzHist')
        obj = eval(safeEval(item.attrs.get('repr')))
        self.DzHist = obj.fromHdf(item)

        item = grp.get('mzhist')
        if (item is None):
            item = grp.get('MzHist')
        obj = eval(safeEval(item.attrs.get('repr')))
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
        obj = eval(safeEval(item.attrs.get('repr')))
        self.bestD = obj.fromHdf(item, sysPath=sysPath)

        item = grp.get('bestmodel')
        if (item is None):
            item = grp.get('bestModel')
        obj = eval(safeEval(item.attrs.get('repr')))
        self.bestModel = obj.fromHdf(item)
        self.bestModel.maxDepth = np.log(self.Hitmap.y[-1])

        item = grp.get('currentd')
        if (item is None):
            item = grp.get('currentD')
        obj = eval(safeEval(item.attrs.get('repr')))
        self.currentD = obj.fromHdf(item, sysPath=sysPath)
        self.relErr = []
        self.addErr = []
        for i in range(self.nSystems):
            item = grp.get('relerr' + str(i))
            if (item is None):
                item = grp.get('relErr'+str(i))
            obj = eval(safeEval(item.attrs.get('repr')))
            aHist = obj.fromHdf(item)
            self.relErr.append(aHist)
            item = grp.get('adderr' + str(i))
            if (item is None):
                item = grp.get('addErr'+str(i))
            obj = eval(safeEval(item.attrs.get('repr')))
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
