from copy import deepcopy
from ....classes.core.StatArray import StatArray
from ...model.Model import Model
from .EmDataPoint import EmDataPoint
from ...system.EmLoop import EmLoop
from ....base.logging import myLogger

try:
    from gatdaem1d import Earth
    from gatdaem1d import Geometry
    from gatdaem1d import TDAEMSystem
except:
    h=("Could not find a Time Domain forward modeller. \n"
       "Please see the package's README for instructions on how to install one")
    print(Warning(h))
import matplotlib.pyplot as plt
import numpy as np
#from ....base import Error as Err
from ....base import fileIO as fIO
from ....base import customFunctions as cf
from ....base.customFunctions import safeEval
from ....base.customFunctions import findNotNans, isInt
from ....base import customPlots as cp
from os.path import split as psplt
from os.path import join


class TdemDataPoint(EmDataPoint):
    """ Initialize a Time domain EMData Point


    TdemDataPoint(x, y, z, e=0.0, d=None, s=None, sys=None, T=None, R=None)

    Parameters
    ----------
    x : np.float64
        The easting co-ordinate of the data point
    y : np.float64
        The northing co-ordinate of the data point
    z : np.float64
        The height of the data point above ground
    e : np.float64, optional
        The elevation of the data point, default is 0.0
    d : list of arrays, optional
        A list of 1D arrays, where each array contains the data in each system.
        The arrays are vertically concatenated inside the TdemDataPoint object
    s : list of arrays, optional
        A list of 1D arrays, where each array contains the errors in each system.
        The arrays are vertically concatenated inside the TdemDataPoint object
    sys : TdemSystem, optional
        Time domain system class
    T : EmLoop, optional
        Transmitter loop class
    R : EmLoop, optional
        Receiver loop class

    Returns
    -------
    out : TdemDataPoint
        A time domain EM sounding

    Notes
    -----
    The data incoming in d is a set of lists with length equal to the number of systems.  These data are unpacked and vertically concatenated in this class. The parameter self.d will have length equal to the sum of the number of time gates in each system.  The same is true for the errors, and the predicted data vector.



    """

    def __init__(self, x, y, z, e=0.0, d=None, s=None, sys=None, T=None, R=None):
        if not T is None:
            assert isinstance(T, EmLoop), TypeError("Transmitter must be of type EmLoop")
        if not R is None:
            assert isinstance(R, EmLoop), TypeError("Receiver must be of type EmLoop")
        if not sys is None:
            for i in range(len(sys)):
                assert isinstance(sys[i], TDAEMSystem), TypeError("System "+str(i)+" must be of type TDAEMSystem")
        # x coordinate
        self.x = StatArray(1) + x
        # y coordinate
        self.y = StatArray(1) + y
        # z coordinate
        self.z = StatArray(1, 'Measurement height above ground', 'm') + z
        # Elevation of data point
        self.e = StatArray(1) + e
        if (sys is None):
            return
        # EMSystem Class
        self.sys = sys
#        if (not sys is None):
        # Number of systems
        self.nSystems = np.int32(np.size(sys))
        # Total number of windows
        self.nWindows = np.int32(sum(s.nwindows() for s in self.sys))
        # StatArray of Data
        self.d = StatArray(self.nWindows, 'Time Domain Data', r'$\frac{V}{Am^{4}}$')
        # StatArray of Standard Deviations
        self.s = StatArray(np.ones(self.nWindows), 'Standard Deviation', r'$\frac{V}{Am^{4}}$')
        # Transfer input to class
        self.iplotActive = []
        if (not d is None):
            i0 = 0
            for i in range(self.nSystems):
                i1 = i0 + self.sys[i].nwindows()
                self.d[i0:i1] = d[i]
                self.s[i0:i1] = s[i]
                i0 = i1
                self.iplotActive.append(findNotNans(d[i]))
#        else:
#            self.nSystems = 0
#            self.nWindows = 0

        # EmLoop Transnmitter
        self.T = deepcopy(T)
        # EmLoop Reciever
        self.R = deepcopy(R)
        # StatArray of Predicted Data
        self.p = StatArray(self.nWindows, 'Predicted Data', r'$\frac{V}{Am^{4}}$')
        # StatArray of Relative Errors
        self.relErr = StatArray(self.nSystems, '$\epsilon_{Relative}x10^{2}$','%')
        # StatArray of Additive Errors
        self.addErr = StatArray(self.nSystems, '$\epsilon_{Additive}$',self.d.units)
        # StatArray of calibration parameters
        # The four columns are Bias,Variance,InphaseBias,QuadratureBias.
#    self.calibration=StatArray([sys.nFreq*4],'Calibration Parameters')
        # Sensitivity Matrix - Stored here so we can update only the necessary
        # parts
        self.J = None
        # Index to non NaN values
        self.iActive = self.getActiveData()

    def deepcopy(self):
        """ Define a deepcopy routine """
        return deepcopy(self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        items = self.__dict__.items()
        for k, v in items:
            if (k != 'sys'):
                setattr(result, k, deepcopy(v, memo))
        result.sys = self.sys
        return result
    
    
    def getChannels(self, system=0):
        """ Return the frequencies in an StatArray """
        return StatArray(self.sys[system].windows.centre, name='Time', units='s')


    def getIplotActive(self):
        """ Get the active data indices per system.  Used for plotting. """
        self.iplotActive=[]
        i0 = 0
        for i in range(self.nSystems):
            i1 = i0 + self.sys[i].nwindows()
            self.iplotActive.append(findNotNans(self.d[i0:i1]))
            i0 = i1


    def hdfName(self):
        """ Reprodicibility procedure """
        return('TdemDataPoint(0.0,0.0,0.0,0.0)')

    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        grp.create_dataset('nSystems', data=self.nSystems)
        for i in range(self.nSystems):
            grp.create_dataset('System' + str(i), data=np.string_(psplt(self.sys[i].sysFname)[-1]))
        self.x.createHdf(grp, 'x', nRepeats=nRepeats, fillvalue=fillvalue)
        self.y.createHdf(grp, 'y', nRepeats=nRepeats, fillvalue=fillvalue)
        self.z.createHdf(grp, 'z', nRepeats=nRepeats, fillvalue=fillvalue)
        self.e.createHdf(grp, 'e', nRepeats=nRepeats, fillvalue=fillvalue)
        self.d.createHdf(grp, 'd', nRepeats=nRepeats, fillvalue=fillvalue)
        self.s.createHdf(grp, 's', nRepeats=nRepeats, fillvalue=fillvalue)
        self.p.createHdf(grp, 'p', nRepeats=nRepeats, fillvalue=fillvalue)
        self.relErr.createHdf(grp, 'relErr', nRepeats=nRepeats, fillvalue=fillvalue)
        self.addErr.createHdf(grp, 'addErr', nRepeats=nRepeats, fillvalue=fillvalue)
        self.T.createHdf(grp, 'T', nRepeats=nRepeats, fillvalue=fillvalue)
        self.R.createHdf(grp, 'R', nRepeats=nRepeats, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """

        if (not index is None):
            assert isInt(index), TypeError('Index must be an int')

        grp = parent.get(myName)

        self.x.writeHdf(grp, 'x',  index=index)
        self.y.writeHdf(grp, 'y',  index=index)
        self.z.writeHdf(grp, 'z',  index=index)
        self.e.writeHdf(grp, 'e',  index=index)
        self.d.writeHdf(grp, 'd',  index=index)
        self.s.writeHdf(grp, 's',  index=index)
        self.p.writeHdf(grp, 'p',  index=index)
        self.relErr.writeHdf(grp, 'relErr',  index=index)
        self.addErr.writeHdf(grp, 'addErr',  index=index)
        self.T.writeHdf(grp, 'T',  index=index)
        self.R.writeHdf(grp, 'R',  index=index)
        #writeNumpy(self.iActive, grp, 'iActive')

#    def toHdf(self, parent, myName):
#        """ Write the TdemDataPoint to an HDF object
#        h5obj: :An HDF File or Group Object.
#        """
#        self.writeHdf(parent, myName, index=np.s_[0])

    def fromHdf(self, grp, index=None, **kwargs):
        """ Reads the object from a HDF group """

        assert ('sysPath' in kwargs), ValueError("missing 1 required argument 'sysPath', the path to directory containing system files")

        sysPath = kwargs.pop('sysPath', None)
        assert (not sysPath is None), ValueError("missing 1 required argument 'sysPath', the path to directory containing system files")
        if (not index is None):
            assert isInt(index), ValueError("index must be of type int")

        item = grp.get('x')
        obj = eval(safeEval(item.attrs.get('repr')))
        x = obj.fromHdf(item, index=index)
        item = grp.get('y')
        obj = eval(safeEval(item.attrs.get('repr')))
        y = obj.fromHdf(item, index=index)
        item = grp.get('z')
        obj = eval(safeEval(item.attrs.get('repr')))
        z = obj.fromHdf(item, index=index)
        item = grp.get('e')
        obj = eval(safeEval(item.attrs.get('repr')))
        e = obj.fromHdf(item, index=index)

        _aPoint = TdemDataPoint(x, y, z, e)

        _aPoint.nSystems = np.int(np.asarray(grp.get('nSystems')))

        slic = None
        if (not index is None):
            slic = np.s_[index,:]

        item = grp.get('d')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.d = obj.fromHdf(item, index=slic)
        item = grp.get('s')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.s = obj.fromHdf(item, index=slic)
        item = grp.get('p')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.p = obj.fromHdf(item, index=slic)
        if (_aPoint.nSystems == 1):
            slic = index
        item = grp.get('relErr')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.relErr = obj.fromHdf(item, index=slic)
        item = grp.get('addErr')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.addErr = obj.fromHdf(item, index=slic)
        item = grp.get('T')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.T = obj.fromHdf(item, index=index)
        item = grp.get('R')
        obj = eval(safeEval(item.attrs.get('repr')))
        _aPoint.R = obj.fromHdf(item, index=index)


        _aPoint.sys = np.ndarray(_aPoint.nSystems, dtype=TDAEMSystem)
        _aPoint.iActive = _aPoint.getActiveData()

        for i in range(_aPoint.nSystems):
            # Get the system file name. h5py has to encode strings using utf-8, so decode it!
            sysFname = str(np.asarray(grp.get('System'+str(i))), 'utf-8')
            # Get the system file
            fName = join(sysPath,sysFname)
            # Check that the file exists, rBodies class does not handle errors
            assert fIO.fileExists(fName), 'Could not open file: ' + fName
            _aPoint.sys[i] = TDAEMSystem(fName)
            _aPoint.sys[i].sysFname = sysFname

        _aPoint.getIplotActive()

        return _aPoint


#  def calibrate(self,Predicted=True):
#    """ Apply calibration factors to the data point """
#    # Make complex numbers from the data
#    if (Predicted):
#      tmp=cf.mergeComplex(self.p)
#    else:
#      tmp=cf.mergeComplex(self.d)
#
#    # Get the calibration factors for each frequency
#    i1=0;i2=self.sys.nFreq
#    G=self.calibration[i1:i2];i1+=self.sys.nFreq;i2+=self.sys.nFreq
#    Phi=self.calibration[i1:i2];i1+=self.sys.nFreq;i2+=self.sys.nFreq
#    Bi=self.calibration[i1:i2];i1+=self.sys.nFreq;i2+=self.sys.nFreq
#    Bq=self.calibration[i1:i2]
#
#    # Calibrate the data
#    tmp[:]=G*np.exp(1j*Phi)*tmp+Bi+(1j*Bq)
#
#    # Split the complex numbers back out
#    if (Predicted):
#      self.p[:]=cf.splitComplex(tmp)
#    else:
#      self.d[:]=cf.splitComplex(tmp)
#

    def plotWaveform(self,**kwargs):
        for i in range(self.nSystems):
            if (self.nSystems > 1):
                plt.subplot(2, 1, i + 1)
            plt.plot(self.sys[i].waveform.time, self.sys[i].waveform.current, **kwargs)
            cp.xlabel('Time (s)')
            cp.ylabel('Normalized Current (A)')
            plt.margins(0.1, 0.1)

    def plot(self, title='Time Domain EM Data', withErr=True, **kwargs):
        """ Plot the Inphase and Quadrature Data for an EM measurement
        """
        ax=plt.gca()

        m = kwargs.pop('marker','v')
        ms = kwargs.pop('markersize',7)
        c = kwargs.pop('color',[cp.wellSeparated[i+1] for i in range(self.nSystems)])
        mfc = kwargs.pop('markerfacecolor',[cp.wellSeparated[i+1] for i in range(self.nSystems)])
        assert len(c) == self.nSystems, ValueError("color must be a list of length "+str(self.nSystems))
        assert len(mfc) == self.nSystems, ValueError("markerfacecolor must be a list of length "+str(self.nSystems))
        mec = kwargs.pop('markeredgecolor','k')
        mew = kwargs.pop('markeredgewidth',1.0)
        a = kwargs.pop('alpha',0.8)
        ls = kwargs.pop('linestyle','none')
        lw = kwargs.pop('linewidth',2)

        xscale = kwargs.pop('xscale','log')
        yscale = kwargs.pop('yscale','log')

        iJ0 = 0
        for j in range(self.nSystems):
            iAct = self.iplotActive[j]
            iJ1 = iJ0 + iAct
            if (withErr):
                plt.errorbar(self.sys[j].windows.centre[iAct],self.d[iJ1],yerr=self.s[iJ1],
                marker=m,
                markersize=ms,
                color=c[j],
                markerfacecolor=mfc[j],
                markeredgecolor=mec,
                markeredgewidth=mew,
                alpha=a,
                linestyle=ls,
                linewidth=lw,
                label='System: '+str(j+1),
                **kwargs)
            else:
                plt.plot(self.sys[j].windows.centre[iAct],self.d[iJ1],
                marker=m,
                markersize=ms,
                markerfacecolor=mfc[j],
                markeredgecolor=mec,
                markeredgewidth=mew,
                alpha=a,
                linestyle=ls,
                linewidth=lw,
                label='System: '+str(j+1),
                **kwargs)
            iJ0 += self.sys[j].nwindows()


        plt.xscale(xscale)
        plt.yscale(yscale)
        cp.xlabel('Time (s)')
        cp.ylabel(cf.getNameUnits(self.d))
        cp.title(title)
        plt.legend()

        return ax

    def plotPredicted(self,title='',**kwargs):
        cp.xlabel('Time (s)')
        cp.ylabel(cf.getNameUnits(self.p))
        cp.title(title)
        c = kwargs.pop('color', cp.wellSeparated[3])
        lw = kwargs.pop('linewidth', 2)
        a = kwargs.pop('alpha', 0.7)
        xscale = kwargs.pop('xscale', 'log')
        yscale = kwargs.pop('yscale', 'log')
        iJ0 = 0
        for i in range(self.nSystems):
            iAct = self.iplotActive[i]
            iJ1 = iJ0 + iAct
            plt.plot(self.sys[i].windows.centre[iAct],  self.p[iJ1], color=c, linewidth=lw, alpha=a, **kwargs)
            iJ0 += self.sys[i].nwindows()

        plt.xscale(xscale)
        plt.yscale(yscale)


    def plotDataResidual(self, title='', **kwargs):
        cp.xlabel('Time (s)')
        cp.ylabel('Data Residual ('+cf.getUnits(self.d)+')')
        cp.title(title)
        lw = kwargs.pop('linewidth',2)
        a = kwargs.pop('alpha',0.7)

        xscale = kwargs.pop('xscale','log')
        yscale = kwargs.pop('yscale','log')

        iJ0 = 0
        for i in range(self.nSystems):
            iAct = self.iplotActive[i]
            iJ1 = iJ0 + iAct
            plt.plot(self.sys[i].windows.centre[iAct],  np.abs(self.d[iJ1] - self.p[iJ1]), color=cp.wellSeparated[i+1], linewidth=lw, alpha=a, **kwargs)
            iJ0 += self.sys[i].nwindows()

        plt.xscale(xscale)
        plt.yscale(yscale, linthreshy=1e-15)




    def scaleJ(self, Jin, power=1.0):
        """ Scales a matrix by the errors in the given data
        Useful if a sensitivity matrix is generated using one data point, but must be scaled by the errors in another """
        J1 = np.zeros(Jin.shape)
        J1[:, :] = Jin * (np.repeat(self.s[self.iActive, np.newaxis]**-power, np.size(J1, 1), 1))
        return J1

    def updateErrors(self, option, err=None, relativeErr=None, additiveErr=None):
        """ Updates the data errors
        Assumes a t^-0.5 behaviour e.g. logarithmic gate averaging
        V0 is assumed to be ln(Error @ 1ms)
        option:      :[0,1,2,3]
                   0 :Assign err to the data errors
                   1 :Use a percentage of the data with a minimum floor
                   2 :Norm of the percentage and minimum floor
                   3 :Additive Error only
        err:         :Specified Errors (only used with option=0)
        relativeErr: :Percentage of the data for error level assignment for each system [r0,r1,...]
        additiveErr:    :Noise floor minimum per system [emin1,emin2,...] !Assumes this is in log10 space!
        """
        ####lg.myLogger("Global"); ####lg.indent()
        ####lg.info('Updating Errors:')
        assert option >= 0 and option <= 3, "Use an option [0,1,2,3]"
        if (option == 0):
            assert (not err is None), "err must be specified with option = 0"
        if (option > 0):
            # Check the relative and add error inputs
            assert (not relativeErr is None), "relativeError must be specified"
            assert (not additiveErr is None), "additiveErr must be specified"
            assert (not isinstance(relativeErr, float)), "Please enter the relative Error as a list [#1,#2,...], the length must match the number of systems "+str(self.nSystems)
            assert (len(relativeErr) == self.nSystems), "len(relativeError) must equal # of systems"
            assert (len(additiveErr) == self.nSystems), "len(additiveErr) must equal # of systems"
        assert (not self.p.prior is None), 'Please assign a distribution to the predicted data using self.p.setPrior()'

        ####lg.debug('option: '+str(option))
        t0 = 0.5 * np.log(1e-3)  # Assign fixed t0 at 1ms
        i0 = 0
        # For each system assign error levels using the user inputs
        for i in range(self.nSystems):
            #      ####lg.debug('System: '+str(i))
            #      ####lg.debug('Option: '+str(option))
            i1 = i0 + self.sys[i].nwindows()
            if (option > 0):
                # Compute the relative error
                rErr = relativeErr[i] * self.d[i0:i1]
                aErr = np.exp(additiveErr[i]*np.log(10.0) - 0.5 * np.log(self.sys[i].windows.centre) + t0)
#        ####lg.debug('rErr: '+str(rErr))
#        ####lg.debug('aErr: '+str(aErr))

            if (option == 0):
                self.s[i0:i1] = err
            elif (option == 1):
                self.s[i0:i1] = np.maximum(rErr, aErr)
            elif (option == 2):
                self.s[i0:i1] = np.sqrt((rErr**2.0) + (aErr**2.0))
            elif (option == 3):
                self.s[i0:i1] = aErr
#      ####lg.debug('Updated Errors for system: '+str(self.s[i0:i1]))
            i0 = i1

        # Update the variance of the predicted data prior
        self.p.prior.variance[:] = self.s[self.iActive]**2.0
#    ####lg.debug('P Variance: '+str(self.p.prior.variance))
        ####lg.debug(' ')
        ####lg.dedent()

    def updateSensitivity(self, J, mod, option, scale=False):
        """ Compute an updated sensitivity matrix using a new model based on an existing matrix """
        # If there is no matrix saved in the data object, compute the entire thing
        # if (self.J is None):
#    return self.sensitivity(mod,scale=scale,modelChanged=True)

        ####lg.myLogger('Global');####lg.indent()
        ####lg.info('Updating the Sensitivity Matrix')
        ####lg.debug('nWindows: '+str(self.nWindows))
        J1 = np.zeros([np.size(self.iActive), mod.nCells[0]])

        if(option == 3):  # Do Nothing!
            J1[:, :] = J[:, :]
            ####lg.dedent()
            return J1

        if (option == 0):  # Created a layer
            J1[:, :mod.iLayer] = J[:, :mod.iLayer]
            J1[:, mod.iLayer + 2:] = J[:, mod.iLayer + 1:]
            tmp = self.sensitivity(mod, ix=[mod.iLayer, mod.iLayer + 1], scale=scale, modelChanged=True)
            J1[:, mod.iLayer:mod.iLayer + 2] = tmp
            ####lg.dedent()
            return J1

        if(option == 1):  # Deleted a layer
            J1[:, :mod.iLayer] = J[:, :mod.iLayer]
            J1[:, mod.iLayer + 1:] = J[:, mod.iLayer + 2:]
            tmp = self.sensitivity(mod, ix=[mod.iLayer], scale=scale, modelChanged=True)
            J1[:, mod.iLayer] = tmp[:, 0]
            ####lg.dedent()
            return J1

        if(option == 2):  # Perturbed a layer
            J1[:, :mod.iLayer] = J[:, :mod.iLayer]
            J1[:, mod.iLayer + 1:] = J[:, mod.iLayer + 1:]
            tmp = self.sensitivity(mod, ix=[mod.iLayer], scale=scale, modelChanged=True)
            J1[:, mod.iLayer] = tmp[:, 0]
            ####lg.dedent()
            return J1

        assert False, __name__ + '.updateSensitivity: Invalid option [0,1,2]'

    def forward(self, mod):
        """ Forward model the data from the given model """

        assert isinstance(mod, Model), TypeError("Invalid model class for forward modeling [1D]")

        self._forward1D(mod)

    def sensitivity(self, mod, ix=None, scale=False, modelChanged=True):
        """ Compute the sensitivty matrix for the given model """

        assert isinstance(mod, Model), TypeError("Invalid model class for sensitivity matrix [1D]")

        return StatArray(self._sensitivity1D(mod, ix, scale, modelChanged), 'Sensitivity', '$\\frac{V}{ASm^{3}}$')

    def _forward1D(self, mod):
        """ Forward model the data from a 1D layered earth model """
        # Generate the Brodie Earth class
        E = Earth(mod.par[:], mod.thk[:-1])
        # Generate the Brodie Geometry class
        G = Geometry(self.z[0], self.T.roll, self.T.pitch, self.T.yaw, -
                     12.64, 0.0, 2.11, self.R.roll, self.R.pitch, self.R.yaw)
        # Forward model the data for each system
        iJ0 = 0
        for i in range(self.nSystems):
            iJ1 = iJ0 + self.sys[i].nwindows()
            fm = self.sys[i].forwardmodel(G, E)
            self.p[iJ0:iJ1] = -fm.SZ[:]  # Store the necessary component
            iJ0 = iJ1

    def _sensitivity1D(self, mod, ix=None, scale=False, modelChanged=True):
        """ Compute the sensitivty matrix for a 1D layered earth model, optionally compute the responses for only the layers in ix """
        # Unfortunately the code requires forward modelled data to compute the
        # sensitivity if the model has changed since last time
        if modelChanged:
            E = Earth(mod.par[:], mod.thk[:-1])
            G = Geometry(self.z[0], self.T.roll, self.T.pitch, self.T.yaw, -
                         12.64, 0.0, 2.11, self.R.roll, self.R.pitch, self.R.yaw)
            for i in range(self.nSystems):
                self.sys[i].forwardmodel(G, E)
        if (ix is None):  # Generate a full matrix if the layers are not specified
            ix = range(mod.nCells[0])
            J = np.zeros([self.nWindows, mod.nCells[0]])
        else:  # Partial matrix for specified layers
            J = np.zeros([self.nWindows, len(ix)])

        iJ0 = 0
        for j in range(self.nSystems):  # For each system
            iJ1 = iJ0 + self.sys[j].nwindows()
            for i in range(len(ix)):  # For the specified layers
                tmp = self.sys[j].derivative(
                    self.sys[j].CONDUCTIVITYDERIVATIVE, ix[i] + 1)
                # Store the necessary component
                J[iJ0:iJ1, i] = -mod.par[ix[i]] * tmp.SZ[:]
            iJ0 = iJ1

        if scale:
            iJ0 = 0
            for j in range(self.nSystems):  # For each system
                iJ1 = iJ0 + self.sys[j].nwindows()
                for i in range(len(ix)):  # For the specified layers
                    # Scale the sensitivity matix rows by the data weights if
                    # required
                    J[iJ0:iJ1, i] /= self.s[iJ0:iJ1]
                iJ0 = iJ1

        J = J[self.iActive, :]
        return J

    def fm_dlogc(self, mod):
            # Generate the Brodie Earth class
        E = Earth(mod.par[:], mod.thk[:-1])
#    ####lg.debug('Parameters: '+str(mod.par))
#    ####lg.debug('Thickness: '+str(mod.thk[:-1]))
        # Generate the Brodie Geometry class
#    ####lg.debug('Geometry: '+str([self.z[0],self.T.roll,self.T.pitch,self.T.yaw,-12.64,0.0,2.11,self.R.roll,self.R.pitch,self.R.yaw]))
        G = Geometry(self.z[0], self.T.roll, self.T.pitch, self.T.yaw, -
                     12.615, 0.0, 2.16, self.R.roll, self.R.pitch, self.R.yaw)
        # Forward model the data for each system
        iJ0 = 0
        for i in range(self.nSystems):
            iJ1 = iJ0 + self.sys[i].nwindows()
            dummy = self.sys[i].forward(G, E)

#      self.p[iJ0:iJ1]=-fm.SZ[:]
#      iJ0=iJ1
        return fm

