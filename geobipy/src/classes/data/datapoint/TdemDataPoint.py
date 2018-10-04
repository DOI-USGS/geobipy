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
                if (s[i] is None):
                    self.s[i0:i1] = 0.1 * d[i] 
                else:
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
    
    def times(self, system=0):
        """ Return the window times in an StatArray """
        return StatArray(self.sys[system].windows.centre, name='Time', units='s')


    def getIplotActive(self):
        """ Get the active data indices per system.  Used for plotting. """
        self.iplotActive=[]
        i0 = 0
        for i in range(self.nSystems):
            i1 = i0 + self.sys[i].nwindows()
            self.iplotActive.append(findNotNans(self.d[i0:i1]))
            i0 = i1

    def dualMoment(self):
        """ Returns True if the number of systems is > 1 """
        return len(self.sys) > 1


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

    def plotPredicted(self,title='Time Domain EM Data',**kwargs):

        noLabels = kwargs.pop('nolabels', False)

        if (not noLabels):
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




    # def scaleJ(self, Jin, power=1.0):
    #     """ Scales a matrix by the errors in the given data
    #     Useful if a sensitivity matrix is generated using one data point, but must be scaled by the errors in another """
    #     J1 = np.zeros(Jin.shape)
    #     J1[:, :] = Jin * (np.repeat(self.s[self.iActive, np.newaxis]**-power, np.size(J1, 1), 1))
    #     return J1


    def updateErrors(self, relativeErr, additiveErr):
        """ Updates the data errors

        Assumes a t^-0.5 behaviour e.g. logarithmic gate averaging
        V0 is assumed to be ln(Error @ 1ms)

        Parameters
        ----------  
        relativeErr : list of scalars or list of array_like
            A fraction percentage that is multiplied by the observed data. The list should have length equal to the number of systems. The entries in each item can be scalar or array_like.
        additiveErr : list of scalars or list of array_like
            An absolute value of additive error. The list should have length equal to the number of systems. The entries in each item can be scalar or array_like.

        Raises
        ------
        TypeError
            If relativeErr or additiveErr is not a list
        TypeError
            If the length of relativeErr or additiveErr is not equal to the number of systems
        TypeError
            If any item in the relativeErr or additiveErr lists is not a scalar or array_like of length equal to the number of time channels
        ValueError
            If any relative or additive errors are <= 0.0
        """    

        #assert (isinstance(relativeErr, list)), TypeError("relativeErr must be a list of size equal to the number of systems {}".format(self.nSystems))
        assert (len(relativeErr) == self.nSystems), TypeError("relativeErr must be a list of size equal to the number of systems {}".format(self.nSystems))

        #assert (isinstance(additiveErr, list)), TypeError("additiveErr must be a list of size equal to the number of systems {}".format(self.nSystems))
        assert (len(relativeErr) == self.nSystems), TypeError("additiveErr must be a list of size equal to the number of systems {}".format(self.nSystems))

        t0 = 0.5 * np.log(1e-3)  # Assign fixed t0 at 1ms
        i0 = 0
        # For each system assign error levels using the user inputs
        for i in range(self.nSystems):
            assert (isinstance(relativeErr[i], float) or isinstance(relativeErr[i], np.ndarray)), TypeError("relativeErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.sys[i].nwindows()))
            assert (isinstance(additiveErr[i], float) or isinstance(additiveErr[i], np.ndarray)), TypeError("additiveErr for system {} must be a float or have size equal to the number of channels {}".format(i+1, self.sys[i].nwindows()))
            assert (np.all(relativeErr[i] > 0.0)), ValueError("relativeErr for system {} cannot contain values <= 0.0.".format(self.nSystems))
            #assert (np.all(additiveErr[i] > 0.0)), ValueError("additiveErr for system {} cannot contain values <= 0.0.".format(self.nSystems))
            i1 = i0 + self.sys[i].nwindows()
            
            # Compute the relative error
            rErr = relativeErr[i] * self.d[i0:i1]
            aErr = np.exp(additiveErr[i]*np.log(10.0) - 0.5 * np.log(self.sys[i].windows.centre) + t0)

            self.s[i0:i1] = np.sqrt((rErr**2.0) + (aErr**2.0))
            i0 = i1

        # Update the variance of the predicted data prior
        if self.p.hasPrior():
            self.p.prior.variance[:] = self.s[self.iActive]**2.0


    def updateSensitivity(self, J, mod, option, scale=False):
        """ Compute an updated sensitivity matrix using a new model based on an existing matrix """
        J1 = np.zeros([np.size(self.iActive), mod.nCells[0]])

        if(option == 3):  # Do Nothing!
            J1[:, :] = J[:, :]
            return J1

        if (option == 0):  # Created a layer
            J1[:, :mod.iLayer] = J[:, :mod.iLayer]
            J1[:, mod.iLayer + 2:] = J[:, mod.iLayer + 1:]
            tmp = self.sensitivity(mod, ix=[mod.iLayer, mod.iLayer + 1], scale=scale, modelChanged=True)
            J1[:, mod.iLayer:mod.iLayer + 2] = tmp
            return J1

        if(option == 1):  # Deleted a layer
            J1[:, :mod.iLayer] = J[:, :mod.iLayer]
            J1[:, mod.iLayer + 1:] = J[:, mod.iLayer + 2:]
            tmp = self.sensitivity(mod, ix=[mod.iLayer], scale=scale, modelChanged=True)
            J1[:, mod.iLayer] = tmp[:, 0]
            return J1

        if(option == 2):  # Perturbed a layer
            J1[:, :mod.iLayer] = J[:, :mod.iLayer]
            J1[:, mod.iLayer + 1:] = J[:, mod.iLayer + 1:]
            tmp = self.sensitivity(mod, ix=[mod.iLayer], scale=scale, modelChanged=True)
            J1[:, mod.iLayer] = tmp[:, 0]
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
        heightTolerance = 0.0
        if (self.z > heightTolerance):
            self._BrodieForward(mod)
        else:
            self._simPEGForward(mod)

    def _BrodieForward(self, mod):
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

    def _simPEGForward(self, mod):
        
        from SimPEG import Maps
        from simpegEM1D import (EM1DSurveyTD, EM1D, set_mesh_1d)

        mesh1D = set_mesh_1d(mod.depth)
        expmap = Maps.ExpMap(mesh1D)
        prob = EM1D(mesh1D, sigmaMap = expmap, chi = mod.chim)

        if (self.dualMoment()):

            print(self.sys[0].loopRadius(), self.sys[0].peakCurrent())

            simPEG_survey = EM1DSurveyTD(
                rx_location=np.array([0., 0., 0.]),
                src_location=np.array([0., 0., 0.]),
                topo=np.r_[0., 0., 0.],
                depth=-mod.depth,
                rx_type='dBzdt',
                wave_type='general',
                src_type='CircularLoop',
                a=self.sys[0].loopRadius(),
                I=self.sys[0].peakCurrent(),
                time=self.sys[0].windows.centre,
                time_input_currents=self.sys[0].waveform.transmitterTime,
                input_currents=self.sys[0].waveform.transmitterCurrent,
                n_pulse=2,
                base_frequency=self.sys[0].baseFrequency(),
                use_lowpass_filter=True,
                high_cut_frequency=450000,
                moment_type='dual',
                time_dual_moment=self.sys[1].windows.centre,
                time_input_currents_dual_moment=self.sys[1].waveform.transmitterTime,
                input_currents_dual_moment=self.sys[1].waveform.transmitterCurrent,
                base_frequency_dual_moment=self.sys[1].baseFrequency(),
            )
        else:

            simPEG_survey = EM1DSurveyTD(
                rx_location=np.array([0., 0., 0.]),
                src_location=np.array([0., 0., 0.]),
                topo=np.r_[0., 0., 0.],
                depth=-mod.depth,
                rx_type='dBzdt',
                wave_type='general',
                src_type='CircularLoop',
                a=self.sys[0].loopRadius(),
                I=self.sys[0].peakCurrent(),
                time=self.sys[0].windows.centre,
                time_input_currents=self.sys[0].waveform.transmitterTime,
                input_currents=self.sys[0].waveform.transmitterCurrent,
                n_pulse=1,
                base_frequency=self.sys[0].baseFrequency(),
                use_lowpass_filter=True,
                high_cut_frequency=7e4,
                moment_type='single',
            )

        prob.pair(simPEG_survey)
            
        self.p[:] = -simPEG_survey.dpred(mod.par)



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

#     def fm_dlogc(self, mod):
#             # Generate the Brodie Earth class
#         E = Earth(mod.par[:], mod.thk[:-1])
# #    ####lg.debug('Parameters: '+str(mod.par))
# #    ####lg.debug('Thickness: '+str(mod.thk[:-1]))
#         # Generate the Brodie Geometry class
# #    ####lg.debug('Geometry: '+str([self.z[0],self.T.roll,self.T.pitch,self.T.yaw,-12.64,0.0,2.11,self.R.roll,self.R.pitch,self.R.yaw]))
#         G = Geometry(self.z[0], self.T.roll, self.T.pitch, self.T.yaw, -
#                      12.615, 0.0, 2.16, self.R.roll, self.R.pitch, self.R.yaw)
#         # Forward model the data for each system
#         iJ0 = 0
#         for i in range(self.nSystems):
#             iJ1 = iJ0 + self.sys[i].nwindows()
#             dummy = self.sys[i].forward(G, E)

# #      self.p[iJ0:iJ1]=-fm.SZ[:]
# #      iJ0=iJ1
#         return fm

