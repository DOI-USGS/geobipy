""" @MTData_Class
Module describing a Magneto-Telluric Data Set where apparent resisitivity and phase are associated with an xyz co-ordinate
"""
from .Data import Data
from ..datapoint.MTDataPoint import MTDataPoint
from ....base import customFunctions as cF
from ....base import customPlots as cP
from ....classes.core.StatArray import StatArray
from ...system.MTSystem import MTSystem
import numpy as np
from ....base import fileIO as fIO
#from ....base import Error as Err
import matplotlib.pyplot as plt


class MTData(Data):
    """ Class defining a magneto-telluric electro magnetic data set"""

    def __init__(self, nPoints=1, nFrequencies=1, systemFname=None):
        """ Initialize the MT data """
        nChannels = 2 * nFrequencies
        # Data Class containing xyz and channel values
        Data.__init__(self, nPoints, nChannels)
        # StatArray of the line number for flight line data
        self.line = StatArray(nPoints, 'Line Number')
        # StatArray of the id number
        self.id = StatArray(nPoints, 'ID Number')
        # StatArray of the elevation
        self.e = StatArray(nPoints, 'Elevation', 'm')
        # Instantiate the system
        self.sys = MTSystem()
        # Assign data names
        self.D.name='MT Data'
        # Assign channel names
        self.names = [None] * nChannels
    
        if (not systemFname is None):
            self.sys.read(systemFname)
            # Set the channel names
            for i in range(2 * self.sys.nFreq):
                self.names[i] = self.getMeasurementType(i) + str(self.getFrequency(i))+' (Hz)'


    @property        
    def apparentResistivity(self):
        """Return the observed apparent resistivities
        
        Returns
        -------
        out : geobipy.StatArray
            Observed apparent resistivity
            
        """
        return StatArray(self.D[:, :self.sys.nFreq], name = 'App Resistivity', units = '$\Omega m$')


    @property        
    def phase(self):
        """Return the observed phase
        
        Returns
        -------
        out : geobipy.StatArray
            Observed Phase
        
        """
        return StatArray(self.D[:, self.sys.nFreq:], name = 'Phase', units = '$^{o}$')

    
    @property        
    def predictedApparentResistivity(self):
        """Return the observed apparent resistivities
        
        Returns
        -------
        out : geobipy.StatArray
            Observed apparent resistivity
            
        """
        return StatArray(self.P[:, :self.sys.nFreq], name = 'App Resistivity', units = '$\Omega m$')


    @property        
    def predictedPhase(self):
        """Return the observed phase
        
        Returns
        -------
        out : geobipy.StatArray
            Observed Phase
        
        """
        return StatArray(self.P[:, self.sys.nFreq:], name = 'Phase', units = '$^{o}$')

    @property
    def nFrequencies(self):
        """The number of frequencies

        Returns
        -------
        out : int
            Number of measurement frequencies
        
        """
        return self.sys.nFreq


    @property
    def frequency(self):
        """The measurement frequencies

        Returns
        -------
        out : geobipy.StatArray
            The frequencies

        """
        return self.sys.freq


    @property
    def noise(self):
        """Generates values from a random normal distribution with a mean of 0.0 and standard deviation self.s
        
        Returns
        -------
        out : array_like
            Noise values of shape [self.nData, 2*self.nFrequencies]

        """
        return np.random.randn(self.Std.shape[0], self.Std.shape[1]) * self.Std
            

    def read(self, dataFname, systemFname):
        """ Read in both the MT data and MT system files
        The data are read in according to the header names.
        """
        # Read in the EM System file
        sys = MTSystem()
        sys.read(systemFname)

        # Get the column headers of the data file
        channels = fIO.getHeaderNames(dataFname)
        nChannels = len(channels)
        # Get the number of points
        nPoints = fIO.getNlines(dataFname, 1)
        # To grab the EM data, skip the following header names. (More can be added to this)
        # Initialize a column identifier for x y z
        tmp = [0, 0, 0]
        i = -1
        for j, line in enumerate(channels):
            line = line.lower()
            if (line in ['e', 'x', 'easting']):
                i += 1
                tmp[0] = i
            elif (line in ['n', 'y','northing']):
                i += 1
                tmp[1] = i
            elif (line in ['alt', 'laser', 'bheight', 'height']):
                i += 1
                tmp[2] = i
            elif(line in ['z', 'dtm','dem_elev', 'elev']):
                i += 1
                iElev = i
            elif(line in ['line']):
                i += 1
                iLine = i
            elif(line in ['id', 'fid']):
                i += 1
                iID = i
        assert i == 5, ('Cannot determine data columns. \n'
                        'Please use a header line and specify the following case insensitive header names \n'
                        'Line [ID or FID] [X or E or easting] [Y or N or northing] [Z or DTM or dem_elev or elev] '
                              '[Alt or Laser or bheight or height] [AR IP] ... [AR IP] \n Do not include brackets []')
        # Initialize column identifiers
        cols = np.zeros(nChannels - i + 2, dtype=int)
        for j, k in enumerate(tmp):
            cols[j] = k
        tmp = range(i + 1, nChannels)
        nData = len(tmp)
        cols[3:] = tmp[:]       

        # Check that the number of channels is even. EM data has two values, inphase and quadrature.
        # Therefore the number of data channels in the file must be even
        assert nData % 2 == 0, "Total number of apparent resistivity + phase channels must be even"

        # Check that the file has errors listed, they may be missing.
        hasErrors = False
        if (nData > 2 * sys.nFreq):
            hasErrors = True
            # If the file has errors, make sure there are enough columns to
            # match the data
            assert (nData == 4 * sys.nFreq), "Number of error columns must 2 times # of Frequencies"

        # Initialize the EMData Class
        self.__init__(nPoints, sys.nFreq)
        self.sys = sys
        # Read in the data, extract the appropriate columns
        tmp = fIO.read_columns(dataFname, cols, 1, nPoints)
        # Assign the co-ordinates
        self.x[:] = tmp[:, 0]
        self.y[:] = tmp[:, 1]
        self.z[:] = tmp[:, 2]
        # EM data columns are in the following order
        # AR1 IP1 AR2 IP2 .... ARN IPN ErrAR1 ErrIP1 ... ErrARN ErrIPN
        # Reshuffle to the following
        # AR1 AR2 ... ARN IP1 IP2 ... IPN and
        # ErrI1 ErrI2 ... ErrIN ErrIP1 ErrIP2 ... ErrIPN
        self.D[:, :self.sys.nFreq] = tmp[:, 3:(2*self.sys.nFreq)+3:2]
        self.D[:, self.sys.nFreq:] = tmp[:, 4:(2*self.sys.nFreq)+3:2]

        if hasErrors:
          self.Std[:, :self.sys.nFreq] = tmp[:, (2*self.sys.nFreq)+3::2]
          self.Std[:, self.sys.nFreq:] = tmp[:, (2*self.sys.nFreq)+4::2]

        # for i in range(self.sys.nFreq):
        #     i1 = (2 * i) + 3
        #     q1 = i1 + 1
        #     self.D[:, i] = tmp[:, i1]
        #     self.D[:, i + self.sys.nFreq] = tmp[:, q1]
        #     if (hasErrors):
        #         ei = i1 + 2 * self.sys.nFreq
        #         eq = q1 + 2 * self.sys.nFreq
        #         self.Std[:, i] = tmp[:, ei]
        #         self.Std[:, i + self.sys.nFreq] = tmp[:, eq]

        # Read the line numbers from the file
        tmp = fIO.read_columns(dataFname, [iLine], 1, nPoints)
        self.line[:] = tmp[:, 0]
        tmp = fIO.read_columns(dataFname, [iID], 1, nPoints)
        self.id[:] = tmp[:, 0]
        tmp = fIO.read_columns(dataFname, [iElev], 1, nPoints)
        self.e[:] = tmp[:, 0]

        # Set the channel names
        for i in range(2 * self.sys.nFreq):
            self.names[i] = self.getMeasurementType(i) + str(self.getFrequency(i))+' (Hz)'

            
    def getNumberActiveData(self):
        """Get the number of active data per data point.

        For each data point, counts the number of channels that are NOT nan.

        Returns
        -------
        out
            StatArray

        """

        return np.sum(~np.isnan(self.D), 1)


    def getChannel(self, channel):
        """ Gets the data in the specified channel """
        assert channel < 2*self.sys.nFreq, 'Requested channel must be less than '+str(2*self.sys.nFreq)

        if (channel < self.sys.nFreq):
            tmp='App Resistivity - Frequency:'
        else:
            tmp='Phase - Degrees:'
        tmp += ' '+str(self.sys.freq[channel%self.sys.nFreq])

        tmp = StatArray(self.D[:, channel], tmp, self.D.units)

        return tmp


    def getMeasurementType(self, channel):
        return 'App Resistivity ' if channel <self.sys.nFreq else 'Phase '


    def getFrequency(self, channel):
        return self.sys.freq[channel%self.sys.nFreq]


    def getLine(self, line):
        """ Gets the data in the given line number """
        i = np.where(self.line == line)[0]
        assert (i.size > 0), 'Could not get line with number '+str(line)
        return self[i]


    def __getitem__(self, i):
        """ Define item getter for Data """
        tmp = MTData(np.size(i), self.nFrequencies)
        tmp.x[:] = self.x[i]
        tmp.y[:] = self.y[i]
        tmp.z[:] = self.z[i]
        tmp.D[:, :] = self.D[i, :]
        tmp.Std[:, :] = self.Std[i, :]
        tmp.line[:] = self.line[i]
        tmp.id[:] = self.id[i]
        tmp.e[:] = self.e[i]
        tmp.sys = self.sys     
        tmp.names = self.names
        return tmp

    
    def getDataPoint(self, i):
        """ Get the ith data point from the data set """
        return MTDataPoint(self.x[i], self.y[i], self.z[i], self.e[i], self.D[i, :], self.Std[i, :], self.sys)


#     def mapChannel(self, channel, *args, **kwargs):
#         """ Create a map of the specified data channel """

#         assert channel < 2*self.sys.nFreq, ValueError('Requested channel must be less than '+str(2*self.sys.nFreq))

#         Data.mapChannel(self, channel, *args, **kwargs)

#         if (channel < self.sys.nFreq):
#             tmp='InPhase - Frequency:'
#         else:
#             tmp='Quadrature - Frequency:'
#         cP.title(tmp+' '+str(self.sys.freq[channel%self.sys.nFreq]))


    def plot(self, channels=None, **kwargs):
        """ Plots the specifed columns as a line plot, if cols is not given, all the columns are plotted """
        i = np.asarray(channels)
        if channels is None:
            i = np.arange(self.nChannels, dtype=np.int)

        i1 = i[i < self.sys.nFreq]
        i2 = i[i >= self.sys.nFreq]

        nPlots = 0
        if (i1.size == 0):
            if (i2.size > 0):
                nPlots += 1
        else:
            nPlots += 1
            if (i2.size > 0):
                nPlots += 1


        ax = plt.gca()
        iPlot = 0

        if i1.size > 0:
            iPlot += 1
            ax = plt.subplot(nPlots, 1, iPlot)
            for j in i1:
                tmp = str(self.getFrequency(j)) + 'Hz'
                self.apparentResistivity[:, j].plot(label=tmp, **kwargs)

            # cP.title('Apparent Resistivity')
            ax.set_xticklabels([])
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
            leg.set_title('Frequency')


        if (i2.size > 0):
            iPlot += 1
            ax = plt.subplot(nPlots, 1, iPlot)
            for j in i2:
                tmp = str(self.getFrequency(j)) + 'Hz'
                self.phase[:, j-self.sys.nFreq].plot(label=tmp, **kwargs)

            # cP.title('Apparent Resistivity')
            ax.set_xticklabels([])
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
            leg.set_title('Phase')

        


    def plotLine(self, line, **kwargs):
        """ Plot the specified line """
        l = self.getLine(line)

        r = kwargs.pop("x", None)
        if (r is None):
            r = StatArray(np.sqrt(l.x**2.0 + l.y**2.0),'Distance',l.x.units)
            r -= r.min()

        log = kwargs.pop('log', None)

        xscale = kwargs.pop('xscale','linear')
        yscale = kwargs.pop('yscale','linear')

        ax = plt.gca()
        ax = plt.subplot(211)

        appRes = l.apparentResistivity

        for i in range(self.sys.nFreq):
            tmp=str(self.getFrequency(i))+' (Hz)'
            dTmp = appRes[:, i]
            dTmp, dum = cF._logSomething(dTmp, log)
            iPositive = np.where(dTmp > 0.0)[0]
            ax.plot(r[iPositive],dTmp[iPositive],label=tmp,**kwargs)
        # cP.title('Apparent Resistivity')
        ax.set_xticklabels([])
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
        leg.set_title('Frequency')

        ylabel = appRes.getNameUnits()
        if (log):
            dum, logLabel = cF._logSomething([10], log)
            ylabel = logLabel + ylabel
        cP.ylabel(ylabel)

        plt.xscale(xscale)
        plt.yscale(yscale)

        ax = plt.subplot(212)

        phase = l.phase

        for i in range(self.sys.nFreq):
            tmp=str(self.getFrequency(i))+' (Hz)'
            dTmp = phase[:,i]
            dTmp, dum = cF._logSomething(dTmp, log)
            iPositive = np.where(dTmp > 0.0)[0]
            ax.plot(r[iPositive],dTmp[iPositive],label=tmp,**kwargs)

        # cP.title('Impedance Phase')
        cP.suptitle("Line number "+str(line))
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        leg=ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
        leg.set_title('Frequency')

        cP.xlabel(cF.getNameUnits(r))

        ylabel = phase.getNameUnits()
        if (log):
            dum, logLabel = cF._logSomething([10], log)
            ylabel = logLabel + ylabel
        cP.ylabel(ylabel)

        plt.xscale(xscale)
        plt.yscale(yscale)
        return ax


    def updateErrors(self, relativeErr, additiveErr):
        """Updates the data errors

        Updates the standard deviation of the data errors using the following model

        .. math::
            \sqrt{(\mathbf{\epsilon}_{rel} \mathbf{d}^{obs})^{2} + \mathbf{\epsilon}^{2}_{add}},
        where :math:`\mathbf{\epsilon}_{rel}` is the relative error, a percentage fraction and :math:`\mathbf{\epsilon}_{add}` is the additive error.
        
        Parameters
        ----------  
        relativeErr : array_like or list of array_like
            A fraction percentage that is multiplied by the observed data.
            If array_like: should have size 2. The first element for apparent resistivity is applied to all frequencies, similarly for the second element for phase.
            If list: should have size 2. Each item should be array like with size self.nFrequencies.  Each array corresponds to the apparent resisitvity and phase relative errors.
        additiveErr : array_like
            An absolute value of additive error. 
            If array_like: should have size 2. The first element for apparent resistivity is applied to all frequencies, similarly for the second element for phase.
            If list: should have size 2. Each item should be array like with size self.nFrequencies.  Each array corresponds to the apparent resisitvity and phase additive errors.

        Raises
        ------
        TypeError
            If the length of relativeErr or additiveErr is not equal to 2
        TypeError
            If any item in the relativeErr or additiveErr lists is not a scalar or array_like of length equal to the number of frequencies
        ValueError
            If any relative or additive errors are <= 0.0

        """    

        assert (np.size(relativeErr) == 2), TypeError("relativeErr must have size equal to 2.")
        assert (np.size(additiveErr) == 2), TypeError("additiveErr must have size equal to 2.")

        # For each system assign error levels using the user inputs
        assert np.all([rel > 0.0 for rel in relativeErr]), ValueError("relativeErr must be > 0.0")
        assert np.all([add > 0.0 for add in additiveErr]), ValueError("additiveErr must be > 0.0")

        if isinstance(relativeErr, list):
            for rel in relativeErr:
                assert np.size(rel) == self.nFrequencies, ValueError("Each item in relativeErr must have length equal to {}".format(self.nFrequencies))
        if isinstance(additiveErr, list):
            for add in additiveErr:
                assert np.size(add) == self.nFrequencies, ValueError("Each item in additiveErr must have length equal to {}".format(self.nFrequencies))

        self.Std[:, :self.sys.nFreq] = np.sqrt((relativeErr[0] * self.D[:, :self.sys.nFreq])**2.0 + additiveErr[0]**2.0)
        self.Std[:, self.sys.nFreq:] = np.sqrt((relativeErr[1] * self.D[:, self.sys.nFreq:])**2.0 + additiveErr[1]**2.0)


    def write(self, filename, channels=None, predictedData=True, withStandardDeviation=True):
        """Write the MTData to file

        Parameters
        ----------
        filename : str
            Name of the file to write data to.
            If extension is 'txt' will write an ascii file
        predictedData : bool, optional
            If True:  Writes the predicted data to the data columns
            If False: Writes the observed data to the data columns
        withStandardDeviation : bool, optional
            If True: Writes the standard deviations after the data

        """

        ext = fIO.getFileExtension(filename).lower()
        assert ext in ('txt'), ValueError("filename must have extension ['txt']")
        if ext == 'txt':
            self.write_ascii(filename, predictedData, withStandardDeviation)


    def write_ascii(self, filename, predictedData=True, withStandardDeviation=True, **kwargs):
        """Write the MTData to an ascii file

        Parameters
        ----------
        filename : str
            Name of the file to write data to.
            If extension is 'txt' will write an ascii file
        predictedData : bool
            If True:  Writes the predicted data to the data columns
            If False: Writes the observed data to the data columns
        withStandardDeviation : bool
            If True: Writes the standard deviations after the data

        """
        with open(filename, 'w') as f:
            header = "Line   fid   Easting   Northing   Height   Elev   "
            for freq in self.frequency:
                header += "AR_{}   IP_{}   ".format(freq, freq)

            i1 = 6 + self.nFrequencies
            i2 = 6 + (2 * self.nFrequencies)

            nCols = i2
            if withStandardDeviation:
                for freq in self.frequency:
                    header += "StdAR_{}   StdIP_{}   ".format(freq, freq)
                nCols += 2 * self.nFrequencies                    

            f.write(header+'\n')

            tmp = np.zeros(nCols)
            for i in range(self.N):
                if (predictedData):
                    d = self.P
                else:
                    d = self.d

                tmp[:6] = [self.line[i], self.id[i], self.x[i], self.y[i], self.z[i], self.e[i]]
                tmp[6:i2:2] = d[i, :self.nFrequencies]
                tmp[7:i2:2] = d[i, self.nFrequencies:]
                if (withStandardDeviation):
                    tmp[i2::2] = self.Std[i, :self.nFrequencies]
                    tmp[i2+1::2] = self.Std[i, self.nFrequencies:]

                f.write(np.array2string(tmp, edgeitems=1e10, max_line_width=1e10)[1:-1]+"\n")




#     def Bcast(self, world):
#         """ Broadcast the FdemData using MPI """
#         dat = None
#         dat = Data.Bcast(self, world)
#         this = FdemData(dat.N, dat.nChannels)
#         this.x = dat.x
#         this.y = dat.y
#         this.z = dat.z
#         this.D = dat.D
#         this.Std = dat.Std
#         this.id = self.id.Bcast(world)
#         this.line = self.line.Bcast(world)
#         this.e = self.e.Bcast(world)
#         this.sys = self.sys.Bcast(world)
#         return this

#     def Scatterv(self, myStart, myChunk, world):
#         """ Scatterv the FdemData using MPI """
# #    print(world.rank,' FdemData.Scatterv')
#         dat = None
#         dat = Data.Scatterv(self, myStart, myChunk, world)
# #    print(world.rank,' After Data.Scatterv')
# #    print(world.rank,dat.D)
#         this = FdemData(dat.N, dat.nChannels)
#         this.x = dat.x
#         this.y = dat.y
#         this.z = dat.z
#         this.D = dat.D
#         this.Std = dat.Std
#         this.id = self.id.Scatterv(myStart, myChunk, world)
#         this.line = self.line.Scatterv(myStart, myChunk, world)
#         this.e = self.e.Scatterv(myStart, myChunk, world)
#         this.sys = self.sys.Bcast(world)
#         return this
