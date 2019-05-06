""" @FdemSystem_Class
Module describing a frequency domain EM acquisition system
"""
from copy import deepcopy
from ...classes.core.myObject import myObject
import numpy as np
from ...classes.core import StatArray
from .CircularLoop import CircularLoop
from ...base import fileIO as fIO
from ...base import MPI as myMPI
from ...base.customFunctions import safeEval
#from ...base import Error as Err


class FdemSystem(myObject):
    """ Defines a Frequency Domain ElectroMagnetic acquisition system """

    def __init__(self, nFrequencies=0, systemFilename=None):
        """ Initialize an FdemSystem """

        if (not systemFilename is None):
            assert (isinstance(systemFilename, str)), TypeError("systemFilename must a file to read the Fdem system information from")
            self.read(systemFilename)
            return

        # Number of Frequencies
        self._nFrequencies = np.int64(nFrequencies)
        # StatArray of frequencies
        self._frequencies = StatArray.StatArray(self.nFrequencies, "Frequencies", "Hz")
        # StatArray of Transmitter loops
        self.T = StatArray.StatArray(self.nFrequencies, "Transmitter Loops", dtype=CircularLoop)
        # StatArray of Reciever loops
        self.R = StatArray.StatArray(self.nFrequencies, "Reciever Loops", dtype=CircularLoop)
        # StatArray of Loop Separations
        self.dist = StatArray.StatArray(self.nFrequencies, "Loop Separations", "m")
        # Instantiate the circularLoops
        for i in range(self.nFrequencies):
            self.T[i] = CircularLoop()
            self.R[i] = CircularLoop()

        self._fileName = systemFilename


    @property
    def frequencies(self):
        return self._frequencies

    @property
    def  nFrequencies(self):
        return self._nFrequencies


    def deepcopy(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        tmp = FdemSystem(nFrequencies = self._nFrequencies)
        tmp._frequencies[:] = self._frequencies
        tmp.dist[:] = self.dist
        for i in range(self._nFrequencies):
            tmp.T[i] = self.T[i].deepcopy()
            tmp.R[i] = self.R[i].deepcopy()
        return tmp


    def read(self, fileName):
        """ Read in a file containing the system information
        
        The system file is structured using columns with the first line containing header information
        Each subsequent row contains the information for each measurement frequency
        freq  tor  tmom  tx ty tz ror rmom  rx   ry rz
        378   z    1     0  0  0  z   1     7.93 0  0
        1776  z    1     0  0  0  z   1     7.91 0  0
        ...

        where tor and ror are the orientations of the transmitter/reciever loops [x or z].
        tmom and rmom are the moments of the loops.
        t/rx,y,z are the loop offsets from the observation locations in the data file.

        """

        self.__init__(fIO.getNlines(fileName, 1))
        with open(fileName) as f:
            tmp = f.readline().lower().split()

            assert 'freq' in tmp, ('Cannot read headers from FdemSystem File ' + fileName +
                                   '\nFirst line of system file should contain\nfreq tor tmom  tx ty tzoff ror rmom  rx   ry rzoff')
            for j, line in enumerate(f):  # For each line in the file
                line = fIO.parseString(line)
                self._frequencies[j] = np.float64(line[0])
                self.T[j] = CircularLoop(
                        line[1], np.int(
                        line[2]), np.int(
                        line[3]), np.float64(
                        line[4]), np.float64(
                        line[5]))
                self.R[j] = CircularLoop(
                        line[6], np.int(
                        line[7]), np.float64(
                        line[8]), np.float64(
                        line[9]), np.float64(
                        line[10]))
                self.dist[j] = np.sqrt(np.power((self.T[j].x - self.R[j].x), 2.0) + np.power((self.T[j].y - self.R[j].y), 2.0))
                # except Exception:
                #     raise SystemExit(
                #         "Could not read from system file:" + fileName + " Line:" + str(j + 2))
        self.fileName = fileName


    def fileInformation(self):
        """Description of the system file."""
        tmp = "The system file is structured using columns with the first line containing header information \n"
        "Each subsequent row contains the information for each measurement frequency \n"
        "freq  tor  tmom  tx ty tz ror rmom  rx   ry rz \n"
        "378   z    1     0  0  0  z   1     7.93 0  0 \n"
        "1776  z    1     0  0  0  z   1     7.91 0  0 \n"
        "... \n"
        "\n"
        "where tor and ror are the orientations of the transmitter/reciever loops [x or z]. \n"
        "tmom and rmom are the moments of the loops. \n"
        "t/rx,y,z are the loop offsets from the observation locations in the data file. \n"
        return tmp


    def getTensorID(self):
        """ For each coil orientation pair, adds the index of the frequency to the appropriate list
        e.g. two coils at the i$^{th}$ frequency with 'x' as their orientation cause i to be added to the 'xx' list."""
        tid = np.zeros(self._nFrequencies, dtype=np.int32)
        for i in range(self._nFrequencies):
            if ((self.T[i].orient == 'x') and self.R[i].orient == 'x'):
                tid[i] = 1
            if ((self.T[i].orient == 'x') and self.R[i].orient == 'y'):
                tid[i] = 2
            if ((self.T[i].orient == 'x') and self.R[i].orient == 'z'):
                tid[i] = 3
            if ((self.T[i].orient == 'y') and self.R[i].orient == 'x'):
                tid[i] = 4
            if ((self.T[i].orient == 'y') and self.R[i].orient == 'y'):
                tid[i] = 5
            if ((self.T[i].orient == 'y') and self.R[i].orient == 'z'):
                tid[i] = 6
            if ((self.T[i].orient == 'z') and self.R[i].orient == 'x'):
                tid[i] = 7
            if ((self.T[i].orient == 'z') and self.R[i].orient == 'y'):
                tid[i] = 8
            if ((self.T[i].orient == 'z') and self.R[i].orient == 'z'):
                tid[i] = 9
        return tid

    def getComponentID(self):
        """ For each coil orientation pair, adds the index of the frequency to the appropriate list
        e.g. two coils at the i$^{th}$ frequency with 'x' as their orientation cause i to be added to the 'xx' list."""
        xx, xy, xz, yx, yy, yz, zx, zy, zz = ([] for i in range(9))
        for i in range(self._nFrequencies):
            if ((self.T[i].orient == 'x') and self.R[i].orient == 'x'):
                xx.append(i)
            if ((self.T[i].orient == 'x') and self.R[i].orient == 'y'):
                xy.append(i)
            if ((self.T[i].orient == 'x') and self.R[i].orient == 'z'):
                xz.append(i)
            if ((self.T[i].orient == 'y') and self.R[i].orient == 'x'):
                yx.append(i)
            if ((self.T[i].orient == 'y') and self.R[i].orient == 'y'):
                yy.append(i)
            if ((self.T[i].orient == 'y') and self.R[i].orient == 'z'):
                yz.append(i)
            if ((self.T[i].orient == 'z') and self.R[i].orient == 'x'):
                zx.append(i)
            if ((self.T[i].orient == 'z') and self.R[i].orient == 'y'):
                zy.append(i)
            if ((self.T[i].orient == 'z') and self.R[i].orient == 'z'):
                zz.append(i)
        return np.asarray(xx), np.asarray(xy), np.asarray(xz), np.asarray(yx), np.asarray(yy), np.asarray(yz), np.asarray(zx), np.asarray(zy), np.asarray(zz)

    def summary(self, out=False):
        """ print a summary of the FdemSystem """
        msg = ("FdemSystem: \n"
               "{} \n"
               "{} \n").format(self.fileName, self._frequencies.summary(True))
        return msg if out else print(msg)

    def hdfName(self):
        return('FdemSystem(0)')

    def toHdf(self, h5obj, myName):
        """ Write the object to a HDF file """
        # Create a new group inside h5obj
        grp = h5obj.create_group(myName)
        grp.attrs["repr"] = self.hdfName()
        grp.create_dataset('nFreq', data=self._nFrequencies)
        self._frequencies.toHdf(grp, 'freq')
        self.dist.toHdf(grp, 'dist')
        T = grp.create_group('T')
        R = grp.create_group('R')
        for i in range(self._nFrequencies):
            self.T[i].toHdf(T, 'T' + str(i))
            self.R[i].toHdf(R, 'R' + str(i))

    def fromHdf(self, grp):
        """ Reads the object from a HDF file """
        nFreq = np.int(np.array(grp.get('nFreq')))
        tmp = FdemSystem(nFreq)
        item = grp.get('freq')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp._frequencies = obj.fromHdf(item)
        item = grp.get('dist')
        obj = eval(safeEval(item.attrs.get('repr')))
        tmp.dist = obj.fromHdf(item)
        for i in range(nFreq):
            item = grp.get('T/T' + str(i))
            tmp.T[i] = eval(safeEval(item.attrs.get('repr')))
            item = grp.get('R/R' + str(i))
            tmp.R[i] = eval(safeEval(item.attrs.get('repr')))
        return tmp

    def Bcast(self, world, root=0):
        """ Broadcast the FdemSystem using MPI """
        nFreq = myMPI.Bcast(self._nFrequencies, world, root=root)
        if (world.rank > 0):
            self = FdemSystem(nFreq)
        this = FdemSystem(nFreq)
        this._frequencies = self._frequencies.Bcast(world, root=root)
        this.dist = self.dist.Bcast(world, root=root)
        for i in range(self._nFrequencies):
            this.T[i] = self.T[i].Bcast(world, root=root)
            this.R[i] = self.R[i].Bcast(world, root=root)
        return this


    def Isend(self, dest, world):
        myMPI.Isend(self._nFrequencies, dest=dest, world=world)
        self._frequencies.Isend(dest=dest, world=world)
        self.dist.Isend(dest=dest, world=world)
        for i in range(self._nFrequencies):
            self.T[i].Isend(dest=dest, world=world)
            self.R[i].Isend(dest=dest, world=world)


    def Irecv(self, source, world):
        nFreq = myMPI.Irecv(source=source, world=world)
        out = FdemSystem(nFreq)
        out._frequencies = out._frequencies.Irecv(source=source, world=world)
        out.dist = out.dist.Irecv(source=source, world=world)
        for i in range(nFreq):
            out.T[i] = out.T[i].Irecv(source=source, world=world)
            out.R[i] = out.R[i].Irecv(source=source, world=world)
        return out













