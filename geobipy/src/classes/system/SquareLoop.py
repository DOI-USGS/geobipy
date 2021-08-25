from copy import deepcopy
import numpy as np
from ...base import MPI as myMPI
from ...base.HDF.hdfWrite import write_nd
from .EmLoop import EmLoop

class SquareLoop(EmLoop):
    """Defines a square loop for EM acquisition systems

    SquareLoop(orient, moment, x, y, z, pitch, roll, yaw, sideLength)

    Parameters
    ----------
    orient : str
        Orientation of the loop, 'x' or 'z'
    moment : int
        Moment of the loop
    x : float
        X location of the loop relative to an observation location
    y : float
        Y location of the loop relative to an observation location
    z : float
        Z location of the loop relative to an observation location
    pitch : float
        Pitch of the loop
    roll : float
        Roll of the loop
    yaw : float
        Yaw of the loop
    sideLength : float
        Length of each side of the loop

    """

    def __init__(self, orient="z", moment=1, x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, sideLength=1.0):
        """ Initialize a loop in an EM system """
        # Orientation of the loop dipole
        self._orient = orient
        # Dipole moment of the loop
        self._moment = np.int32(moment)
        self.data = np.hstack([x, y, z, pitch, roll, yaw])
        # Not sure yet
        self._x = self.data[0]
        # Not sure yet
        self._y = self.data[1]
        # Not sure yet
        self._z = self.data[2]
        # Pitch of the loop
        self._pitch = self.data[3]
        # Roll of the loop
        self._roll = self.data[4]
        # Yaw of the loop
        self._yaw = self.data[5]
        # Radius of the loop
        self._sideLength = sideLength

    @property
    def area(self):
        return self.sideLength * self.sideLength

    @property
    def moment(self):
        return self._moment

    @property
    def orient(self):
        return self._orient

    @property
    def pitch(self):
        return self._pitch

    @property
    def roll(self):
        return self._roll

    @property
    def sideLength(self):
        return self._sideLength

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def yaw(self):
        return self._yaw

    @property
    def z(self):
        return self._z


    def deepcopy(self):
        return deepcopy(self)


    def __deepcopy__(self, memo={}):
        return SquareLoop(self.orient, self.moment, self.x, self.y, self.z, self.pitch, self.roll, self.yaw, self.sideLength)


    def summary(self):
        """Print a summary"""
        print("EmLoop:")
        print("Orientation: :" + str(self.orient))
        print("Moment:      :" + str(self.moment))
        print("X:          :"  + str(self.x))
        print("Y:          :"  + str(self.y))
        print("Z:          :"  + str(self.z))
        print("Pitch:       :" + str(self.pitch))
        print("Roll:        :" + str(self.roll))
        print("Yaw:         :" + str(self.yaw))
        print("SideLength:      :" + str(self.sideLength))

    def hdfName(self):
        """Create a reproducibility string that can be instantiated from a hdf file """
        return 'SquareLoop()'

    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()

        if (not nRepeats is None):
            grp.create_dataset('orientation', [nRepeats],    dtype="S1")
            grp.create_dataset('moment',      [nRepeats],    dtype=np.int32,   fillvalue=fillvalue)
            grp.create_dataset('data',        [nRepeats, 6], dtype=np.float64, fillvalue=fillvalue)
            grp.create_dataset('sideLength',      [nRepeats],    dtype=np.float64, fillvalue=fillvalue)
        else:
            grp.create_dataset('orientation', [1], dtype="S1")
            grp.create_dataset('moment',      [1], dtype=np.int32,   fillvalue=fillvalue)
            grp.create_dataset('data',        [6], dtype=np.float64, fillvalue=fillvalue)
            grp.create_dataset('sideLength',      [1], dtype=np.float64, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        if (index is None):
            parent[myName+'/orientation'][0] = np.string_(self.orient)
        else:
            parent[myName+'/orientation'][index] = np.string_(self.orient)
        write_nd(self.moment, parent, myName+'/moment', index=index)
        write_nd(self.data,   parent, myName+'/data',   index=index)
        write_nd(self.sideLength, parent, myName+'/sidelength', index=index)

    def toHdf(self, parent, myName):

        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()

        grp.create_dataset('orientation', data = np.string_(self.orient))
        grp.create_dataset('moment',      data = self.moment)
        grp.create_dataset('data',        data = self.data)
        grp.create_dataset('sidelength',      data = self.sideLength)

    @classmethod
    def fromHdf(cls, h5grp, index=None):
        """ Reads in the object from a HDF file """

        if (index is None):
            try:
                o = np.array(h5grp.get('orientation'))
                m = np.array(h5grp.get('moment'))
                d = np.array(h5grp.get('data'))
                try:
                    r = np.array(h5grp.get('sidelength')[index])
                except:
                    r = 1.0
            except:
                assert False, ValueError("HDF data was created as a larger array, specify the row index to read from")

            return cls(o, m, d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5], r)
        else:
            o = np.array(h5grp.get('orientation')[index])
            m = np.array(h5grp.get('moment')[index])
            d = np.array(h5grp.get('data')[index])
            try:
                r = np.array(h5grp.get('sidelength')[index])
            except:
                r = 1.0
            return cls(o, m, d[0], d[1], d[2], d[3], d[4], d[5], r)


    def Bcast(self, world, root=0):
        """Broadcast using MPI

        Parameters
        ----------
        world : mpi4py.MPI.COMM_WORLD
            An MPI communicator

        Returns
        -------
        out : SquareLoop
            A SquareLoop on each core

        """

        o = myMPI.Bcast(self.orient,      world, root=root)
        m = myMPI.Bcast(self.moment,      world, root=root)
        x = myMPI.Bcast(self.x,           world, root=root)
        y = myMPI.Bcast(self.y,           world, root=root)
        z = myMPI.Bcast(self.z,           world, root=root)
        pitch = myMPI.Bcast(self.pitch,   world, root=root)
        roll = myMPI.Bcast(self.roll,     world, root=root)
        yaw = myMPI.Bcast(self.yaw,       world, root=root)
        sideLength = myMPI.Bcast(self.sideLength, world, root=root)

        return SquareLoop(o, m, x, y, z, pitch, roll, yaw, sideLength)


    def Isend(self, dest, world):
        req = world.isend(self.orient, dest=dest)
        req.wait()
        myMPI.Isend(self.moment, dest=dest, world=world)
        tmp = np.empty(7, np.float64)
        tmp[:] = np.asarray([self.x, self.y, self.z, self.pitch, self.roll, self.yaw, self.sideLength])
        req = world.Isend(tmp, dest=dest)


    def Irecv(self, source, world):
        req = world.irecv(source=source)
        o = req.wait()
        m = myMPI.Irecv(source, world)
        tmp = np.empty(7, np.float64)
        req = world.Irecv(tmp, source=source)
        req.Wait()
        return SquareLoop(o, m, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6])


    def __str__(self):
        """ Define print(self) """
        return 'SquareLoop("{0}",{1},{2},{3},{4},{5},{6},{7},{8})'.format(
            self.orient, self.moment,
            self.x,      self.y,    self.z,
            self.pitch,  self.roll, self.yaw, self.sideLength)

















