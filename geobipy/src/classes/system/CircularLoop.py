from copy import deepcopy
import numpy as np
from ...base import MPI as myMPI
from ...base.HDF.hdfWrite import writeNumpy
from .EmLoop import EmLoop

class CircularLoop(EmLoop):
    """Defines a circular loop for EM acquisition systems

    CircularLoop('z', 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

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
    radius : float
        Radius of the loop

    """

    def __init__(self, orient="z", moment=0, x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0):
        """ Initialize a loop in an EM system """
        # Orientation of the loop dipole
        self.orient = orient
        # Dipole moment of the loop
        self.moment = np.int32(moment)
        self.data = np.zeros(6,dtype=np.float64) + [x, y, z, pitch, roll, yaw]
        # Not sure yet
        self.x = self.data[0]
        # Not sure yet
        self.y = self.data[1]
        # Not sure yet
        self.z = self.data[2]
        # Pitch of the loop
        self.pitch = self.data[3]
        # Roll of the loop
        self.roll = self.data[4]
        # Yaw of the loop
        self.yaw = self.data[5]
        # Radius of the loop
        self.radius = radius

    
    def deepcopy(self):
        return deepcopy(self)


    def __deepcopy__(self, memo):
        return CircularLoop(self.orient, self.moment, self.x, self.y, self.z, self.pitch, self.roll, self.yaw, self.radius)


    def summary(self):
        """Print a summary"""
        print("EmLoop:")
        print("Orientation: :" + str(self.orient))
        print("Moment:      :" + str(self.moment))
        print("X:          :" + str(self.x))
        print("Y:          :" + str(self.y))
        print("Z:         :" + str(self.z))
        print("Pitch:       :" + str(self.pitch))
        print("Roll:        :" + str(self.roll))
        print("Yaw:         :" + str(self.yaw))
        print("Radius:      :" + str(self.radius))

    def hdfName(self):
        """Create a reproducibility string that can be instantiated from a hdf file """
        return 'CircularLoop()'

    def createHdf(self, parent, myName, nRepeats=None, fillvalue=None):
        """ Create the hdf group metadata in file
        parent: HDF object to create a group inside
        myName: Name of the group
        """
        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()

        if (not nRepeats is None):
            grp.create_dataset('orientation', [nRepeats], dtype="S1")
            grp.create_dataset('moment', [nRepeats], dtype=np.int32, fillvalue=fillvalue)
            grp.create_dataset('data', [nRepeats, 6], dtype=np.float64, fillvalue=fillvalue)
            grp.create_dataset('radius', [nRepeats], dtype=np.float64, fillvalue=fillvalue)
        else:
            grp.create_dataset('orientation', [1], dtype="S1")
            grp.create_dataset('moment', [1], dtype=np.int32, fillvalue=fillvalue)
            grp.create_dataset('data', [6], dtype=np.float64, fillvalue=fillvalue)
            grp.create_dataset('radius', [1], dtype=np.float64, fillvalue=fillvalue)


    def writeHdf(self, parent, myName, create=True, index=None):
        """ Write the StatArray to an HDF object
        parent: Upper hdf file or group
        myName: object hdf name. Assumes createHdf has already been called
        create: optionally create the data set as well before writing
        """
        # create a new group inside h5obj
        if (create):
            self.createHdf(parent, myName)

        if (index is None):
            parent[myName+'/orientation'][0]=np.string_(self.orient)
        else:
            parent[myName+'/orientation'][index]=np.string_(self.orient)
        writeNumpy(self.moment, parent, myName+'/moment', index=index)
        writeNumpy(self.data, parent, myName+'/data', index=index)
        writeNumpy(self.radius, parent, myName+'/radius', index=index)

    def toHdf(self, parent, myName):

        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()

        grp.create_dataset('orientation', data = np.string_(self.orient))
        grp.create_dataset('moment', data = self.moment)
        grp.create_dataset('data', data=self.data)
        grp.create_dataset('radius', data=self.radius)


    def fromHdf(self, h5grp, index=None):
        """ Reads in the object from a HDF file """
        if (index is None):
            try:
                o = np.array(h5grp.get('orientation'))
                m = np.array(h5grp.get('moment'))
                d = np.array(h5grp.get('data'))
                try:
                    r = np.array(h5grp.get('radius')[index])
                except:
                    r = 1.0
            except:
                assert False, ValueError("HDF data was created as a larger array, specify the row index to read from")
        else:
            o = np.array(h5grp.get('orientation')[index])
            m = np.array(h5grp.get('moment')[index])
            d = np.array(h5grp.get('data')[index])
            try:
                r = np.array(h5grp.get('radius')[index])
            except:
                r = 1.0

        return CircularLoop(o, m, d[0], d[1], d[2], d[3], d[4], d[5], r)


    def Bcast(self, world, root=0):
        """Broadcast using MPI
        
        Parameters
        ----------
        world : mpi4py.MPI.COMM_WORLD
            An MPI communicator

        Returns
        -------
        out : CircularLoop
            A CircularLoop on each core
        
        """
        o = myMPI.Bcast(self.orient, world, root=root)
        m = myMPI.Bcast(self.moment, world, root=root)
        x = myMPI.Bcast(self.x, world, root=root)
        y = myMPI.Bcast(self.y, world, root=root)
        z = myMPI.Bcast(self.z, world, root=root)
        pitch = myMPI.Bcast(self.pitch, world, root=root)
        roll = myMPI.Bcast(self.roll, world, root=root)
        yaw = myMPI.Bcast(self.yaw, world, root=root)
        radius = myMPI.Bcast(self.radius, world, root=root)
        return CircularLoop(o, m, x, y, z, pitch, roll, yaw, radius)

    def __str__(self):
        """ Define print(self) """
        return 'CircularLoop("{0}",{1},{2},{3},{4},{5},{6},{7},{8})'.format(
            self.orient,self.moment,self.x,self.y,self.z,self.pitch,self.roll,self.yaw,self.radius)
