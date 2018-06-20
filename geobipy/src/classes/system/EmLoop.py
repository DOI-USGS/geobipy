""" @EmLoop
Module describing a transmitter or reciever coil in an EM system
"""
import numpy as np
from ...base import MPI as myMPI
from ...classes.core.myObject import myObject
from ...base.HDF.hdfWrite import writeNumpy

class EmLoop(myObject):
    """ Defines a loop in an EM system i.e. transmitter or reciever
    orient: : Orientation of the loop dipole
    moment: : ??Magnetic?? Moment of the loop
    tx:     : ?? Not sure yet
    ty:     : ?? Not sure yet
    off:    : Loop Offset
    """

    def __init__(self, orient="N/A", moment=0, x=0.0, y=0.0, zoff=0.0, pitch=0.0, roll=0.0, yaw=0.0):
        """ Initialize a loop in an EM system """
        # Orientation of the loop dipole
        self.orient = orient
        # Dipole moment of the loop
        self.moment = np.int32(moment)
        self.data = np.zeros(6,dtype=np.float64)
        self.data[:] = [x,y,zoff,pitch,roll,yaw]
        # Not sure yet
        self.tx = self.data[0]
        # Not sure yet
        self.ty = self.data[1]
        # Not sure yet
        self.off = self.data[2]
        # Pitch of the loop
        self.pitch = self.data[3]
        # Roll of the loop
        self.roll = self.data[4]
        # Yaw of the loop
        self.yaw = self.data[5]

    def summary(self):
        """ Print a summary of the EmLoop """
        print("EmLoop:")
        print("Orientation: :" + str(self.orient))
        print("Moment:      :" + str(self.moment))
        print("Tx:          :" + str(self.tx))
        print("Ty:          :" + str(self.ty))
        print("Off:         :" + str(self.off))
        print("Pitch:       :" + str(self.pitch))
        print("Roll:        :" + str(self.roll))
        print("Yaw:         :" + str(self.yaw))

    def hdfName(self):
        """ Create a reproducibility string that can be instantiated from a hdf file """
        return 'EmLoop()'

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
        else:
            grp.create_dataset('orientation', [1], dtype="S1")
            grp.create_dataset('moment', [1], dtype=np.int32, fillvalue=fillvalue)
            grp.create_dataset('data', [6], dtype=np.float64, fillvalue=fillvalue)

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

    def toHdf(self, parent, myName):

        # create a new group inside h5obj
        grp = parent.create_group(myName)
        grp.attrs["repr"] = self.hdfName()

        grp.create_dataset('orientation', data = np.string_(self.orient))
        grp.create_dataset('moment', data = self.moment)
        grp.create_dataset('data', data=self.data)


    def fromHdf(self, h5grp, index=None):
        """ Reads in the object from a HDF file """
        if (index is None):
            try:
                o = np.array(h5grp.get('orientation'))
                m = np.array(h5grp.get('moment'))
                d = np.array(h5grp.get('data'))
            except:
                assert False, ValueError("HDF data was created as a larger array, specify the row index to read from")
        else:
            o = np.array(h5grp.get('orientation')[index])
            m = np.array(h5grp.get('moment')[index])
            d = np.array(h5grp.get('data')[index])

        return EmLoop(o,m,d[0],d[1],d[2],d[3],d[4],d[5])


    def Bcast(self, world):
        """ Broadcast an EM loop using MPI """
        o = myMPI.Bcast(self.orient, world)
        m = myMPI.Bcast(self.moment, world)
        tx = myMPI.Bcast(self.tx, world)
        ty = myMPI.Bcast(self.ty, world)
        off = myMPI.Bcast(self.off, world)
        pitch = myMPI.Bcast(self.pitch, world)
        roll = myMPI.Bcast(self.roll, world)
        yaw = myMPI.Bcast(self.yaw, world)
        return EmLoop(o,m,tx,ty,off,pitch,roll,yaw)

    def __str__(self):
        """ Define print(self) """
        return 'EmLoop("{0}",{1},{2},{3},{4},{5},{6},{7})'.format(
            self.orient,self.moment,self.tx,self.ty,self.off,self.pitch,self.roll,self.yaw)
